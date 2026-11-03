"""Banco de pruebas del pipeline de voz con los modelos reales.

Sintetiza ordenes con Piper, las hace pasar por el pipeline completo como si
vinieran del microfono (palabra clave -> VAD -> Whisper) y mide:

    - latencia de la transcripcion final: del fin de la frase al texto
    - precision: tasa de error por palabra contra lo que se dijo
    - costo por bloque de la palabra clave y del VAD (presupuesto: 32 ms)
    - tiempo hasta el primer audio del TTS

Las mediciones salen de los eventos del bus (`trace.span` de `@traced` y los
`voice.*`), no de instrumentacion a mano: lo que mide el banco es lo mismo que
vera el HUD.

Uso:

    python scripts/voice_bench.py              # en tiempo real, ~30 s
    python scripts/voice_bench.py --fast       # sin esperar el reloj
    python scripts/voice_bench.py --real-wake  # "Hey Jarvis" de verdad (voz en ingles)

DECISION: por defecto la palabra clave se dispara en un instante programado y
no con el detector real, porque `hey_jarvis` esta entrenado con voces inglesas
y no reconoce a la voz de Piper en espanol. El detector real corre igual en
cada bloque para medir su costo. Con `--real-wake` se antepone "Hey Jarvis"
dicho por una voz inglesa de Piper, y entonces si dispara el detector real.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))

import numpy as np

from jarvis.bus import InMemoryEventBus
from jarvis.config import load_config, project_root
from jarvis.contracts import Event, EventType, SpeechOutput, Utterance
from jarvis.logging import bind_bus, configure_logging
from jarvis.voice import VoiceEngines, VoicePipeline
from jarvis.voice.audio import resample_linear
from jarvis.voice.factory import build_stt, build_tts, build_vad, build_wake
from jarvis.voice.io import ArraySource, MemorySink
from jarvis.voice.models import ModelMissingError, download_piper_voice
from jarvis.voice.text import normalize

if TYPE_CHECKING:
    from collections.abc import Callable

    from jarvis.voice.audio import AudioArray
    from jarvis.voice.engines import TextToSpeech, WakeWordDetector

PHRASES = [
    "Sube el volumen al setenta por ciento.",
    "Abre el navegador y busca el clima de mañana.",
    "Pon música de los Beatles en Spotify.",
    "Cierra todas las ventanas.",
    "Baja el brillo de la pantalla.",
    "Recuérdame llamar a mamá a las cinco.",
]

ENGLISH_VOICE = "en_US-lessac-medium"

#: Metas del documento maestro, para comparar.
TARGET_FINAL_MS = 400.0
FRAME_BUDGET_MS = 32.0


class ScheduledWake:
    """Envuelve al detector real: lo corre para medirlo, pero dispara cuando toca.

    El bloque actual se lee de la fuente y no se cuenta aqui: mientras Jarvis
    escucha, el detector no recibe bloques y un contador propio se desfasaria.
    """

    def __init__(
        self,
        real: WakeWordDetector,
        fire_at_frames: set[int],
        current_frame: Callable[[], int],
    ) -> None:
        self._real = real
        self._fire = fire_at_frames
        self._current = current_frame

    @property
    def keyword(self) -> str:
        return self._real.keyword

    def score(self, frame: AudioArray) -> float:
        self._real.score(frame)
        return 1.0 if self._current() - 1 in self._fire else 0.0

    def reset(self) -> None:
        self._real.reset()


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Tasa de error por palabra, sin mayusculas, acentos ni puntuacion."""

    def words(text: str) -> list[str]:
        cleaned = "".join(c if c.isalnum() or c.isspace() else " " for c in normalize(text))
        return cleaned.split()

    ref, hyp = words(reference), words(hypothesis)
    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        current = [i]
        for j, h in enumerate(hyp, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (r != h)))
        previous = current
    return previous[-1] / max(1, len(ref))


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(fraction * (len(ordered) - 1)))]


def synthesize(tts: TextToSpeech, text: str, rate: int) -> AudioArray:
    return resample_linear(np.concatenate(list(tts.synthesize(text))), tts.sample_rate, rate)


async def run(args: argparse.Namespace) -> dict[str, object]:
    configure_logging(level="WARNING", development=True)
    config = load_config()
    voice = config.voice.model_copy(update={"stt_partials": False})
    rate, frame_ms = voice.sample_rate, voice.frame_ms
    frame_samples = rate * frame_ms // 1000

    print("Cargando modelos ...")
    started = time.perf_counter()
    tts = build_tts(voice)
    stt = build_stt(voice)
    vad = build_vad(voice)
    real_wake = build_wake(voice)
    print(f"  listos en {time.perf_counter() - started:.1f} s  (stt={voice.stt_model})")

    english: TextToSpeech | None = None
    if args.real_wake:
        from jarvis.voice.adapters.piper_tts import PiperTTS

        piper_dir = project_root() / voice.models_dir / "piper"
        if not (piper_dir / f"{ENGLISH_VOICE}.onnx").exists():
            print(f"  bajando la voz {ENGLISH_VOICE} para decir 'Hey Jarvis' ...")
            download_piper_voice(ENGLISH_VOICE, piper_dir)
        english = PiperTTS(
            model_path=piper_dir / f"{ENGLISH_VOICE}.onnx",
            config_path=piper_dir / f"{ENGLISH_VOICE}.onnx.json",
        )

    # Una sola "grabacion" con todas las ordenes, separadas por silencio.
    gap = np.zeros(int(1.6 * rate), dtype=np.float32)
    wake_clip = synthesize(english, "Hey Jarvis.", rate) if english else None
    parts: list[AudioArray] = [gap]
    fire_at: set[int] = set()
    for phrase in PHRASES:
        if wake_clip is not None:
            parts.append(wake_clip)
        else:
            offset = sum(p.shape[0] for p in parts)
            fire_at.add(offset // frame_samples)
        parts.append(synthesize(tts, phrase, rate))
        parts.append(gap)
    stream = np.concatenate(parts)

    source = ArraySource(stream, sample_rate=rate, frame_ms=frame_ms, realtime=not args.fast)
    wake: WakeWordDetector = (
        real_wake if english else ScheduledWake(real_wake, fire_at, lambda: source.frames_delivered)
    )
    bus = InMemoryEventBus()
    events: list[Event] = []

    async def collect(event: Event) -> None:
        events.append(event)

    async with bus:
        bind_bus(bus)
        bus.subscribe("*", collect, name="bench", max_queue=1_000_000)
        sink = MemorySink(realtime=False)
        pipeline = VoicePipeline(
            bus,
            VoiceEngines(source=source, wake=wake, vad=vad, stt=stt, tts=tts, sink=sink),
            voice,
            speak_responses=False,
        )
        audio_s = stream.shape[0] / rate
        mode = "rapido" if args.fast else "tiempo real"
        print(f"Reproduciendo {len(PHRASES)} ordenes ({audio_s:.0f} s de audio, {mode}) ...")
        await pipeline.start()
        await pipeline.source_finished.wait()
        await pipeline.wait_idle(30)

        print("Midiendo el TTS ...")
        speech_results: list[SpeechOutput] = []
        for text in ["Listo.", "Volumen al setenta por ciento.", PHRASES[1]]:
            speech_results.append(await pipeline.speak(text))
        await pipeline.stop()
        await bus.wait_idle(5)
        bind_bus(None)

    return report(events, speech_results, english is not None)


def report(events: list[Event], speech: list[SpeechOutput], real_wake: bool) -> dict[str, object]:
    spans: dict[str, list[float]] = defaultdict(list)
    for event in events:
        if event.type is EventType.TRACE_SPAN:
            spans[str(event.payload["operation"])].append(float(event.payload["duration_ms"]))  # type: ignore[arg-type]

    ended_at = {
        e.correlation_id: e.timestamp for e in events if e.type is EventType.VOICE_SPEECH_ENDED
    }
    finals = [e for e in events if e.type is EventType.VOICE_STT_FINAL]
    wakes = [e for e in events if e.type is EventType.VOICE_WAKE_DETECTED]

    rows = []
    for index, event in enumerate(finals):
        utterance = event.payload_as(Utterance)
        latency = (event.timestamp - ended_at[event.correlation_id]).total_seconds() * 1000
        expected = PHRASES[index] if index < len(PHRASES) else ""
        rows.append(
            {
                "dicho": expected,
                "oido": utterance.text,
                "wer": word_error_rate(expected, utterance.text),
                "latencia_ms": latency,
            }
        )

    print()
    print(f"Palabras clave detectadas: {len(wakes)} de {len(PHRASES)}", end="")
    print("  (detector real)" if real_wake else "  (disparo programado)")
    print()
    print(f"{'latencia':>9}  {'WER':>5}  transcripcion")
    for row in rows:
        mark = "ok" if row["wer"] == 0 else "  "
        print(f"{row['latencia_ms']:>7.0f}ms  {row['wer']:>5.2f}  {mark} {row['oido']}")

    latencies = [float(r["latencia_ms"]) for r in rows]
    summary: dict[str, object] = {"frases": rows}
    print()
    if latencies:
        p50, p95 = statistics.median(latencies), percentile(latencies, 0.95)
        wer = statistics.mean(float(r["wer"]) for r in rows)
        verdict = "cumple" if p50 <= TARGET_FINAL_MS else "NO cumple"
        print(
            f"Transcripcion final  p50 {p50:.0f} ms  p95 {p95:.0f} ms  "
            f"(meta {TARGET_FINAL_MS:.0f} ms: {verdict})   WER medio {wer:.2f}"
        )
        summary.update({"final_p50_ms": p50, "final_p95_ms": p95, "wer": wer})
    for name in ("voice.wake", "voice.vad"):
        values = spans.get(name, [])
        if values:
            p50, p95 = statistics.median(values), percentile(values, 0.95)
            share = p95 / FRAME_BUDGET_MS * 100
            print(
                f"{name:<20} p50 {p50:.2f} ms  p95 {p95:.2f} ms por bloque  "
                f"({share:.0f}% del presupuesto de {FRAME_BUDGET_MS:.0f} ms)"
            )
            summary[f"{name}_p95_ms"] = p95
    first_audio = [s.first_audio_ms for s in speech if s.first_audio_ms is not None]
    if first_audio:
        print(
            "TTS primer audio     "
            + "  ".join(f"{ms:.0f} ms" for ms in first_audio)
            + "   (frases de 1, 5 y 9 palabras)"
        )
        summary["tts_first_audio_ms"] = first_audio
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fast", action="store_true", help="No esperar al reloj.")
    parser.add_argument("--real-wake", action="store_true", help="Usar el detector real.")
    parser.add_argument("--json", type=Path, help="Guardar el resumen en un archivo JSON.")
    args = parser.parse_args()
    try:
        summary = asyncio.run(run(args))
    except ModelMissingError as error:
        print(error)
        return 2
    if args.json:
        args.json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
