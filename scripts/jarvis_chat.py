"""Habla con Jarvis escribiendo. La demostracion de la Fase 2 sin microfono.

Cada linea pasa por la cascada completa -- clasificador, enrutador, ruta
rapida, Ollama y la nube si hay clave -- y se muestra que nivel la atendio,
por que y cuanto tardo. Con `--voz`, ademas, Jarvis dice las respuestas por
los altavoces con Piper, frase a frase conforme se generan.

Uso:

    python scripts/jarvis_chat.py
    python scripts/jarvis_chat.py --voz

Ctrl+C o una linea vacia para salir.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))

from jarvis.bus import InMemoryEventBus
from jarvis.cognition import build_cascade
from jarvis.config_watcher import ConfigManager
from jarvis.contracts import CascadeRecord, Event, EventType, RouteDecision
from jarvis.logging import bind_bus, configure_logging

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from jarvis.contracts import Language
    from jarvis.voice import VoicePipeline
    from jarvis.voice.engines import Transcription


class _Silence:
    """Fuente de audio muda: el pipeline solo se usa para hablar."""

    def __init__(self, sample_rate: int, frame_ms: int) -> None:
        self.sample_rate = sample_rate
        self._frame = np.zeros(sample_rate * frame_ms // 1000, dtype=np.float32)
        self._seconds = frame_ms / 1000
        self._closed = asyncio.Event()

    async def frames(self) -> AsyncIterator[np.ndarray]:
        while not self._closed.is_set():
            yield self._frame
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._closed.wait(), self._seconds)

    async def close(self) -> None:
        self._closed.set()


class _NoSTT:
    """Nunca se llama: sin palabra clave no hay nada que transcribir."""

    def transcribe(self, audio: np.ndarray, *, language: Language | None) -> Transcription:
        raise RuntimeError("jarvis_chat no transcribe")


async def _speaker(bus: InMemoryEventBus, manager: ConfigManager) -> VoicePipeline:
    from jarvis.voice import VoiceEngines, VoicePipeline
    from jarvis.voice.adapters.sounddevice_io import SpeakerSink
    from jarvis.voice.factory import NeverWake, build_tts
    from jarvis.voice.vad import EnergyVAD

    voice = manager.current.voice
    engines = VoiceEngines(
        source=_Silence(voice.sample_rate, voice.frame_ms),
        wake=NeverWake(),
        vad=EnergyVAD(),
        stt=_NoSTT(),
        tts=build_tts(voice),
        sink=SpeakerSink(device=voice.output_device),
    )
    pipeline = VoicePipeline(bus, engines, lambda: manager.current.voice)
    await pipeline.start()
    return pipeline


async def run(args: argparse.Namespace) -> int:
    bus = InMemoryEventBus()
    manager = ConfigManager(bus=bus)
    configure_logging(
        level="WARNING",
        development=True,
        log_file=None,
        log_user_content=manager.current.log_user_content,
    )
    routes: list[RouteDecision] = []
    records: list[CascadeRecord] = []

    async def on_route(event: Event) -> None:
        routes.append(event.payload_as(RouteDecision))

    async def on_record(event: Event) -> None:
        records.append(event.payload_as(CascadeRecord))

    async with bus:
        bind_bus(bus)
        await manager.start()
        bus.subscribe(EventType.COGNITION_ROUTED, on_route)
        bus.subscribe(EventType.COGNITION_COMPLETED, on_record)
        cascade = build_cascade(bus, lambda: manager.current)
        pipeline = await _speaker(bus, manager) if args.voz else None
        print(f"Calentando {manager.current.cognition.local_model} ...")
        await cascade.start(listen_voice=False)
        print(await cascade.health())
        print("Escribe una orden. Linea vacia para salir.\n")
        loop = asyncio.get_running_loop()
        try:
            while True:
                text = (await loop.run_in_executor(None, input, "tu> ")).strip()
                if not text:
                    break
                routes.clear()
                response = await cascade.handle_text(text)
                await bus.wait_idle(2)
                record = records[-1]
                path = " -> ".join(f"N{int(r.level)} ({r.reason})" for r in routes)
                first = (
                    f", 1a frase {record.first_chunk_ms:.0f} ms" if record.first_chunk_ms else ""
                )
                cost = f", {record.cost_usd:.5f} USD" if record.cost_usd else ""
                print(f"jarvis> {response.text}")
                print(
                    f"        [{record.intent} | {path} | {record.total_ms:.0f} ms{first}{cost}]\n"
                )
                if pipeline is not None:
                    await pipeline.wait_idle(60)
        finally:
            await cascade.stop()
            if pipeline is not None:
                await pipeline.stop()
            await manager.stop()
            bind_bus(None)
    return 0


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stdin.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--voz", action="store_true", help="Decir las respuestas por los altavoces."
    )
    args = parser.parse_args()
    try:
        return asyncio.run(run(args))
    except (KeyboardInterrupt, EOFError):
        print("\nHasta luego.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
