"""Pruebas del pipeline de voz con motores falsos.

Lo que aqui se verifica es la orquestacion -- la maquina de estados, los
plazos, el orden de los eventos y las garantias de privacidad -- no la calidad
de ningun modelo. Para eso esta `test_voice_engines.py`.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import numpy as np
import pytest

from jarvis.config import VoiceConfig
from jarvis.contracts import (
    CascadeLevel,
    CoreState,
    Emotion,
    Event,
    EventType,
    Response,
    ResponseChunk,
    SpeechActivity,
    SpeechEndReason,
    SpeechOutput,
    StateChange,
    Utterance,
    WakeDetection,
)
from jarvis.voice import VoiceEngines, VoicePipeline
from jarvis.voice.io import ArraySource, MemorySink
from tests.voice_fakes import RATE, FakeSTT, FakeTTS, FakeVAD, FakeWake, timeline

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from jarvis.bus import InMemoryEventBus
    from jarvis.voice.audio import AudioArray


def _config(**overrides: object) -> VoiceConfig:
    """Configuracion corta para que las pruebas no esperen de mas."""
    values: dict[str, object] = {
        "vad_silence_ms": 320,
        "vad_min_speech_ms": 64,
        "speech_start_timeout_s": 1.0,
        "max_listen_s": 5.0,
        "wake_cooldown_s": 1.0,
        "stt_partials": False,
    }
    values.update(overrides)
    return VoiceConfig(**values)  # type: ignore[arg-type]


class Harness:
    """Pipeline + bus + motores falsos, con todos los eventos capturados."""

    def __init__(
        self,
        bus: InMemoryEventBus,
        audio: AudioArray,
        *,
        config: VoiceConfig | Callable[[], VoiceConfig] | None = None,
        stt: FakeSTT | None = None,
        tts: FakeTTS | None = None,
        realtime: bool = False,
        tail_silence_ms: int = 600,
    ) -> None:
        self.bus = bus
        self.events: list[Event] = []
        cfg = config if config is not None else _config()
        frame_ms = (cfg() if callable(cfg) else cfg).frame_ms
        self.source = ArraySource(
            audio,
            sample_rate=RATE,
            frame_ms=frame_ms,
            realtime=realtime,
            tail_silence_ms=tail_silence_ms,
        )
        self.wake = FakeWake()
        self.vad = FakeVAD()
        self.stt = stt or FakeSTT()
        self.tts = tts or FakeTTS()
        self.sink = MemorySink(realtime=realtime)
        self.pipeline = VoicePipeline(
            bus,
            VoiceEngines(
                source=self.source,
                wake=self.wake,
                vad=self.vad,
                stt=self.stt,
                tts=self.tts,
                sink=self.sink,
            ),
            cfg,
        )

        self.reached: dict[CoreState, asyncio.Event] = {
            state: asyncio.Event() for state in CoreState
        }

        async def collect(event: Event) -> None:
            self.events.append(event)
            if event.type is EventType.CORE_STATE_CHANGED:
                self.reached[event.payload_as(StateChange).current].set()

        bus.subscribe("*", collect, name="test.collector", max_queue=100_000)

    async def run(self, timeout: float = 5.0) -> None:  # noqa: ASYNC109
        """Arranca, consume toda la fuente y espera a que todo termine."""
        await self.pipeline.start()
        await asyncio.wait_for(self.pipeline.source_finished.wait(), timeout)
        assert await self.pipeline.wait_idle(timeout)
        assert await self.bus.wait_idle(timeout)

    async def until(self, state: CoreState, timeout: float = 5.0) -> None:  # noqa: ASYNC109
        """Espera a que el pipeline entre en un estado."""
        await asyncio.wait_for(self.reached[state].wait(), timeout)

    def of(self, event_type: EventType) -> list[Event]:
        return [e for e in self.events if e.type is event_type]

    def states(self) -> list[CoreState]:
        return [e.payload_as(StateChange).current for e in self.of(EventType.CORE_STATE_CHANGED)]


@pytest.fixture
async def harness_factory(bus: InMemoryEventBus) -> AsyncIterator[Callable[..., Harness]]:
    created: list[Harness] = []

    def make(audio: AudioArray, **kwargs: object) -> Harness:
        harness = Harness(bus, audio, **kwargs)  # type: ignore[arg-type]
        created.append(harness)
        return harness

    yield make
    for harness in created:
        await harness.pipeline.stop()


# ---------------------------------------------------------------------------
# Una interaccion completa
# ---------------------------------------------------------------------------


async def test_una_interaccion_completa_de_principio_a_fin(harness_factory) -> None:
    """Wake -> escucha -> fin por silencio -> transcripcion final -> a dormir."""
    h = harness_factory(timeline(("silence", 500), ("wake", 300), ("speech", 800)))
    await h.run()

    finals = h.of(EventType.VOICE_STT_FINAL)
    assert len(finals) == 1
    utterance = finals[0].payload_as(Utterance)
    assert utterance.is_final
    assert utterance.text == "sube el volumen"  # la palabra clave se quita
    assert utterance.duration_ms > 800

    ended = h.of(EventType.VOICE_SPEECH_ENDED)[0].payload_as(SpeechActivity)
    assert ended.reason is SpeechEndReason.SILENCIO
    assert h.states() == [CoreState.DESPIERTO, CoreState.ESCUCHANDO, CoreState.DORMIDO]
    assert h.pipeline.state is CoreState.DORMIDO


async def test_los_eventos_llegan_en_orden_y_con_el_mismo_id(harness_factory) -> None:
    """Todo lo de una interaccion comparte el id de correlacion que nace en el wake."""
    h = harness_factory(timeline(("silence", 300), ("wake", 300), ("speech", 600)))
    await h.run()

    wake = h.of(EventType.VOICE_WAKE_DETECTED)[0].payload_as(WakeDetection)
    voice_types = [
        e.type
        for e in h.events
        if e.type.value.startswith("voice.") and e.type is not EventType.VOICE_LEVEL
    ]
    assert voice_types == [
        EventType.VOICE_WAKE_DETECTED,
        EventType.VOICE_SPEECH_STARTED,
        EventType.VOICE_SPEECH_ENDED,
        EventType.VOICE_STT_FINAL,
    ]
    interaction = [e for e in h.events if e.type.value.startswith("voice.")]
    assert {e.correlation_id for e in interaction} == {wake.correlation_id}


async def test_el_preroll_rescata_el_audio_previo_al_disparo(harness_factory) -> None:
    """Lo que Whisper recibe empieza ANTES de la escucha: los 500 ms del buffer."""
    h = harness_factory(timeline(("silence", 1000), ("wake", 300), ("speech", 600)))
    await h.run()

    wake = h.of(EventType.VOICE_WAKE_DETECTED)[0].payload_as(WakeDetection)
    assert wake.preroll_ms == 500
    audio = h.stt.calls[0]
    # Los primeros bloques son silencio del buffer y despues la palabra clave.
    assert np.max(np.abs(audio[: RATE // 4])) < 0.01
    assert np.max(np.abs(audio[: RATE // 2 + 1024])) > 0.7


async def test_el_nivel_se_publica_a_30_hz_mientras_escucha(harness_factory) -> None:
    """Un evento de nivel por bloque de 32 ms, con la probabilidad del VAD."""
    h = harness_factory(timeline(("wake", 300), ("speech", 960)))
    await h.run()

    levels = h.of(EventType.VOICE_LEVEL)
    # 960 ms de habla + 320 ms de silencio de cierre = 40 bloques, como minimo.
    assert len(levels) >= 40
    assert all(e.payload["speech_probability"] is not None for e in levels)


# ---------------------------------------------------------------------------
# La palabra clave es sagrada
# ---------------------------------------------------------------------------


async def test_sin_palabra_clave_no_sale_nada(harness_factory) -> None:
    """Habla sin "Jarvis": ni eventos de voz, ni transcripcion, ni VAD."""
    h = harness_factory(timeline(("speech", 2000), ("silence", 500), ("speech", 1000)))
    await h.run()

    assert not [e for e in h.events if e.type.value.startswith("voice.")]
    assert h.stt.calls == []
    assert h.vad.frames == 0
    assert h.wake.frames > 0


async def test_el_nivel_en_reposo_solo_si_se_pide(harness_factory) -> None:
    """`publish_idle_level` es la unica forma de que salga algo en DORMIDO."""
    h = harness_factory(timeline(("speech", 640)), config=_config(publish_idle_level=True))
    await h.run()

    levels = h.of(EventType.VOICE_LEVEL)
    assert len(levels) >= 20
    assert all(e.correlation_id is None for e in levels)
    assert h.stt.calls == []


async def test_el_enfriamiento_evita_dobles_disparos(harness_factory) -> None:
    """Dos palabras clave separadas por menos del enfriamiento cuentan una vez."""
    h = harness_factory(
        timeline(("wake", 200), ("speech", 300), ("silence", 400), ("wake", 200)),
        config=_config(wake_cooldown_s=60.0),
    )
    await h.run()

    assert len(h.of(EventType.VOICE_WAKE_DETECTED)) == 1


async def test_dos_interacciones_seguidas_tienen_ids_distintos(harness_factory) -> None:
    h = harness_factory(
        timeline(
            ("wake", 200),
            ("speech", 400),
            ("silence", 1200),
            ("wake", 200),
            ("speech", 400),
        ),
        config=_config(wake_cooldown_s=0.5),
    )
    await h.run()

    finals = h.of(EventType.VOICE_STT_FINAL)
    assert len(finals) == 2
    assert finals[0].correlation_id != finals[1].correlation_id


# ---------------------------------------------------------------------------
# Plazos
# ---------------------------------------------------------------------------


async def test_jarvis_y_silencio_da_una_transcripcion_vacia(harness_factory) -> None:
    """Caso normal, no error: se publica una final vacia y no se llama a Whisper."""
    # Un solo bloque de palabra clave despues del que dispara: menos que
    # `vad_min_speech_ms`, asi que el VAD no lo toma por el inicio de una frase.
    h = harness_factory(timeline(("wake", 64)), tail_silence_ms=1500)
    await h.run()

    final = h.of(EventType.VOICE_STT_FINAL)[0].payload_as(Utterance)
    assert final.is_empty
    assert final.is_final
    ended = h.of(EventType.VOICE_SPEECH_ENDED)[0].payload_as(SpeechActivity)
    assert ended.reason is SpeechEndReason.SIN_HABLA
    assert h.stt.calls == []
    assert h.pipeline.state is CoreState.DORMIDO


async def test_el_tope_de_escucha_corta_con_elegancia(harness_factory) -> None:
    """Hablar sin parar mas de `max_listen_s` cierra y transcribe lo que hubo."""
    h = harness_factory(
        timeline(("wake", 200), ("speech", 3000)),
        config=_config(max_listen_s=1.0),
    )
    await h.run()

    ended = h.of(EventType.VOICE_SPEECH_ENDED)[0].payload_as(SpeechActivity)
    assert ended.reason is SpeechEndReason.TOPE
    assert len(h.of(EventType.VOICE_STT_FINAL)) == 1


async def test_un_golpe_corto_no_cuenta_como_frase(harness_factory) -> None:
    """Menos de `vad_min_speech_ms` de sonido no abre la frase."""
    h = harness_factory(
        timeline(("wake", 200), ("silence", 200), ("speech", 32), ("silence", 1500)),
        config=_config(vad_min_speech_ms=200),
    )
    await h.run()

    assert h.of(EventType.VOICE_SPEECH_STARTED) == []
    reason = h.of(EventType.VOICE_SPEECH_ENDED)[0].payload_as(SpeechActivity).reason
    assert reason is SpeechEndReason.SIN_HABLA


async def test_la_configuracion_se_lee_en_caliente(harness_factory) -> None:
    """Con una funcion como configuracion, cada bloque usa el valor vigente.

    El pipeline consulta la configuracion una vez por bloque, asi que contar
    las consultas dice exactamente por que bloque va: el umbral se "recarga"
    pasada la primera palabra clave, que con el umbral viejo no dispara.
    """
    calls = 0

    def provider() -> VoiceConfig:
        nonlocal calls
        calls += 1
        # El detector falso da 0.95: 0.99 es inalcanzable, 0.5 no.
        return _config(wake_threshold=0.99 if calls < 40 else 0.5)

    h = harness_factory(
        timeline(("wake", 200), ("speech", 400), ("silence", 1500), ("wake", 200), ("speech", 400)),
        config=provider,
    )
    await h.run()

    assert len(h.of(EventType.VOICE_WAKE_DETECTED)) == 1
    assert len(h.of(EventType.VOICE_STT_FINAL)) == 1


# ---------------------------------------------------------------------------
# Transcripcion
# ---------------------------------------------------------------------------


async def test_las_parciales_llegan_antes_que_la_final(harness_factory) -> None:
    h = harness_factory(
        timeline(("wake", 200), ("speech", 2000)),
        config=_config(stt_partials=True, stt_partial_interval_ms=300),
        realtime=True,
    )
    await h.run(timeout=10)

    partials = h.of(EventType.VOICE_STT_PARTIAL)
    assert partials
    assert all(not e.payload_as(Utterance).is_final for e in partials)
    final_index = h.events.index(h.of(EventType.VOICE_STT_FINAL)[0])
    assert all(h.events.index(p) < final_index for p in partials)


async def test_un_fallo_de_whisper_no_falla_en_silencio(harness_factory) -> None:
    """La excepcion se convierte en `core.error` y el pipeline vuelve a dormir."""
    h = harness_factory(timeline(("wake", 200), ("speech", 400)), stt=FakeSTT(fail=True))
    await h.run()

    errors = h.of(EventType.CORE_ERROR)
    assert len(errors) == 1
    assert errors[0].payload["module"] == "voice.stt"
    assert h.of(EventType.VOICE_STT_FINAL) == []
    assert h.pipeline.state is CoreState.DORMIDO


async def test_la_captura_rota_se_anuncia(bus: InMemoryEventBus) -> None:
    """Si el microfono muere, el bus se entera."""

    class BrokenSource:
        sample_rate = RATE

        async def frames(self) -> AsyncIterator[AudioArray]:
            yield np.zeros(512, dtype=np.float32)
            msg = "se desconecto el microfono"
            raise OSError(msg)

        async def close(self) -> None:
            pass

    errors: list[Event] = []

    async def on_error(event: Event) -> None:
        errors.append(event)

    bus.subscribe(EventType.CORE_ERROR, on_error)
    pipeline = VoicePipeline(
        bus,
        VoiceEngines(
            source=BrokenSource(),
            wake=FakeWake(),
            vad=FakeVAD(),
            stt=FakeSTT(),
            tts=FakeTTS(),
            sink=MemorySink(),
        ),
        _config(),
    )
    await pipeline.start()
    await asyncio.wait_for(pipeline.source_finished.wait(), 2)
    await bus.wait_idle(2)
    await pipeline.stop()

    assert errors[0].payload["module"] == "voice.capture"
    assert "microfono" in str(errors[0].payload["detail"])


# ---------------------------------------------------------------------------
# Habla
# ---------------------------------------------------------------------------


async def test_hablar_publica_amplitud_a_60_hz(harness_factory) -> None:
    h = harness_factory(timeline(("silence", 100)))
    await h.pipeline.start()
    result = await h.pipeline.speak("Listo. Ya esta.", emotion=Emotion.AFIRMATIVA)
    await h.bus.wait_idle(2)

    assert not result.interrupted
    assert result.duration_ms == pytest.approx(600.0, abs=1.0)  # dos frases de 300 ms
    assert result.first_audio_ms is not None
    amplitudes = h.of(EventType.VOICE_TTS_AMPLITUDE)
    assert len(amplitudes) == pytest.approx(0.6 * 60, abs=2)
    assert h.sink.audio.shape[0] == round(0.6 * RATE)
    first_tts = next(e for e in h.events if e.type.value.startswith("voice.tts."))
    assert first_tts.type is EventType.VOICE_TTS_STARTED
    assert h.states()[-2:] == [CoreState.HABLANDO, CoreState.DORMIDO]


async def test_las_respuestas_de_la_cascada_se_dicen_solas(harness_factory) -> None:
    """Basta publicar `cognition.response`: el pipeline la dice en voz alta."""
    h = harness_factory(timeline(("silence", 100)))
    await h.pipeline.start()
    response = Response(
        text="Volumen al setenta.",
        handled_by_level=CascadeLevel.FAST_PATH,
        correlation_id="abc123",
    )
    await h.bus.publish(Event.of(EventType.COGNITION_RESPONSE, response, source="test"))
    await h.bus.wait_idle(2)
    assert await h.pipeline.wait_idle(2)
    await h.bus.wait_idle(2)

    assert h.tts.texts == ["Volumen al setenta."]
    finished = h.of(EventType.VOICE_TTS_FINISHED)[0]
    assert finished.correlation_id == "abc123"


async def test_no_habla_encima_del_usuario(harness_factory) -> None:
    """Si llega una respuesta mientras escucha, espera a que termine la frase."""
    h = harness_factory(timeline(("wake", 200), ("speech", 1200)), realtime=True)
    await h.pipeline.start()
    await h.until(CoreState.ESCUCHANDO)
    speaking = asyncio.create_task(h.pipeline.speak("Un momento."))
    await asyncio.wait_for(speaking, 10)
    await h.bus.wait_idle(2)

    final_at = h.events.index(h.of(EventType.VOICE_STT_FINAL)[0])
    tts_at = h.events.index(h.of(EventType.VOICE_TTS_STARTED)[0])
    assert final_at < tts_at


async def test_la_palabra_clave_interrumpe_a_jarvis(harness_factory) -> None:
    """Barge-in: "Jarvis" mientras habla lo calla y pasa directo a escuchar."""
    audio = timeline(("silence", 400), ("wake", 200), ("speech", 400))
    h = harness_factory(audio, tts=FakeTTS(piece_ms=1000), realtime=True)
    await h.pipeline.start()
    result = await h.pipeline.speak("Una frase larga. Otra mas. Y otra.")
    await asyncio.wait_for(h.pipeline.source_finished.wait(), 10)
    assert await h.pipeline.wait_idle(10)
    await h.bus.wait_idle(2)

    assert result.interrupted
    assert result.duration_ms < 1000
    assert h.sink.aborted == 1
    barge = h.of(EventType.VOICE_BARGE_IN)[0].payload_as(SpeechOutput)
    assert barge.interrupted
    wake = h.of(EventType.VOICE_WAKE_DETECTED)[0].payload_as(WakeDetection)
    assert wake.barge_in
    assert len(h.of(EventType.VOICE_STT_FINAL)) == 1
    assert CoreState.HABLANDO in h.states()
    assert h.states()[h.states().index(CoreState.HABLANDO) + 1] is CoreState.DESPIERTO


async def test_sin_barge_in_la_palabra_clave_no_interrumpe(harness_factory) -> None:
    audio = timeline(("silence", 200), ("wake", 200), ("silence", 200))
    h = harness_factory(
        audio,
        tts=FakeTTS(piece_ms=1000),
        realtime=True,
        config=_config(barge_in_enabled=False),
    )
    await h.pipeline.start()
    result = await h.pipeline.speak("Una frase.")

    assert not result.interrupted
    assert h.of(EventType.VOICE_WAKE_DETECTED) == []


async def test_detener_a_media_escucha_cancela(harness_factory) -> None:
    h = harness_factory(timeline(("wake", 200), ("speech", 3000)), realtime=True)
    await h.pipeline.start()
    await h.until(CoreState.ESCUCHANDO)
    await h.pipeline.stop()
    await h.bus.wait_idle(2)

    ended = h.of(EventType.VOICE_SPEECH_ENDED)[0].payload_as(SpeechActivity)
    assert ended.reason is SpeechEndReason.CANCELADA
    assert h.stt.calls == []


async def test_si_la_fuente_se_acaba_a_media_frase_se_transcribe_lo_que_hubo(
    harness_factory,
) -> None:
    """Regresion: sin mas bloques ningun plazo vencia y la escucha quedaba abierta."""
    h = harness_factory(timeline(("wake", 200), ("speech", 600)), tail_silence_ms=0)
    await h.run()
    result = await asyncio.wait_for(h.pipeline.speak("Listo."), 5)

    ended = h.of(EventType.VOICE_SPEECH_ENDED)[0].payload_as(SpeechActivity)
    assert ended.reason is SpeechEndReason.SILENCIO
    assert len(h.of(EventType.VOICE_STT_FINAL)) == 1
    assert not result.interrupted


# ---------------------------------------------------------------------------
# Respuestas en streaming (Fase 2)
# ---------------------------------------------------------------------------


def _chunk(text: str, index: int, cid: str = "c1") -> Event:
    chunk = ResponseChunk(text=text, index=index, level=CascadeLevel.LOCAL_LLM, correlation_id=cid)
    return Event.of(EventType.COGNITION_RESPONSE_CHUNK, chunk, source="test", correlation_id=cid)


async def test_dice_los_trozos_en_orden_y_no_repite_la_respuesta(harness_factory) -> None:
    h = harness_factory(timeline(("silence", 100)))
    await h.pipeline.start()
    await h.bus.publish(_chunk("Primera frase.", 0))
    await h.bus.publish(_chunk("Segunda frase.", 1))
    streamed = Response(
        text="Primera frase. Segunda frase.",
        handled_by_level=CascadeLevel.LOCAL_LLM,
        streamed=True,
        correlation_id="c1",
    )
    await h.bus.publish(Event.of(EventType.COGNITION_RESPONSE, streamed, source="test"))
    await h.bus.wait_idle(2)
    assert await h.pipeline.wait_idle(5)
    await h.bus.wait_idle(2)

    assert h.tts.texts == ["Primera frase.", "Segunda frase."]
    # Entre una frase y la siguiente no se vuelve a dormir: el HUD no parpadea.
    assert h.states() == [CoreState.HABLANDO, CoreState.DORMIDO]


async def test_una_orden_nueva_descarta_lo_pendiente_de_la_anterior(harness_factory) -> None:
    audio = timeline(("silence", 400), ("wake", 200), ("speech", 400))
    h = harness_factory(audio, tts=FakeTTS(piece_ms=1000), realtime=True)
    await h.pipeline.start()
    for index in range(3):
        await h.bus.publish(_chunk(f"Frase vieja {index}.", index, cid="vieja"))
    await asyncio.wait_for(h.pipeline.source_finished.wait(), 10)
    assert await h.pipeline.wait_idle(10)
    await h.bus.wait_idle(2)

    # La primera sono y la palabra clave la corto; las otras dos ya no se dicen.
    assert h.tts.texts == ["Frase vieja 0."]
    await h.bus.publish(_chunk("Frase vieja tardia.", 3, cid="vieja"))
    await h.bus.wait_idle(2)
    assert await h.pipeline.wait_idle(5)
    assert h.tts.texts == ["Frase vieja 0."]


async def test_la_transcripcion_no_queda_en_el_registro(harness_factory, monkeypatch) -> None:
    """Se espia el logger del pipeline: structlog cachea el suyo con el nivel de la
    primera configuracion, asi que leer un archivo dependeria del orden de las pruebas."""
    import jarvis.voice.pipeline as pipeline_module

    logged: list[dict[str, object]] = []

    class Spy:
        def __getattr__(self, level: str):  # info, warning, exception...
            def record(event: str, **fields: object) -> None:
                logged.append({"event": event, **fields})

            return record

    monkeypatch.setattr(pipeline_module, "_log", Spy())
    h = harness_factory(timeline(("silence", 300), ("wake", 300), ("speech", 600)))
    await h.run()
    assert h.of(EventType.VOICE_STT_FINAL)[0].payload_as(Utterance).text == "sube el volumen"
    final = next(entry for entry in logged if entry["event"] == "voice.final")
    assert final["text"] == "<15 caracteres>"
    assert all("sube el volumen" not in str(entry) for entry in logged)
