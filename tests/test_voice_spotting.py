"""La palabra clave por transcripcion (`wake_engine = "whisper"`), con motores falsos.

Aqui no hay tono de "palabra clave" sintetica: el audio es solo habla, y es el
texto que devuelve el STT falso lo que decide si Jarvis despierta.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from jarvis.contracts import (
    CoreState,
    EventType,
    Language,
    StateChange,
    Utterance,
    WakeDetection,
)
from jarvis.voice.engines import Transcription
from jarvis.voice.text import split_wake_word
from tests.test_voice_pipeline import Harness, _config
from tests.voice_fakes import FakeSTT, FakeTTS, timeline

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from jarvis.voice.audio import AudioArray


class ScriptedSTT(FakeSTT):
    """Devuelve un texto distinto en cada llamada, en orden."""

    def __init__(self, *texts: str) -> None:
        super().__init__(texts[0])
        self._texts = list(texts)

    def transcribe(self, audio: AudioArray, *, language: Language | None) -> Transcription:
        self.calls.append(audio)
        text = self._texts.pop(0) if len(self._texts) > 1 else self._texts[0]
        return Transcription(text=text, confidence=0.9, language=language or Language.ES)


@pytest.fixture
async def make(bus) -> AsyncIterator[Callable[..., Harness]]:
    created: list[Harness] = []

    def build(audio: AudioArray, **kwargs: object) -> Harness:
        kwargs.setdefault("config", _config(wake_engine="whisper"))
        harness = Harness(bus, audio, **kwargs)  # type: ignore[arg-type]
        created.append(harness)
        return harness

    yield build
    for harness in created:
        await harness.pipeline.stop()


def _states(h: Harness) -> list[CoreState]:
    return [e.payload_as(StateChange).current for e in h.of(EventType.CORE_STATE_CHANGED)]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Jarvis, ¿qué hora es?", "¿qué hora es?"),
        ("Hey Yarvis", ""),
        ("Charvis sube el volumen", "sube el volumen"),
        ("hola, ¿qué tal?", None),
        ("dile a Jarvis que venga", None),
    ],
)
def test_separa_la_palabra_clave_de_la_orden(text, expected):
    assert split_wake_word(text, "jarvis") == expected


async def test_jarvis_y_la_orden_de_corrido_se_transcriben_una_sola_vez(make) -> None:
    stt = FakeSTT("Jarvis, ¿qué hora es?")
    h = make(timeline(("silence", 300), ("speech", 1200)), stt=stt)
    await h.run()

    finals = h.of(EventType.VOICE_STT_FINAL)
    assert len(finals) == 1
    assert finals[0].payload_as(Utterance).text == "¿qué hora es?"
    wake = h.of(EventType.VOICE_WAKE_DETECTED)[0].payload_as(WakeDetection)
    assert wake.keyword == "jarvis"
    assert finals[0].correlation_id == wake.correlation_id
    assert len(stt.calls) == 1, "la misma transcripcion trae la palabra clave y la orden"
    assert _states(h) == [CoreState.DESPIERTO, CoreState.ESCUCHANDO, CoreState.DORMIDO]


async def test_sin_la_palabra_clave_no_sale_nada(make) -> None:
    stt = FakeSTT("hola, ¿cómo estás? ¿vamos a comer?")
    h = make(
        timeline(("silence", 300), ("speech", 1200), ("silence", 700), ("speech", 800)), stt=stt
    )
    await h.run()

    assert len(stt.calls) == 2, "cada frase se reviso"
    voice = [e for e in h.events if e.type.value.startswith("voice.")]
    assert voice == [], "ni palabra clave, ni transcripcion, ni nivel"
    assert _states(h) == []


async def test_jarvis_solo_abre_la_escucha_para_la_orden(make) -> None:
    stt = ScriptedSTT("Jarvis.", "Jarvis, sube el volumen")
    audio = timeline(("silence", 300), ("speech", 500), ("silence", 900), ("speech", 900))
    h = make(audio, stt=stt)
    await h.run()

    finals = [e.payload_as(Utterance) for e in h.of(EventType.VOICE_STT_FINAL)]
    assert [u.text for u in finals] == ["sube el volumen"]
    assert len(h.of(EventType.VOICE_WAKE_DETECTED)) == 1
    assert len(stt.calls) == 2


async def test_jarvis_para_interrumpe_mientras_habla(make) -> None:
    stt = FakeSTT("Jarvis, para.")
    audio = timeline(("silence", 300), ("speech", 600), ("silence", 1200))
    h = make(audio, stt=stt, tts=FakeTTS(piece_ms=1500), realtime=True)
    await h.pipeline.start()
    result = await h.pipeline.speak("Una frase larga. Otra mas. Y otra.")
    await asyncio.wait_for(h.pipeline.source_finished.wait(), 10)
    assert await h.pipeline.wait_idle(10)
    await h.bus.wait_idle(2)

    assert result.interrupted
    wake = h.of(EventType.VOICE_WAKE_DETECTED)[0].payload_as(WakeDetection)
    assert wake.barge_in
    assert h.of(EventType.VOICE_STT_FINAL)[0].payload_as(Utterance).text == "para."
