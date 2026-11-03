"""Motores de voz falsos y deterministas para probar el pipeline sin modelos.

El audio sintetico codifica lo que "pasa" por amplitud:

    silencio    ceros
    habla       seno a 0.3 de amplitud  -> el VAD falso lo toma por voz
    palabra     seno a 0.9 de amplitud  -> el detector falso dispara

Asi una linea de tiempo como `timeline(("wake", 300), ("speech", 800))` describe
una interaccion completa sin un solo modelo cargado.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from jarvis.contracts import Language
from jarvis.voice.audio import frame_level
from jarvis.voice.engines import Transcription

if TYPE_CHECKING:
    from collections.abc import Iterator

    from jarvis.voice.audio import AudioArray

RATE = 16000

_AMPLITUDES = {"silence": 0.0, "speech": 0.3, "wake": 0.9}


def tone(milliseconds: float, amplitude: float, *, rate: int = RATE) -> AudioArray:
    """Un seno de 220 Hz."""
    samples = round(milliseconds * rate / 1000)
    t = np.arange(samples, dtype=np.float64) / rate
    return (amplitude * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)


def timeline(*parts: tuple[str, float]) -> AudioArray:
    """Concatena tramos de silencio, habla y palabra clave."""
    return np.concatenate([tone(ms, _AMPLITUDES[kind]) for kind, ms in parts])


class FakeWake:
    """Dispara con los bloques fuertes (la "palabra clave" sintetica)."""

    keyword = "fake_jarvis"

    def __init__(self) -> None:
        self.resets = 0
        self.frames = 0

    def score(self, frame: AudioArray) -> float:
        self.frames += 1
        _, peak = frame_level(frame)
        return 0.95 if peak > 0.7 else 0.0

    def reset(self) -> None:
        self.resets += 1


class FakeVAD:
    """Habla es cualquier cosa audible."""

    def __init__(self) -> None:
        self.frames = 0

    def probability(self, frame: AudioArray) -> float:
        self.frames += 1
        rms, _ = frame_level(frame)
        return 1.0 if rms > 0.05 else 0.0

    def reset(self) -> None:
        pass


class FakeSTT:
    """Devuelve un texto fijo y guarda el audio que le pidieron transcribir."""

    def __init__(
        self,
        text: str = "Jarvis, sube el volumen",
        *,
        delay_s: float = 0.0,
        fail: bool = False,
    ) -> None:
        self.text = text
        self.delay_s = delay_s
        self.fail = fail
        self.calls: list[AudioArray] = []

    def transcribe(self, audio: AudioArray, *, language: Language | None) -> Transcription:
        self.calls.append(audio)
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.fail:
            msg = "el modelo exploto"
            raise RuntimeError(msg)
        return Transcription(text=self.text, confidence=0.9, language=language or Language.ES)


class FakeTTS:
    """Sintetiza tantos trozos de seno como frases tenga el texto."""

    sample_rate = RATE

    def __init__(self, *, piece_ms: float = 300.0, delay_s: float = 0.0) -> None:
        self.piece_ms = piece_ms
        self.delay_s = delay_s
        self.texts: list[str] = []

    def synthesize(self, text: str) -> Iterator[AudioArray]:
        self.texts.append(text)
        for _sentence in [s for s in text.split(".") if s.strip()]:
            if self.delay_s:
                time.sleep(self.delay_s)
            yield tone(self.piece_ms, 0.2)
