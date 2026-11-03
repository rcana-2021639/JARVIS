"""Sintesis local con Piper."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from jarvis.voice.models import ModelMissingError

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from jarvis.voice.audio import AudioArray

__all__ = ["PiperTTS"]


class PiperTTS:
    """Piper sobre ONNX. Produce un trozo de audio por frase."""

    def __init__(self, *, model_path: Path, config_path: Path) -> None:
        """Carga la voz."""
        from piper import PiperVoice

        if not model_path.exists():
            raise ModelMissingError("la voz de Piper", model_path)
        if not config_path.exists():
            raise ModelMissingError("la configuracion de la voz de Piper", config_path)
        self._voice = PiperVoice.load(model_path, config_path=config_path)
        self._sample_rate = int(self._voice.config.sample_rate)

    @property
    def sample_rate(self) -> int:
        """Frecuencia de la voz (22050 Hz en las voces 'high' y 'medium')."""
        return self._sample_rate

    def synthesize(self, text: str) -> Iterator[AudioArray]:
        """Audio frase por frase."""
        for piece in self._voice.synthesize(text):
            yield np.asarray(piece.audio_float_array, dtype=np.float32)
