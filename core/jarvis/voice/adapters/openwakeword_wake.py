"""Palabra clave con openWakeWord, y Silero VAD con el mismo runtime ONNX."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from jarvis.voice.audio import float_to_pcm16
from jarvis.voice.models import ModelMissingError

if TYPE_CHECKING:
    from pathlib import Path

    from jarvis.voice.audio import AudioArray

__all__ = ["OpenWakeWordDetector", "SileroVAD"]

#: openWakeWord calcula una prediccion cada 80 ms (1280 muestras a 16 kHz).
_OWW_CHUNK = 1280


def _require(path: Path, what: str) -> str:
    if not path.exists():
        raise ModelMissingError(what, path)
    return str(path)


class OpenWakeWordDetector:
    """Detector de palabra clave sobre openWakeWord.

    openWakeWord predice sobre bloques de 80 ms y el pipeline entrega bloques de
    32 ms. Se acumulan aqui hasta completar uno de 80, y solo ese bloque produce
    una puntuacion; los demas devuelven 0. Devolver la ultima puntuacion en cada
    bloque haria que un unico pico contara dos o tres veces.
    """

    def __init__(
        self,
        *,
        model_path: Path,
        melspec_path: Path,
        embedding_path: Path,
        keyword: str,
    ) -> None:
        """Carga los tres modelos ONNX."""
        from openwakeword.model import Model

        self._model = Model(
            wakeword_models=[_require(model_path, "palabra clave")],
            inference_framework="onnx",
            melspec_model_path=_require(melspec_path, "rasgos de openWakeWord"),
            embedding_model_path=_require(embedding_path, "rasgos de openWakeWord"),
        )
        self._keyword = keyword
        self._pending = np.zeros(0, dtype=np.int16)

    @property
    def keyword(self) -> str:
        """Lo que detecta."""
        return self._keyword

    def score(self, frame: AudioArray) -> float:
        """Puntuacion del bloque de 80 ms que este bloque completo, o 0."""
        self._pending = np.concatenate((self._pending, float_to_pcm16(frame)))
        best = 0.0
        while self._pending.shape[0] >= _OWW_CHUNK:
            piece, self._pending = self._pending[:_OWW_CHUNK], self._pending[_OWW_CHUNK:]
            predictions = self._model.predict(piece)
            best = max(best, *(float(value) for value in predictions.values()))
        return best

    def reset(self) -> None:
        """Vacia los buffers de rasgos para que el mismo audio no vuelva a disparar."""
        self._model.reset()
        self._pending = np.zeros(0, dtype=np.int16)


class SileroVAD:
    """Silero VAD, el que trae openWakeWord. Mucho mas robusto al ruido que la energia."""

    def __init__(self, *, model_path: Path) -> None:
        """Carga el modelo."""
        from openwakeword.vad import VAD

        self._vad = VAD(model_path=_require(model_path, "Silero VAD"))

    def probability(self, frame: AudioArray) -> float:
        """Probabilidad de habla del bloque.

        Silero mantiene estado recurrente entre llamadas: el mismo bloque puntua
        distinto segun lo que oyo antes, que es justo lo que lo hace bueno.
        """
        pcm = float_to_pcm16(frame)
        return float(self._vad.predict(pcm, frame_size=pcm.shape[0]))

    def reset(self) -> None:
        """Reinicia el estado recurrente."""
        self._vad.reset_states()
