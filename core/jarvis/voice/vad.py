"""Deteccion de actividad de voz y del fin de una frase.

Dos piezas distintas que conviene no mezclar:

- un `VoiceActivityDetector` dice, bloque a bloque, que tan probable es que haya
  habla (Silero en produccion, `EnergyVAD` como respaldo);
- el `EndpointDetector` convierte esa serie de probabilidades en dos sucesos con
  sentido: "empezo a hablar" y "termino de hablar".

Separarlas permite cambiar el VAD sin tocar las reglas de cuando cortar, que son
las que el usuario afina en `jarvis.toml`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

from jarvis.voice.audio import frame_level

if TYPE_CHECKING:
    from jarvis.voice.audio import AudioArray

__all__ = ["EndpointDetector", "EndpointEvent", "EnergyVAD"]


class EnergyVAD:
    """VAD por energia con piso de ruido adaptativo. No necesita ningun modelo.

    Sigue el ruido de fondo con un minimo que sube despacio y baja de golpe, y
    mide cuantos decibelios esta cada bloque por encima de ese piso. Funciona en
    una habitacion tranquila; con un ventilador o musica de fondo se confunde,
    por eso es el respaldo y no el valor por defecto.
    """

    def __init__(
        self,
        *,
        margin_db: float = 12.0,
        slope_db: float = 3.0,
        floor_rise_db_per_frame: float = 0.05,
        initial_floor_db: float = -60.0,
    ) -> None:
        """Configura el detector.

        Args:
            margin_db: decibelios sobre el piso de ruido en los que la
                probabilidad cruza 0.5.
            slope_db: lo abrupta que es la transicion alrededor de ese margen.
            floor_rise_db_per_frame: lo rapido que el piso sube cuando el ruido
                crece. Despacio a proposito: si subiera rapido, una frase larga
                acabaria siendo "el ruido de fondo".
            initial_floor_db: piso de partida, antes de haber oido nada.
        """
        self._margin_db = margin_db
        self._slope_db = slope_db
        self._rise = floor_rise_db_per_frame
        self._initial_floor_db = initial_floor_db
        self._floor_db = initial_floor_db

    @property
    def floor_db(self) -> float:
        """Piso de ruido estimado ahora mismo."""
        return self._floor_db

    def probability(self, frame: AudioArray) -> float:
        """Probabilidad de habla segun cuanto sobresale el bloque del ruido."""
        rms, _ = frame_level(frame)
        level_db = 20.0 * math.log10(max(rms, 1e-6))
        if level_db < self._floor_db:
            self._floor_db = level_db
        else:
            self._floor_db += self._rise
        excess = level_db - (self._floor_db + self._margin_db)
        return 1.0 / (1.0 + math.exp(-excess / self._slope_db))

    def reset(self) -> None:
        """Conserva el piso de ruido: el ruido de la habitacion no cambia entre frases."""


class EndpointEvent(StrEnum):
    """Sucesos que detecta el `EndpointDetector`."""

    STARTED = "STARTED"
    ENDED = "ENDED"


@dataclass(slots=True)
class EndpointDetector:
    """Decide cuando empieza y cuando termina una frase.

    Empieza tras `min_speech_ms` de habla continua, para que un golpe en la mesa
    no cuente como frase. Termina tras `silence_ms` de silencio continuo una vez
    empezada. Antes de empezar no hay fin posible: el plazo de "nadie hablo"
    lo controla el pipeline, no este detector.
    """

    threshold: float
    frame_ms: float
    min_speech_ms: float
    silence_ms: float
    started: bool = field(default=False, init=False)
    speech_ms: float = field(default=0.0, init=False)
    """Habla acumulada desde el inicio, sin contar los silencios intermedios."""
    _speech_run_ms: float = field(default=0.0, init=False)
    _silence_run_ms: float = field(default=0.0, init=False)

    def update(self, probability: float) -> EndpointEvent | None:
        """Alimenta un bloque y devuelve el suceso que provoco, si alguno."""
        is_speech = probability >= self.threshold
        if is_speech:
            self._speech_run_ms += self.frame_ms
            self._silence_run_ms = 0.0
        else:
            self._silence_run_ms += self.frame_ms
            if not self.started:
                self._speech_run_ms = 0.0

        if not self.started:
            if self._speech_run_ms >= self.min_speech_ms and is_speech:
                self.started = True
                self.speech_ms = self._speech_run_ms
                return EndpointEvent.STARTED
            return None

        if is_speech:
            self.speech_ms += self.frame_ms
            return None
        if self._silence_run_ms >= self.silence_ms:
            return EndpointEvent.ENDED
        return None
