"""Contratos del pipeline de voz."""

from __future__ import annotations

from datetime import datetime
from typing import Self

from pydantic import Field, model_validator

from jarvis.contracts.base import BaseContract, CorrelationId, utc_now
from jarvis.contracts.enums import Language

__all__ = ["Utterance"]


class Utterance(BaseContract):
    """Un fragmento de audio ya transcrito a texto.

    El motor de transcripcion emite muchas Utterance parciales mientras el
    usuario habla (`is_final=False`, que el HUD muestra en gris apareciendo
    palabra por palabra) y exactamente una final cuando el detector de fin de
    habla decide que la frase termino.

    Solo la final alimenta al clasificador de intencion. Las parciales existen
    para que la interfaz se sienta viva y, desde la Fase 7, para la ejecucion
    optimista: empezar a abrir YouTube mientras el usuario aun termina la frase.
    """

    text: str = Field(
        description="Texto transcrito. Puede venir vacio si el usuario no dijo nada.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confianza del motor de transcripcion, de 0 a 1.",
    )
    language: Language = Field(
        default=Language.ES,
        description="Idioma detectado. El pipeline distingue espanol de ingles tecnico.",
    )
    duration_ms: float = Field(
        ge=0.0,
        description="Duracion del audio transcrito en milisegundos.",
    )
    started_at: datetime = Field(
        default_factory=utc_now,
        description="Instante en que empezo el audio de esta transcripcion.",
    )
    ended_at: datetime = Field(
        default_factory=utc_now,
        description="Instante en que termino el audio de esta transcripcion.",
    )
    is_final: bool = Field(
        default=False,
        description="Si es la transcripcion definitiva o una parcial en vivo.",
    )
    correlation_id: CorrelationId = Field(
        description="Interaccion a la que pertenece esta transcripcion.",
    )

    @model_validator(mode="after")
    def _validate_time_range(self) -> Self:
        """El audio no puede terminar antes de empezar."""
        if self.ended_at < self.started_at:
            msg = (
                f"ended_at ({self.ended_at.isoformat()}) es anterior a "
                f"started_at ({self.started_at.isoformat()})"
            )
            raise ValueError(msg)
        return self

    @property
    def is_empty(self) -> bool:
        """Si la transcripcion no trajo texto util.

        Pasa mas seguido de lo que parece: el usuario dice "Jarvis" y se queda
        callado, o el disparo fue un falso positivo. El pipeline debe tratarlo
        como caso normal, no como error.
        """
        return not self.text.strip()
