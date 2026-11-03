"""Contratos de la contabilidad de la cascada (Fase 2).

Cuatro mensajes que la cascada publica ademas de la respuesta:

    ResponseChunk   una frase de la respuesta, en cuanto existe (streaming a voz)
    RouteDecision   a que nivel se mando una orden y por que
    CostUpdate      lo que costo una llamada a la nube y como va el dia
    CascadeRecord   el resumen de una interaccion: niveles, tiempos, costo

Son la materia prima con la que la Fase 12 afina los umbrales y lo que el panel
cognitivo del HUD muestra en vivo.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from jarvis.contracts.base import BaseContract, CorrelationId, utc_now
from jarvis.contracts.enums import CascadeLevel, Emotion, PrivacyMode

__all__ = ["CascadeRecord", "CostUpdate", "ResponseChunk", "RouteDecision"]


class ResponseChunk(BaseContract):
    """Un trozo hablable de una respuesta que todavia se esta generando.

    Viaja en `cognition.response.chunk`. El pipeline de voz los dice en orden
    conforme llegan: asi el primer audio sale antes de que el modelo termine.
    La `Response` final de la misma interaccion llega con `streamed=True` y ya
    no se vuelve a decir.
    """

    text: str = Field(min_length=1, description="Una o mas frases completas.")
    index: int = Field(ge=0, description="Orden del trozo dentro de la respuesta.")
    emotion: Emotion = Field(default=Emotion.NEUTRAL)
    level: CascadeLevel = Field(description="Nivel que lo esta generando.")
    correlation_id: CorrelationId = Field(description="Interaccion a la que pertenece.")


class RouteDecision(BaseContract):
    """Una decision del enrutador. Viaja en `cognition.routed`.

    Hay una por intento: si el nivel 2 duda y se escala, se publica otra con
    `attempt=2` y el motivo de la escalada.
    """

    level: CascadeLevel = Field(description="Nivel elegido.")
    reason: str = Field(min_length=1, description="Motivo corto: patron, codigo, confianza...")
    intent: str = Field(min_length=1, description="Intencion clasificada.")
    model: str | None = Field(default=None, description="Modelo concreto, si hay uno.")
    attempt: int = Field(default=1, ge=1, description="1 la primera vez; sube al escalar.")
    privacy_mode: PrivacyMode = Field(description="Modo vigente al decidir.")
    correlation_id: CorrelationId = Field(description="Interaccion.")
    decided_at: datetime = Field(default_factory=utc_now)


class CostUpdate(BaseContract):
    """Lo que costo una llamada a la nube. Viaja en `cognition.cost.updated`."""

    model: str = Field(min_length=1)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cost_usd: float = Field(ge=0.0, description="Costo de esta llamada.")
    spent_today_usd: float = Field(ge=0.0, description="Acumulado del dia, incluida esta.")
    budget_today_usd: float = Field(ge=0.0, description="Tope diario. Cero es sin limite.")
    budget_exhausted: bool = Field(description="Si con esta llamada se agoto el dia.")
    correlation_id: CorrelationId | None = Field(default=None)


class CascadeRecord(BaseContract):
    """El resumen de como se resolvio una orden. Viaja en `cognition.completed`.

    Se publica SIEMPRE, tambien cuando todo falla: una interaccion sin registro
    es una interaccion que la Fase 12 no puede aprender.
    """

    text: str = Field(description="Lo que el usuario dijo.")
    intent: str = Field(min_length=1)
    intent_confidence: float = Field(ge=0.0, le=1.0)
    suggested_level: CascadeLevel = Field(description="Lo que propuso el clasificador.")
    handled_by_level: CascadeLevel = Field(description="El nivel que dio la respuesta final.")
    levels_tried: list[CascadeLevel] = Field(description="En orden. [2, 3] es una escalada.")
    reasons: list[str] = Field(default_factory=list, description="Motivo de cada decision.")
    model: str | None = Field(default=None, description="Modelo que contesto, si hubo.")
    answer_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confianza propia del modelo que contesto. None en el nivel 1.",
    )
    classify_ms: float = Field(ge=0.0)
    first_chunk_ms: float | None = Field(
        default=None,
        ge=0.0,
        description="Desde que llego la orden hasta la primera frase hablable.",
    )
    total_ms: float = Field(ge=0.0)
    cost_usd: float = Field(default=0.0, ge=0.0)
    degraded: bool = Field(
        default=False,
        description="Si se respondio con menos de lo que pedia la orden (nube cortada o caida).",
    )
    correlation_id: CorrelationId = Field(description="Interaccion.")
    finished_at: datetime = Field(default_factory=utc_now)
