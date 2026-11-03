"""Contratos de la cascada cognitiva: intencion, orden y respuesta."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field, JsonValue

from jarvis.contracts.base import BaseContract, CorrelationId, new_id, utc_now
from jarvis.contracts.enums import CascadeLevel, CommandOrigin, Emotion
from jarvis.contracts.tools import ToolCall

__all__ = ["Command", "Intent", "Response"]


class Intent(BaseContract):
    """Lo que el clasificador entendio que el usuario quiere.

    El clasificador debe resolver en menos de 10ms, asi que no razona: hace
    coincidencia contra un catalogo de patrones y, si no hay coincidencia clara,
    consulta un clasificador de embeddings pequeno. `suggested_level` es su
    recomendacion, no una orden: el enrutador de la Fase 2 tiene la ultima
    palabra y puede escalar por otros motivos.
    """

    name: str = Field(
        min_length=1,
        description="Nombre canonico de la intencion: media.volume_up, apps.open, chat.free...",
    )
    entities: dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Parametros extraidos del texto: {'app': 'spotify'}, {'delta': 10}.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Que tan seguro esta el clasificador de haber acertado.",
    )
    suggested_level: CascadeLevel = Field(
        description="Nivel de la cascada que el clasificador cree suficiente.",
    )

    @property
    def is_conversational(self) -> bool:
        """Si la intencion no tiene forma fija y necesita un modelo de lenguaje."""
        return self.suggested_level is not CascadeLevel.FAST_PATH


class Command(BaseContract):
    """Una orden lista para ejecutarse.

    Es la union entre lo que el usuario dijo literalmente (`raw_text`, que se
    conserva porque el modelo a veces lo necesita entero) y lo que el
    clasificador entendio (`intent`).
    """

    command_id: str = Field(
        default_factory=new_id,
        description="Identificador de esta orden concreta.",
    )
    intent: Intent = Field(
        description="Intencion clasificada.",
    )
    raw_text: str = Field(
        description="El texto tal cual llego, sin normalizar. Nunca se descarta.",
    )
    origin: CommandOrigin = Field(
        description="Por que canal entro la orden.",
    )
    correlation_id: CorrelationId = Field(
        description="Interaccion a la que pertenece.",
    )
    created_at: datetime = Field(
        default_factory=utc_now,
        description="Cuando se construyo la orden.",
    )


class Response(BaseContract):
    """Lo que Jarvis contesta, mas la contabilidad de como lo resolvio.

    `handled_by_level` y `estimated_cost_usd` no son telemetria opcional: son la
    materia prima con la que la Fase 12 afina los umbrales de la cascada, y lo
    que el panel cognitivo del HUD muestra en vivo. Toda respuesta los lleva,
    incluidas las de nivel 1 (donde el costo es exactamente cero).
    """

    text: str = Field(
        description="El texto de la respuesta.",
    )
    should_speak: bool = Field(
        default=True,
        description="Si debe sintetizarse en voz o solo aparecer en el chat del HUD.",
    )
    emotion: Emotion = Field(
        default=Emotion.NEUTRAL,
        description="Intencion expresiva con la que se sintetiza.",
    )
    handled_by_level: CascadeLevel = Field(
        description="Que nivel de la cascada atendio esta respuesta.",
    )
    estimated_cost_usd: float = Field(
        default=0.0,
        ge=0.0,
        description="Costo estimado en dolares. Cero en los niveles 1 y 2.",
    )
    actions: list[ToolCall] = Field(
        default_factory=list,
        description="Herramientas invocadas para producir esta respuesta.",
    )
    streamed: bool = Field(
        default=False,
        description=(
            "Si el texto ya se dijo trozo a trozo por `cognition.response.chunk`. "
            "La voz no lo repite; el chat y la bitacora lo usan entero."
        ),
    )
    correlation_id: CorrelationId = Field(
        description="Interaccion a la que responde.",
    )
    created_at: datetime = Field(
        default_factory=utc_now,
        description="Cuando se genero la respuesta.",
    )

    def model_post_init(self, context: Any, /) -> None:
        """Cobrar por algo que no salio a la nube seria un error de contabilidad."""
        if self.handled_by_level is not CascadeLevel.CLOUD_LLM and self.estimated_cost_usd > 0:
            msg = (
                f"El nivel {self.handled_by_level.name} no tiene costo, pero se "
                f"reporto {self.estimated_cost_usd} USD"
            )
            raise ValueError(msg)

    @property
    def is_spoken_length_ok(self) -> bool:
        """Si la respuesta respeta el limite de dos oraciones para lo hablado.

        La personalidad de Jarvis exige brevedad al hablar: si el usuario quiere
        un ensayo, lo lee en el chat, no lo escucha dictado. Esto no bloquea
        nada -- es una senal para que el orquestador decida partir la respuesta.
        """
        if not self.should_speak:
            return True
        return self.text.count(".") + self.text.count("?") + self.text.count("!") <= 2
