"""El sobre que viaja por el bus de eventos."""

from __future__ import annotations

import builtins
from datetime import datetime
from fnmatch import fnmatchcase
from typing import TypeVar

from pydantic import Field, JsonValue

from jarvis.contracts.base import BaseContract, CorrelationId, new_id, utc_now
from jarvis.contracts.enums import EventType

__all__ = ["Event"]

ContractT = TypeVar("ContractT", bound=BaseContract)


class Event(BaseContract):
    """Un mensaje publicado en el bus.

    DECISION: `payload` es un diccionario serializable y no un modelo generico
    `Event[T]`. Dos razones concretas. Primera, este mismo sobre cruza el
    WebSocket hacia el HUD (Electron) y la app movil (React Native), y un
    generico de Pydantic produce un esquema JSON con referencias anidadas que
    los generadores de tipos de TypeScript manejan mal. Segunda, el bus reparte
    eventos de tipos heterogeneos por el mismo canal; tiparlo con un generico
    obliga a `Event[Any]` en la firma del bus y la ventaja teorica se pierde
    igual.

    A cambio, los consumidores recuperan el tipo con `payload_as(Contrato)`, que
    valida de verdad en vez de confiar.
    """

    event_id: str = Field(
        default_factory=new_id,
        description="Identificador unico de este evento.",
    )
    type: EventType = Field(
        description="Tipo del evento. Su valor con puntos permite suscripcion por glob.",
    )
    payload: dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Carga util ya serializada.",
    )
    source: str = Field(
        min_length=1,
        description="Modulo que lo publico: voice.wake, cognition.router, bus...",
    )
    correlation_id: CorrelationId | None = Field(
        default=None,
        description="Interaccion a la que pertenece. None en eventos de sistema.",
    )
    timestamp: datetime = Field(
        default_factory=utc_now,
        description="Cuando se publico.",
    )

    @classmethod
    def of(
        cls,
        event_type: EventType,
        payload: BaseContract,
        *,
        source: str,
        correlation_id: CorrelationId | None = None,
    ) -> Event:
        """Construye un evento a partir de un contrato, serializandolo.

        Es la forma preferida de publicar: garantiza que la carga util siempre
        tenga la forma de un contrato validado y no un diccionario armado a mano.
        """
        return cls(
            type=event_type,
            payload=payload.to_wire(),
            source=source,
            correlation_id=correlation_id,
        )

    def payload_as(self, contract: builtins.type[ContractT]) -> ContractT:
        """Recupera la carga util como el contrato indicado, validandola.

        Lanza `pydantic.ValidationError` si la carga no encaja. Es intencional:
        un consumidor que recibe un tipo de evento pero una carga con otra forma
        tiene un problema real que debe explotar ahora, no corromper el estado.

        El `builtins.type` no es un adorno: esta clase tiene un campo llamado
        `type`, que dentro del cuerpo de la clase tapa al builtin. Sin calificar,
        la anotacion se lee como "el campo type" y mypy la rechaza.
        """
        return contract.from_wire(dict(self.payload))

    @property
    def age_ms(self) -> float:
        """Milisegundos transcurridos desde que se publico.

        El bus lo usa para medir latencia de entrega por tipo de evento.
        """
        return (utc_now() - self.timestamp).total_seconds() * 1000.0

    def matches(self, pattern: str) -> bool:
        """Si el tipo de este evento encaja con un patron glob como `voice.*`."""
        return fnmatchcase(self.type.value, pattern)

    def __str__(self) -> str:
        """Representacion corta para registros de consola."""
        suffix = f" [{self.correlation_id[:8]}]" if self.correlation_id else ""
        return f"{self.type.value} <- {self.source}{suffix}"
