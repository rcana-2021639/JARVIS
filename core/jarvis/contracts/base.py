"""Base comun de todos los contratos y utilidades de identidad y tiempo."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any, Self

from pydantic import BaseModel, ConfigDict

__all__ = ["BaseContract", "CorrelationId", "new_correlation_id", "new_id", "utc_now"]

CorrelationId = str
"""Identificador que acompana a una interaccion completa de punta a punta.

Nace cuando llega una orden (voz, texto, movil o rutina) y viaja por el
clasificador, el enrutador, cada llamada a herramienta y la bitacora de
auditoria. Es lo que permite reconstruir despues que paso exactamente en una
interaccion concreta a partir de registros dispersos.
"""


def utc_now() -> datetime:
    """Devuelve el instante actual en UTC, siempre con zona horaria explicita.

    Se usa como `default_factory` en todos los contratos. Nunca usamos
    `datetime.now()` sin zona: un timestamp ingenuo que cruza un WebSocket
    hacia el HUD o se guarda en SQLite es un error latente.
    """
    return datetime.now(UTC)


def new_id() -> str:
    """Genera un identificador corto y unico para eventos, llamadas y registros."""
    return uuid.uuid4().hex


def new_correlation_id() -> CorrelationId:
    """Genera el identificador de una interaccion nueva."""
    return uuid.uuid4().hex


class BaseContract(BaseModel):
    """Modelo base de todo contrato que cruza el bus, el disco o la red.

    Las tres decisiones que hereda cualquier contrato:

    `frozen=True` -- los contratos son mensajes, no estado mutable. Una vez
    publicado un evento, nadie puede modificarlo por debajo a otro consumidor.
    Para derivar una version modificada existe `model_copy(update=...)`.

    `extra="forbid"` -- un campo que no esta en el esquema es un error, no algo
    que se ignora en silencio. Si el HUD manda `nivel_sugerido` en vez de
    `suggested_level`, queremos enterarnos al instante y no tres fases despues.

    `use_enum_values=False` -- los enums se conservan como enums en memoria y
    solo se convierten a string al serializar. El codigo compara con
    `CoreState.PENSANDO`, nunca con la cadena `"PENSANDO"`.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        use_enum_values=False,
        str_strip_whitespace=True,
        validate_default=True,
        ser_json_timedelta="float",
    )

    def to_wire(self) -> dict[str, Any]:
        """Serializa a un diccionario listo para JSON, MessagePack o SQLite.

        Los enums salen como su valor string y los datetime como ISO 8601. Es la
        representacion que consumen el HUD (Electron) y la app movil.
        """
        return self.model_dump(mode="json")

    @classmethod
    def from_wire(cls, payload: dict[str, Any]) -> Self:
        """Reconstruye el contrato desde su forma serializada, validando todo."""
        return cls.model_validate(payload)
