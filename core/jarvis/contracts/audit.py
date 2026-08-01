"""Contrato de la bitacora de auditoria."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field, JsonValue

from jarvis.contracts.base import BaseContract, CorrelationId, new_id, utc_now
from jarvis.contracts.enums import CommandOrigin, RiskLevel

__all__ = ["AuditEntry"]

_REDACTED = "<credencial:{site}>"


class AuditEntry(BaseContract):
    """Una linea de la bitacora de solo-anadir.

    Es lo que convierte "acceso total" en algo auditable en vez de algo que hay
    que creer. Cada accion que Jarvis ejecuta deja una de estas, incluidas las
    de nivel VERDE que no piden permiso.

    El campo `previous_hash` prepara el encadenamiento de hashes de la Fase 8:
    cada entrada referencia el hash de la anterior, de modo que alterar una
    entrada vieja invalida toda la cadena posterior y se nota.

    REGLA ABSOLUTA: `arguments` va saneado. El valor de una credencial jamas
    entra aqui -- se sustituye por `<credencial:sitio>` antes de construir la
    entrada. Ver `redact_credential`.
    """

    entry_id: str = Field(
        default_factory=new_id,
        description="Identificador de esta entrada.",
    )
    action: str = Field(
        min_length=1,
        description="Que se hizo, en forma `servidor.herramienta`.",
    )
    arguments: dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Argumentos SANEADOS. Nunca valores de credenciales.",
    )
    level: RiskLevel = Field(
        description="Nivel con el que se evaluo la accion.",
    )
    origin: CommandOrigin = Field(
        description="Por que canal llego la orden que la origino.",
    )
    confirmed: bool = Field(
        default=False,
        description="Si el usuario la confirmo hablando. Obligatorio en nivel ROJO.",
    )
    success: bool = Field(
        description="Si la accion termino bien.",
    )
    result: str | None = Field(
        default=None,
        description="Resumen legible del resultado o del error.",
    )
    duration_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Cuanto tardo.",
    )
    correlation_id: CorrelationId = Field(
        description="Interaccion que la origino.",
    )
    timestamp: datetime = Field(
        default_factory=utc_now,
        description="Cuando ocurrio.",
    )
    previous_hash: str | None = Field(
        default=None,
        description="Hash de la entrada anterior. Encadena la bitacora (Fase 8).",
    )

    def model_post_init(self, context: Any, /) -> None:
        """Una accion ROJA sin confirmacion registrada seria una bitacora que miente."""
        if self.level is RiskLevel.ROJO and self.success and not self.confirmed:
            msg = (
                f"La accion '{self.action}' es ROJO y se registro como exitosa sin "
                f"confirmacion. Una accion ROJA solo se ejecuta tras confirmacion hablada."
            )
            raise ValueError(msg)
        if self.level is RiskLevel.NEGRO and self.success:
            msg = (
                f"La accion '{self.action}' es NEGRO y se registro como exitosa. "
                f"El nivel NEGRO se rechaza siempre, sin excepcion posible."
            )
            raise ValueError(msg)

    @staticmethod
    def redact_credential(site: str) -> str:
        """Devuelve el marcador que sustituye a una credencial en los registros.

        El valor real nunca pasa por aqui: el modulo de boveda entrega la
        credencial directamente al navegador y solo informa a que sitio
        correspondia. Esta funcion existe para que ese marcador tenga una sola
        forma canonica en todo el sistema.
        """
        return _REDACTED.format(site=site)
