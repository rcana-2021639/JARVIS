"""Contratos del bus de herramientas (servidores MCP)."""

from __future__ import annotations

from typing import Any

from pydantic import Field, JsonValue

from jarvis.contracts.base import BaseContract, CorrelationId, new_id
from jarvis.contracts.enums import RiskLevel

__all__ = ["ToolCall", "ToolResult"]


class ToolCall(BaseContract):
    """La invocacion de una herramienta expuesta por un servidor MCP.

    Cada herramienta declara obligatoriamente su nivel de riesgo. El Guardian
    (Fase 8) lee `risk_level` antes de dejar pasar la llamada y puede subirlo
    segun los argumentos concretos: borrar un temporal es AMBAR, borrar algo en
    Documentos es ROJO.
    """

    call_id: str = Field(
        default_factory=new_id,
        description="Identificador de esta invocacion. Lo referencia el ToolResult.",
    )
    server: str = Field(
        min_length=1,
        description="Servidor MCP que expone la herramienta: files, apps, shell, browser...",
    )
    tool: str = Field(
        min_length=1,
        description="Nombre de la herramienta dentro del servidor.",
    )
    arguments: dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Argumentos ya validados contra el esquema de entrada de la herramienta.",
    )
    risk_level: RiskLevel = Field(
        description="Nivel declarado por la herramienta, antes de las reglas contextuales.",
    )
    correlation_id: CorrelationId = Field(
        description="Interaccion que origino esta llamada.",
    )

    @property
    def qualified_name(self) -> str:
        """Nombre completo `servidor.herramienta`, tal como aparece en la bitacora."""
        return f"{self.server}.{self.tool}"


class ToolResult(BaseContract):
    """El resultado de ejecutar una herramienta.

    Nunca se lanza una excepcion a traves del bus: un fallo es un ToolResult con
    `success=False` y `error` poblado. Asi el orquestador puede decidir si
    reintenta con una estrategia alterna (Fase 12) sin envolver todo en
    try/except, y el fallo queda registrado igual que un exito.
    """

    call_id: str = Field(
        description="Identificador de la ToolCall que produjo este resultado.",
    )
    success: bool = Field(
        description="Si la herramienta cumplio su cometido.",
    )
    data: JsonValue = Field(
        default=None,
        description="Datos devueltos por la herramienta. None si no devuelve nada.",
    )
    error: str | None = Field(
        default=None,
        description="Mensaje de error legible. Obligatorio cuando success es False.",
    )
    duration_ms: float = Field(
        ge=0.0,
        description="Cuanto tardo la ejecucion, en milisegundos.",
    )

    def model_post_init(self, context: Any, /) -> None:
        """Un fallo sin explicacion viola el principio de que nada falla en silencio."""
        if not self.success and not self.error:
            msg = "Un ToolResult con success=False debe traer 'error' explicando que paso"
            raise ValueError(msg)
        if self.success and self.error:
            msg = "Un ToolResult con success=True no puede traer 'error'"
            raise ValueError(msg)
