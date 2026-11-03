"""La interfaz comun de los niveles 2 y 3.

El enrutador, el bucle de herramientas y las pruebas hablan con un
`LanguageModel` sin saber si detras hay Ollama, la API de Claude o un falso
determinista. Los tipos de aqui son internos del nucleo: no cruzan el bus, por
eso son dataclasses y no contratos Pydantic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal, Protocol

from jarvis.contracts.enums import CascadeLevel

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

__all__ = [
    "ChatMessage",
    "Complexity",
    "GateDecision",
    "LLMReply",
    "LLMRequest",
    "LLMUnavailableError",
    "LanguageModel",
    "LocalModel",
    "ToolDefinition",
    "ToolUse",
]

Role = Literal["system", "user", "assistant", "tool"]


class GateDecision(StrEnum):
    """Que hacer con una respuesta local segun su confianza temprana."""

    STREAM = "STREAM"
    """Confianza holgada: decir cada frase en cuanto exista."""

    BUFFER = "BUFFER"
    """Confianza al limite: generar entera, medirla, y solo entonces decidir."""

    ABORT = "ABORT"
    """Confianza baja: cortar ya y escalar. No se dice nada."""


class Complexity(StrEnum):
    """Que tan pesada es una tarea. Elige el modelo de la nube."""

    LIGHT = "LIGHT"
    STANDARD = "STANDARD"
    DEEP = "DEEP"


@dataclass(frozen=True, slots=True)
class ToolUse:
    """Una llamada a herramienta que pidio el modelo."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """Un turno de la conversacion en el formato neutro de la cascada."""

    role: Role
    content: str
    tool_calls: tuple[ToolUse, ...] = ()
    tool_call_id: str | None = None
    """En un mensaje `tool`: a que llamada responde."""
    is_error: bool = False
    """En un mensaje `tool`: si la herramienta fallo."""


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """Lo que el modelo ve de una herramienta. Misma forma que una tool de MCP."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(slots=True)
class LLMRequest:
    """Una peticion a un modelo."""

    messages: list[ChatMessage]
    tools: Sequence[ToolDefinition] = ()
    complexity: Complexity = Complexity.STANDARD
    max_tokens: int | None = None
    on_text: Callable[[str], Awaitable[None]] | None = None
    """Recibe la respuesta en trozos conforme se genera."""
    gate: Callable[[float], GateDecision] | None = None
    """Solo el nivel 2: decide con la confianza temprana si hablar, esperar o cortar."""
    correlation_id: str | None = None
    """Interaccion que paga esta llamada; acompana al gasto publicado."""


@dataclass(slots=True)
class LLMReply:
    """Lo que devolvio un modelo."""

    text: str
    level: CascadeLevel
    model: str
    tool_calls: list[ToolUse] = field(default_factory=list)
    confidence: float | None = None
    """Confianza combinada (logprobs + autoevaluacion). None si el modelo no la da."""
    early_confidence: float | None = None
    gate: GateDecision | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    streamed: bool = False
    """Si el texto ya salio por `on_text`."""
    stop_reason: str = "end_turn"


class LLMUnavailableError(RuntimeError):
    """Un nivel no pudo contestar: apagado, sin red, sin clave, sin presupuesto.

    Nunca es silencioso: el enrutador lo convierte en una escalada, una
    degradacion o un aviso hablado honesto.
    """

    def __init__(self, level: CascadeLevel, reason: str, *, spoke: bool = False) -> None:
        """Args: level, el nivel que fallo; reason, por que; spoke, si ya alcanzo a hablar."""
        super().__init__(f"Nivel {level.name} no disponible: {reason}")
        self.level = level
        self.reason = reason
        self.spoke = spoke


class LanguageModel(Protocol):
    """Un nivel de la cascada que habla con un modelo de lenguaje."""

    @property
    def level(self) -> CascadeLevel:
        """El nivel que atiende: 2 o 3."""
        ...

    async def generate(self, request: LLMRequest) -> LLMReply:
        """Genera una respuesta. Lanza `LLMUnavailableError` si no puede."""
        ...

    async def available(self) -> bool:
        """Si en este momento puede atender, sin gastar nada."""
        ...


class LocalModel(LanguageModel, Protocol):
    """El nivel 2: ademas de generar, se precalienta y dice que modelo usa."""

    @property
    def model(self) -> str:
        """El modelo configurado."""
        ...

    async def warm_up(self, system: str) -> bool:
        """Carga el modelo y su prompt de sistema antes de la primera orden."""
        ...
