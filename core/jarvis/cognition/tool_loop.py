"""El bucle de herramientas: pedir al modelo, ejecutar, devolver, repetir.

Cuatro protecciones:

    tope de iteraciones   `max_tool_iterations`; al llegar se pide una
                          respuesta final sin herramientas
    deteccion de bucles   la misma herramienta con los mismos argumentos dos
                          veces no se ejecuta otra vez: se corta el ciclo
    tiempo limite         cada herramienta tiene el suyo; nada cuelga el bucle
    estrategia alterna    un fallo por tiempo se reintenta con el doble de
                          plazo; si la herramienta declara una alterna se usa;
                          y el error vuelve al modelo con la orden de cambiar
                          de enfoque en vez de repetir

DECISION: `ToolSpec` tiene la misma forma que una herramienta de MCP (nombre,
descripcion, esquema de entrada). En la Fase 6 cada servidor MCP se registra
aqui como un `ToolProvider` sin tocar el bucle.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import JsonValue, TypeAdapter, ValidationError

from jarvis.cognition.llm import ChatMessage, LLMReply, LLMRequest, ToolDefinition, ToolUse
from jarvis.contracts.enums import EventType, RiskLevel
from jarvis.contracts.events import Event
from jarvis.contracts.tools import ToolCall, ToolResult
from jarvis.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from jarvis.bus.base import EventBus
    from jarvis.cognition.llm import LanguageModel

__all__ = ["ToolLoop", "ToolLoopResult", "ToolRegistry", "ToolSpec"]

_log = get_logger(__name__)
_JSON = TypeAdapter(JsonValue)
_ARGS = TypeAdapter(dict[str, JsonValue])


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """Una herramienta registrada."""

    server: str
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], Awaitable[JsonValue]]
    risk: RiskLevel = RiskLevel.VERDE
    timeout_s: float | None = None
    fallback: str | None = None
    """Herramienta alterna (`servidor.nombre`) a probar si esta falla."""

    @property
    def qualified_name(self) -> str:
        """`servidor.nombre`, como la ve el modelo y la bitacora."""
        return f"{self.server}.{self.name}"

    def definition(self) -> ToolDefinition:
        """Lo que se le muestra al modelo."""
        # Los nombres de herramienta de las APIs no admiten puntos.
        return ToolDefinition(
            name=self.qualified_name.replace(".", "__"),
            description=self.description,
            input_schema=self.input_schema,
        )


class ToolRegistry:
    """Las herramientas disponibles y su ejecucion con plazo y eventos."""

    def __init__(self, bus: EventBus | None, default_timeout_s: Callable[[], float]) -> None:
        """Args: bus, donde se publican inicio y fin; default_timeout_s, plazo vigente."""
        self._bus = bus
        self._default_timeout = default_timeout_s
        self._tools: dict[str, ToolSpec] = {}

    def register(self, tool: ToolSpec) -> None:
        """Agrega o reemplaza una herramienta."""
        self._tools[tool.qualified_name] = tool

    def get(self, name: str) -> ToolSpec | None:
        """Por nombre calificado (`demo.calculator`) o como lo ve el modelo."""
        return self._tools.get(name) or self._tools.get(name.replace("__", "."))

    def definitions(self) -> list[ToolDefinition]:
        """Lo que se le pasa al modelo, en orden estable (no invalida la cache)."""
        return [self._tools[key].definition() for key in sorted(self._tools)]

    def default_timeout_s(self) -> float:
        """El plazo vigente para herramientas que no declaran el suyo."""
        return self._default_timeout()

    def __len__(self) -> int:
        """Cuantas herramientas hay."""
        return len(self._tools)

    async def call(
        self,
        name: str,
        arguments: dict[str, Any],
        correlation_id: str,
        *,
        timeout_s: float | None = None,
    ) -> tuple[ToolCall | None, ToolResult]:
        """Ejecuta una herramienta. Nunca lanza: un fallo es un `ToolResult`."""
        tool = self.get(name)
        if tool is None:
            # Nada falla en silencio: pedir algo que no existe es un fallo con nombre.
            return None, ToolResult(
                call_id="sin-llamada",
                success=False,
                error=f"La herramienta {name} no esta disponible todavia",
                duration_ms=0.0,
            )
        call = ToolCall(
            server=tool.server,
            tool=tool.name,
            arguments=_ARGS.validate_python(arguments),
            risk_level=tool.risk,
            correlation_id=correlation_id,
        )
        await self._publish(EventType.TOOL_CALL_STARTED, call, correlation_id)
        limit = timeout_s or tool.timeout_s or self._default_timeout()
        started = time.perf_counter()
        try:
            data = await asyncio.wait_for(tool.handler(arguments), limit)
            result = ToolResult(
                call_id=call.call_id,
                success=True,
                data=_JSON.validate_python(data),
                duration_ms=(time.perf_counter() - started) * 1000.0,
            )
        except TimeoutError:
            result = self._failure(call, started, f"Tardo mas de {limit:g} s")
        except (ValidationError, ValueError, TypeError, KeyError) as error:
            result = self._failure(call, started, f"Argumentos invalidos: {error}")
        except Exception as error:
            _log.exception("tool.failed", tool=tool.qualified_name)
            result = self._failure(call, started, f"{type(error).__name__}: {error}")
        await self._publish(EventType.TOOL_CALL_FINISHED, result, correlation_id)
        return call, result

    @staticmethod
    def _failure(call: ToolCall, started: float, error: str) -> ToolResult:
        return ToolResult(
            call_id=call.call_id,
            success=False,
            error=error,
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )

    async def _publish(
        self, event_type: EventType, payload: ToolCall | ToolResult, correlation_id: str
    ) -> None:
        if self._bus is not None:
            await self._bus.publish(
                Event.of(
                    event_type, payload, source="cognition.tools", correlation_id=correlation_id
                )
            )


@dataclass(slots=True)
class ToolLoopResult:
    """Lo que salio del bucle."""

    reply: LLMReply
    calls: list[ToolCall] = field(default_factory=list)
    results: list[ToolResult] = field(default_factory=list)
    iterations: int = 0
    stopped_by: str | None = None
    """None si el modelo termino solo; `max_iterations` o `loop` si se le corto."""


_ALTERNATE_HINT = (
    "La herramienta fallo: {error}. No repitas la misma llamada. Prueba otra "
    "herramienta, otros argumentos, o explica al usuario que no se pudo."
)
_WRAP_UP = (
    "Ya no hay mas herramientas disponibles en esta respuesta. Contesta al usuario "
    "con lo que tienes, y si algo no se pudo hacer, dilo."
)


class ToolLoop:
    """Ciclo de llamada a herramientas sobre cualquier `LanguageModel`."""

    def __init__(self, registry: ToolRegistry, max_iterations: Callable[[], int]) -> None:
        """Args: registry, las herramientas; max_iterations, el tope vigente."""
        self._registry = registry
        self._max_iterations = max_iterations

    async def run(
        self, model: LanguageModel, request: LLMRequest, correlation_id: str
    ) -> ToolLoopResult:
        """Corre el ciclo hasta que el modelo conteste sin pedir herramientas."""
        outcome = ToolLoopResult(reply=LLMReply(text="", level=model.level, model=""))
        seen: set[str] = set()
        messages = list(request.messages)
        limit = self._max_iterations()
        tools = list(request.tools) or self._registry.definitions()

        while True:
            if outcome.iterations >= limit:
                outcome.stopped_by = "max_iterations"
                break
            outcome.iterations += 1
            step = LLMRequest(
                messages=messages,
                tools=tools,
                complexity=request.complexity,
                max_tokens=request.max_tokens,
                on_text=request.on_text,
                gate=request.gate,
            )
            reply = await model.generate(step)
            self._accumulate(outcome, reply)
            if not reply.tool_calls:
                return outcome

            messages.append(
                ChatMessage("assistant", reply.text, tool_calls=tuple(reply.tool_calls))
            )
            repeated = False
            for use in reply.tool_calls:
                signature = f"{use.name}:{json.dumps(use.arguments, sort_keys=True, default=str)}"
                if signature in seen:
                    repeated = True
                    messages.append(
                        _tool_message(use, "Ya llamaste a esto con lo mismo.", error=True)
                    )
                    continue
                seen.add(signature)
                messages.append(await self._execute(use, correlation_id, outcome))
            if repeated:
                outcome.stopped_by = "loop"
                _log.warning("tool_loop.repeated_call", correlation_id=correlation_id)
                break

        # Cortado por tope o por bucle: una ultima vuelta sin herramientas.
        messages.append(ChatMessage("user", _WRAP_UP))
        final = await model.generate(
            LLMRequest(
                messages=messages,
                complexity=request.complexity,
                max_tokens=request.max_tokens,
                on_text=request.on_text,
            )
        )
        self._accumulate(outcome, final)
        return outcome

    async def _execute(
        self, use: ToolUse, correlation_id: str, outcome: ToolLoopResult
    ) -> ChatMessage:
        call, result = await self._registry.call(use.name, use.arguments, correlation_id)
        if not result.success and result.error and result.error.startswith("Tardo"):
            # Estrategia alterna 1: un tiempo agotado merece una segunda
            # oportunidad con el doble de plazo antes de rendirse.
            tool = self._registry.get(use.name)
            if tool is not None:
                retry_timeout = 2 * (tool.timeout_s or self._registry.default_timeout_s())
                call, result = await self._registry.call(
                    use.name, use.arguments, correlation_id, timeout_s=retry_timeout
                )
        if not result.success:
            tool = self._registry.get(use.name)
            if tool is not None and tool.fallback and self._registry.get(tool.fallback):
                # Estrategia alterna 2: la herramienta declara una sustituta.
                _log.info("tool_loop.fallback", tool=use.name, fallback=tool.fallback)
                call, result = await self._registry.call(
                    tool.fallback, use.arguments, correlation_id
                )
        if call is not None:
            outcome.calls.append(call)
        outcome.results.append(result)
        if result.success:
            return _tool_message(use, json.dumps(result.data, ensure_ascii=False, default=str))
        # Estrategia alterna 3: el modelo se entera y debe cambiar de enfoque.
        return _tool_message(use, _ALTERNATE_HINT.format(error=result.error), error=True)

    @staticmethod
    def _accumulate(outcome: ToolLoopResult, reply: LLMReply) -> None:
        previous = outcome.reply
        reply.input_tokens += previous.input_tokens
        reply.output_tokens += previous.output_tokens
        reply.cost_usd += previous.cost_usd
        reply.streamed = reply.streamed or previous.streamed
        outcome.reply = reply


def _tool_message(use: ToolUse, content: str, *, error: bool = False) -> ChatMessage:
    return ChatMessage("tool", content, tool_call_id=use.id, is_error=error)
