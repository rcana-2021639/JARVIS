"""Nivel 3: la API de Claude. Con costo, solo cuando hace falta.

seleccion de modelo   LIGHT -> Haiku, STANDARD -> Sonnet, DEEP -> Opus.
                      Opus solo si el usuario lo pide o la tarea es de
                      razonamiento excepcional (lo decide el enrutador).
costo                 cada llamada se cobra con la tabla de precios de la
                      configuracion, se suma al dia y se publica al bus.
tope diario           con el dia agotado, ni se intenta: se lanza
                      `LLMUnavailableError` antes de tocar la red.
reintentos            429, 5xx y errores de red se reintentan con espera
                      exponencial, pero solo si todavia no se hablo nada.
privacidad            `network_allowed` se consulta antes de cada llamada:
                      con LOCAL_ONLY no sale ni un byte, aunque el
                      enrutador se equivocara.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, cast

import anthropic

from jarvis.cognition.budget import cost_usd
from jarvis.cognition.llm import Complexity, LLMReply, LLMRequest, LLMUnavailableError, ToolUse
from jarvis.contracts.cascade import CostUpdate
from jarvis.contracts.enums import CascadeLevel
from jarvis.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from jarvis.cognition.budget import SpendLedger
    from jarvis.cognition.llm import ChatMessage
    from jarvis.config import CognitionConfig

__all__ = ["ClaudeClient"]

_log = get_logger(__name__)

_RETRYABLE = (
    anthropic.RateLimitError,
    anthropic.InternalServerError,
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
)
_REFUSAL_TEXT = "No puedo ayudar con eso."


class ClaudeClient:
    """Cliente de la API de Anthropic con streaming, herramientas y contabilidad."""

    def __init__(
        self,
        config: Callable[[], CognitionConfig],
        ledger: SpendLedger,
        api_key: Callable[[], str | None],
        network_allowed: Callable[[], bool],
        on_cost: Callable[[CostUpdate], Awaitable[None]] | None = None,
        *,
        client_factory: Callable[[str], anthropic.AsyncAnthropic] | None = None,
    ) -> None:
        """Prepara el cliente. No abre ninguna conexion hasta la primera llamada.

        Args:
            config: la seccion vigente.
            ledger: el gasto del dia.
            api_key: devuelve la clave (del llavero). None si no hay.
            network_allowed: False en LOCAL_ONLY. Se consulta en cada llamada.
            on_cost: recibe cada gasto para publicarlo en vivo.
            client_factory: construye el cliente del SDK; las pruebas lo cambian.
        """
        self._config = config
        self._ledger = ledger
        self._api_key = api_key
        self._network_allowed = network_allowed
        self._on_cost = on_cost
        self._factory = client_factory or (
            lambda key: anthropic.AsyncAnthropic(api_key=key, max_retries=0)
        )
        self._client: anthropic.AsyncAnthropic | None = None

    @property
    def level(self) -> CascadeLevel:
        """Siempre el nivel 3."""
        return CascadeLevel.CLOUD_LLM

    def unavailable_reason(self) -> str | None:
        """Por que no se puede usar ahora mismo, o None si se puede."""
        if not self._network_allowed():
            return "modo solo-local"
        if not self._ledger.can_spend():
            return "presupuesto del dia agotado"
        if not self._api_key():
            return "no hay clave de la API en el llavero"
        return None

    async def available(self) -> bool:
        """Si se puede llamar sin violar privacidad ni presupuesto. No toca la red."""
        return self.unavailable_reason() is None

    def model_for(self, complexity: Complexity) -> str:
        """Que modelo atiende cada complejidad."""
        cfg = self._config()
        return {
            Complexity.LIGHT: cfg.cloud_model_light,
            Complexity.STANDARD: cfg.cloud_model_standard,
            Complexity.DEEP: cfg.cloud_model_deep,
        }[complexity]

    async def close(self) -> None:
        """Cierra las conexiones."""
        if self._client is not None:
            await self._client.close()
            self._client = None

    async def generate(self, request: LLMRequest) -> LLMReply:
        """Llama a Claude con reintentos. Lanza `LLMUnavailableError` si no puede."""
        reason = self.unavailable_reason()
        if reason is not None:
            raise LLMUnavailableError(self.level, reason)
        cfg = self._config()
        model = self.model_for(request.complexity)
        attempts = cfg.cloud_retries + 1
        for attempt in range(attempts):
            streamed = [False]
            try:
                return await self._call(request, model, streamed)
            except _RETRYABLE as error:
                if streamed[0] or attempt == attempts - 1:
                    raise LLMUnavailableError(
                        self.level,
                        f"la API no responde ({type(error).__name__})",
                        spoke=streamed[0],
                    ) from error
                delay = cfg.cloud_retry_base_s * (2**attempt)
                _log.warning(
                    "cloud_llm.retry",
                    attempt=attempt + 1,
                    delay_s=delay,
                    error=type(error).__name__,
                )
                await asyncio.sleep(delay)
            except anthropic.AuthenticationError as error:
                raise LLMUnavailableError(self.level, "la clave de la API no es valida") from error
            except anthropic.APIStatusError as error:
                raise LLMUnavailableError(
                    self.level, f"la API rechazo la peticion (HTTP {error.status_code})"
                ) from error
        raise LLMUnavailableError(self.level, "sin intentos")  # pragma: no cover

    # ------------------------------------------------------------------

    def _sdk(self) -> anthropic.AsyncAnthropic:
        if self._client is None:
            key = self._api_key()
            if not key:
                raise LLMUnavailableError(self.level, "no hay clave de la API en el llavero")
            self._client = self._factory(key)
        return self._client

    async def _call(self, request: LLMRequest, model: str, streamed: list[bool]) -> LLMReply:
        cfg = self._config()
        system, messages = _to_anthropic(request.messages)
        params: dict[str, Any] = {
            "model": model,
            "max_tokens": request.max_tokens or cfg.cloud_max_tokens,
            "messages": messages,
            "cache_control": {"type": "ephemeral"},
            "output_config": {"effort": _effort(request.complexity, cfg.cloud_effort)},
        }
        if system:
            params["system"] = system
        thinking = _thinking(model)
        if thinking is not None:
            params["thinking"] = thinking
        if request.tools:
            params["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.input_schema}
                for t in request.tools
            ]

        sdk = cast("Any", self._sdk())
        async with sdk.messages.stream(**params) as stream:
            async for event in stream:
                if event.type == "text" and request.on_text is not None:
                    await request.on_text(event.text)
                    streamed[0] = True
            final = await stream.get_final_message()

        usage = final.usage
        input_tokens = int(usage.input_tokens or 0)
        output_tokens = int(usage.output_tokens or 0)
        cost = cost_usd(
            cfg.cloud_prices,
            model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_write_tokens=int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
            cache_read_tokens=int(getattr(usage, "cache_read_input_tokens", 0) or 0),
        )
        await self._charge(model, input_tokens, output_tokens, cost, request.correlation_id)

        text = "".join(block.text for block in final.content if block.type == "text").strip()
        calls = [
            ToolUse(id=block.id, name=block.name, arguments=dict(block.input))
            for block in final.content
            if block.type == "tool_use"
        ]
        if final.stop_reason == "refusal" and not text:
            text = _REFUSAL_TEXT
        return LLMReply(
            text=text,
            level=self.level,
            model=model,
            tool_calls=calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            streamed=streamed[0],
            stop_reason=str(final.stop_reason or "end_turn"),
        )

    async def _charge(
        self, model: str, input_tokens: int, output_tokens: int, cost: float, cid: str | None
    ) -> None:
        result = self._ledger.record(cost)
        _log.info(
            "cloud_llm.cost",
            model=model,
            cost_usd=round(cost, 6),
            spent_today=round(result.spent_today, 4),
        )
        if self._on_cost is not None:
            await self._on_cost(
                CostUpdate(
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost,
                    spent_today_usd=result.spent_today,
                    budget_today_usd=result.budget,
                    budget_exhausted=result.exhausted,
                    correlation_id=cid,
                )
            )


def _effort(complexity: Complexity, configured: str) -> str:
    # Lo profundo se piensa con calma; lo hablado, con el esfuerzo configurado.
    return "high" if complexity is Complexity.DEEP else configured


def _thinking(model: str) -> dict[str, str] | None:
    """El modo de razonamiento que admite cada familia, el mas rapido posible.

    Haiku 5.5 acepta apagarlo; Sonnet 5.5 no, pero `between_tools` lo limita a
    entre herramientas; Opus 5.5 siempre razona (se controla con el esfuerzo).
    """
    if model.startswith("claude-haiku"):
        return {"type": "disabled"}
    if model.startswith("claude-sonnet-5-5"):
        return {"type": "between_tools"}
    return None


def _to_anthropic(messages: list[ChatMessage]) -> tuple[str, list[dict[str, Any]]]:
    """Del formato neutro al de la API: sistema aparte y resultados agrupados."""
    system_parts: list[str] = []
    out: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "system":
            system_parts.append(message.content)
        elif message.role == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": message.tool_call_id,
                "content": message.content,
                "is_error": message.is_error,
            }
            # Todos los resultados de una ronda van en un solo mensaje de usuario.
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
        elif message.role == "assistant" and message.tool_calls:
            blocks: list[dict[str, Any]] = []
            if message.content:
                blocks.append({"type": "text", "text": message.content})
            blocks.extend(
                {"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments}
                for c in message.tool_calls
            )
            out.append({"role": "assistant", "content": blocks})
        else:
            out.append({"role": message.role, "content": message.content})
    return "\n\n".join(system_parts), out
