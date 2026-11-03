"""Nivel 2: el modelo local via Ollama.

Dos modos de llamada:

    estructurado   la conversacion normal. Ollama fuerza el esquema
                   {"confianza", "respuesta"} y devuelve logprobs. La confianza
                   llega primero: con ella el `gate` decide hablar en
                   streaming, generar entera y medir, o cortar y escalar.
    herramientas   cuando la orden necesita una herramienta. Sin esquema (Ollama
                   no combina ambos), sin streaming a voz, y la confianza sale
                   solo de los logprobs.

Confianza combinada = w * exp(media de logprobs) + (1 - w) * autoevaluacion,
con w = `confidence_logprob_weight`. La media de logprobs mide que tan
"natural" le resulto al modelo su propio texto; la autoevaluacion, si sabe lo
que se le pregunta. Cada una por separado se equivoca de formas distintas.

DECISION: qwen3 escribe su razonamiento como texto aunque se le pida
`think: false`, salvo cuando el esquema lo encierra. Por eso el modo
estructurado es el normal y en el de herramientas se agrega `/no_think` y se
limpian etiquetas <think>.
"""

from __future__ import annotations

import json
import math
import re
from typing import TYPE_CHECKING, Any

import httpx2 as httpx

from jarvis.cognition.llm import (
    GateDecision,
    LLMReply,
    LLMRequest,
    LLMUnavailableError,
    ToolUse,
)
from jarvis.cognition.streaming import StructuredStream
from jarvis.contracts.base import new_id
from jarvis.contracts.enums import CascadeLevel
from jarvis.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from jarvis.cognition.llm import ChatMessage
    from jarvis.config import CognitionConfig

__all__ = ["OUTPUT_SCHEMA", "OllamaClient", "combine_confidence"]

_log = get_logger(__name__)

OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "confianza": {"type": "number", "minimum": 0, "maximum": 1},
        "respuesta": {"type": "string"},
    },
    "required": ["confianza", "respuesta"],
}
_THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_PING_TIMEOUT_S = 1.0
_CONNECT_TIMEOUT_S = 2.0


def combine_confidence(logprobs: list[float], self_assessed: float | None, weight: float) -> float:
    """Combina las dos senales. Sin logprobs, manda la autoevaluacion; sin ella, al reves."""
    token = math.exp(sum(logprobs) / len(logprobs)) if logprobs else None
    if token is None and self_assessed is None:
        return 0.0
    if token is None:
        return float(self_assessed or 0.0)
    if self_assessed is None:
        return token
    return weight * token + (1.0 - weight) * self_assessed


class OllamaClient:
    """Cliente de Ollama con streaming, herramientas y confianza propia."""

    def __init__(
        self,
        config: Callable[[], CognitionConfig],
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """Args: config, la seccion vigente; transport, para pruebas sin servidor."""
        self._config = config
        self._transport = transport
        self._client: httpx.AsyncClient | None = None

    @property
    def level(self) -> CascadeLevel:
        """Siempre el nivel 2."""
        return CascadeLevel.LOCAL_LLM

    @property
    def model(self) -> str:
        """El modelo configurado."""
        return self._config().local_model

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            cfg = self._config()
            self._client = httpx.AsyncClient(
                base_url=cfg.ollama_url,
                transport=self._transport,
                timeout=httpx.Timeout(cfg.local_timeout_s, connect=_CONNECT_TIMEOUT_S),
            )
        return self._client

    async def close(self) -> None:
        """Cierra las conexiones."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def available(self) -> bool:
        """Si Ollama responde. No carga el modelo."""
        try:
            response = await self._http().get("/api/version", timeout=_PING_TIMEOUT_S)
        except httpx.HTTPError:
            return False
        return response.status_code == httpx.codes.OK

    async def warm_up(self, system: str) -> bool:
        """Carga el modelo y deja el prompt de sistema en la cache de Ollama.

        La primera peticion en frio tarda segundos; esto la paga al arrancar y
        no cuando el usuario espera.
        """
        cfg = self._config()
        body = {
            "model": cfg.local_model,
            "messages": [{"role": "system", "content": system}],
            "stream": False,
            "think": False,
            "keep_alive": _keep_alive(cfg.local_keep_alive),
            "options": {"num_predict": 1, "num_ctx": cfg.local_num_ctx},
        }
        try:
            response = await self._http().post("/api/chat", json=body)
            response.raise_for_status()
        except httpx.HTTPError as error:
            _log.warning("local_llm.warm_up_failed", error=str(error))
            return False
        _log.info("local_llm.warm", model=cfg.local_model)
        return True

    async def generate(self, request: LLMRequest) -> LLMReply:
        """Genera una respuesta. Lanza `LLMUnavailableError` si Ollama falla."""
        if request.tools:
            return await self._generate_with_tools(request)
        return await self._generate_structured(request)

    # ------------------------------------------------------------------

    def _body(self, request: LLMRequest, *, structured: bool) -> dict[str, Any]:
        cfg = self._config()
        body: dict[str, Any] = {
            "model": cfg.local_model,
            "messages": [_to_ollama(m) for m in request.messages],
            "stream": structured,
            "think": False,
            "logprobs": True,
            "keep_alive": _keep_alive(cfg.local_keep_alive),
            "options": {
                "temperature": cfg.local_temperature,
                "num_predict": request.max_tokens or cfg.local_max_tokens,
                "num_ctx": cfg.local_num_ctx,
            },
        }
        if structured:
            body["format"] = OUTPUT_SCHEMA
        else:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    },
                }
                for t in request.tools
            ]
            first = body["messages"][0]
            if first["role"] == "system":
                first["content"] += "\n/no_think"
        return body

    async def _generate_structured(self, request: LLMRequest) -> LLMReply:
        cfg = self._config()
        parser = StructuredStream()
        answer_logprobs: list[float] = []
        decision: GateDecision | None = None
        early: float | None = None
        streamed = False
        sent = 0
        stats: dict[str, Any] = {}

        try:
            async with self._http().stream(
                "POST", "/api/chat", json=self._body(request, structured=True)
            ) as response:
                await _raise_for_status(response, cfg.local_model)
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    chunk = json.loads(line)
                    if chunk.get("error"):
                        raise LLMUnavailableError(self.level, str(chunk["error"]), spoke=streamed)
                    phase_before = parser.phase
                    new_text = parser.feed(chunk.get("message", {}).get("content", ""))
                    for item in chunk.get("logprobs") or []:
                        value = item.get("logprob")
                        if not isinstance(value, int | float):
                            continue
                        if phase_before == "answer" or new_text:
                            answer_logprobs.append(float(value))

                    if decision is None and parser.confidence is not None:
                        # DECISION: la compuerta usa solo la autoevaluacion. La
                        # probabilidad de los digitos ("0.7" contra "0.8") mide
                        # duda sobre el numero, no sobre la respuesta: medido,
                        # provocaba escaladas de mas.
                        early = parser.confidence
                        decision = request.gate(early) if request.gate else GateDecision.STREAM
                        if decision is GateDecision.ABORT:
                            _log.info("local_llm.gate_abort", early=round(early, 3))
                            break

                    pending = parser.answer[sent:]
                    if pending and decision is GateDecision.STREAM and request.on_text:
                        await request.on_text(pending)
                        sent = len(parser.answer)
                        streamed = True
                    if chunk.get("done"):
                        stats = chunk
                        break
        except LLMUnavailableError:
            raise
        except (httpx.ConnectError, httpx.ConnectTimeout) as error:
            raise LLMUnavailableError(self.level, "Ollama no esta corriendo") from error
        except (httpx.HTTPError, ValueError) as error:
            # Incluye a Ollama muriendo a media respuesta: se corta la lectura y
            # el enrutador decide, sabiendo si ya se alcanzo a hablar.
            raise LLMUnavailableError(
                self.level, f"se corto la respuesta ({type(error).__name__})", spoke=streamed
            ) from error

        self_assessed, answer = parser.finish()
        confidence = combine_confidence(
            answer_logprobs, self_assessed, cfg.confidence_logprob_weight
        )
        if not answer and decision is not GateDecision.ABORT:
            # JSON roto o vacio: no hay respuesta confiable que dar.
            confidence = 0.0
        return LLMReply(
            text=answer.strip(),
            level=self.level,
            model=cfg.local_model,
            confidence=round(confidence, 4),
            early_confidence=None if early is None else round(early, 4),
            gate=decision,
            input_tokens=int(stats.get("prompt_eval_count") or 0),
            output_tokens=int(stats.get("eval_count") or 0),
            streamed=streamed,
            stop_reason="aborted" if decision is GateDecision.ABORT else "end_turn",
        )

    async def _generate_with_tools(self, request: LLMRequest) -> LLMReply:
        cfg = self._config()
        try:
            response = await self._http().post(
                "/api/chat", json=self._body(request, structured=False)
            )
            await _raise_for_status(response, cfg.local_model)
            data = response.json()
        except LLMUnavailableError:
            raise
        except (httpx.ConnectError, httpx.ConnectTimeout) as error:
            raise LLMUnavailableError(self.level, "Ollama no esta corriendo") from error
        except (httpx.HTTPError, ValueError) as error:
            raise LLMUnavailableError(
                self.level, f"fallo la llamada ({type(error).__name__}: {error})"
            ) from error

        message = data.get("message", {})
        calls = [
            ToolUse(
                id=new_id(),
                name=str(call["function"]["name"]),
                arguments=dict(call["function"].get("arguments") or {}),
            )
            for call in message.get("tool_calls") or []
        ]
        logprobs = [
            float(item["logprob"])
            for item in data.get("logprobs") or []
            if isinstance(item.get("logprob"), int | float)
        ]
        text = _THINK.sub("", str(message.get("content", ""))).strip()
        return LLMReply(
            text=text,
            level=self.level,
            model=cfg.local_model,
            tool_calls=calls,
            confidence=round(combine_confidence(logprobs, None, 1.0), 4) if logprobs else None,
            input_tokens=int(data.get("prompt_eval_count") or 0),
            output_tokens=int(data.get("eval_count") or 0),
        )


def _keep_alive(value: str) -> int | str:
    """Ollama acepta -1 como numero para "para siempre" y duraciones como texto."""
    try:
        return int(value)
    except ValueError:
        return value


def _to_ollama(message: ChatMessage) -> dict[str, Any]:
    out: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.tool_calls:
        out["tool_calls"] = [
            {"function": {"name": c.name, "arguments": c.arguments}} for c in message.tool_calls
        ]
    return out


async def _raise_for_status(response: httpx.Response, model: str) -> None:
    if response.status_code == httpx.codes.OK:
        return
    await response.aread()
    detail = response.text[:200]
    if response.status_code == httpx.codes.NOT_FOUND:
        reason = f"el modelo {model} no esta descargado (ollama pull {model})"
    else:
        reason = f"HTTP {response.status_code}: {detail}"
    raise LLMUnavailableError(CascadeLevel.LOCAL_LLM, reason)
