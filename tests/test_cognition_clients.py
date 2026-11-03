"""Pruebas de los clientes de los niveles 2 y 3 y del bucle de herramientas.

Sin red: Ollama se simula con un transporte de httpx2 y la API de Claude con un
SDK falso que reproduce la forma de sus eventos de streaming.
"""

from __future__ import annotations

import asyncio
import json
import socket
from dataclasses import dataclass, field
from typing import Any

import anthropic
import httpx2 as httpx
import pytest

from jarvis.cognition.budget import SpendLedger
from jarvis.cognition.cloud_llm import ClaudeClient
from jarvis.cognition.demo_tools import demo_tools, safe_arithmetic
from jarvis.cognition.llm import (
    ChatMessage,
    Complexity,
    GateDecision,
    LLMReply,
    LLMRequest,
    LLMUnavailableError,
    ToolUse,
)
from jarvis.cognition.local_llm import OllamaClient
from jarvis.cognition.tool_loop import ToolLoop, ToolRegistry, ToolSpec
from jarvis.config import CognitionConfig
from jarvis.contracts import CascadeLevel, EventType

# ---------------------------------------------------------------------------
# Nivel 2: Ollama simulado
# ---------------------------------------------------------------------------


def _ndjson(pieces: list[str], *, logprob: float = -0.1) -> bytes:
    lines = [
        json.dumps(
            {
                "message": {"content": p},
                "logprobs": [{"token": p, "logprob": logprob}],
                "done": False,
            }
        )
        for p in pieces
    ]
    lines.append(
        json.dumps(
            {
                "message": {"content": ""},
                "done": True,
                "eval_count": len(pieces),
                "prompt_eval_count": 40,
            }
        )
    )
    return ("\n".join(lines) + "\n").encode()


_ANSWER = ['{"confianza": ', "0.9", ', "respuesta": "', "Me gusta", " el jazz.", " Mucho.", '"}']


class _DyingStream(httpx.AsyncByteStream):
    """Entrega unas lineas y luego la conexion muere, como Ollama al apagarse."""

    def __init__(self, body: bytes, lines_before_death: int) -> None:
        self._lines = body.splitlines(keepends=True)[:lines_before_death]

    async def __aiter__(self):
        for line in self._lines:
            yield line
        raise httpx.ReadError("conexion perdida")


def _ollama(handler) -> OllamaClient:
    config = CognitionConfig()
    return OllamaClient(lambda: config, transport=httpx.MockTransport(handler))


def _request(**kwargs: Any) -> LLMRequest:
    return LLMRequest(messages=[ChatMessage("system", "s"), ChatMessage("user", "hola")], **kwargs)


async def test_ollama_habla_en_streaming_con_confianza_alta():
    seen_body: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen_body.update(json.loads(request.content))
        return httpx.Response(200, content=_ndjson(_ANSWER))

    spoken: list[str] = []

    async def on_text(delta: str) -> None:
        spoken.append(delta)

    client = _ollama(handler)
    reply = await client.generate(_request(on_text=on_text, gate=lambda early: GateDecision.STREAM))
    await client.close()
    assert "".join(spoken) == "Me gusta el jazz. Mucho."
    assert reply.streamed and reply.gate is GateDecision.STREAM
    assert reply.confidence is not None and reply.confidence > 0.85
    assert reply.output_tokens == len(_ANSWER)
    # Lo que exige el documento: esquema con autoevaluacion, logprobs, modelo en RAM.
    assert seen_body["format"]["required"] == ["confianza", "respuesta"]
    assert seen_body["logprobs"] is True
    assert seen_body["keep_alive"] == -1
    assert seen_body["think"] is False


async def test_ollama_corta_si_la_compuerta_aborta():
    client = _ollama(lambda r: httpx.Response(200, content=_ndjson(_ANSWER)))
    spoken: list[str] = []

    async def on_text(delta: str) -> None:
        spoken.append(delta)

    reply = await client.generate(_request(on_text=on_text, gate=lambda e: GateDecision.ABORT))
    assert reply.gate is GateDecision.ABORT
    assert spoken == []
    assert reply.early_confidence is not None


async def test_ollama_buffer_no_habla_pero_devuelve_todo():
    client = _ollama(lambda r: httpx.Response(200, content=_ndjson(_ANSWER, logprob=-2.0)))
    spoken: list[str] = []

    async def on_text(delta: str) -> None:
        spoken.append(delta)

    reply = await client.generate(_request(on_text=on_text, gate=lambda e: GateDecision.BUFFER))
    assert spoken == []
    assert not reply.streamed
    assert reply.text == "Me gusta el jazz. Mucho."
    # Logprobs bajos arrastran la confianza combinada hacia abajo.
    assert reply.confidence is not None and reply.confidence < 0.6


async def test_ollama_apagado_no_crashea():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("rechazada", request=request)

    client = _ollama(handler)
    assert not await client.available()
    with pytest.raises(LLMUnavailableError, match="no esta corriendo") as caught:
        await client.generate(_request())
    assert caught.value.level is CascadeLevel.LOCAL_LLM
    assert not caught.value.spoke


async def test_ollama_muriendo_a_media_respuesta_avisa_que_ya_hablo():
    body = _ndjson(_ANSWER)
    client = _ollama(lambda r: httpx.Response(200, stream=_DyingStream(body, 5)))
    spoken: list[str] = []

    async def on_text(delta: str) -> None:
        spoken.append(delta)

    with pytest.raises(LLMUnavailableError) as caught:
        await client.generate(_request(on_text=on_text, gate=lambda e: GateDecision.STREAM))
    assert caught.value.spoke
    assert "".join(spoken).startswith("Me gusta")


async def test_ollama_modelo_no_descargado():
    client = _ollama(lambda r: httpx.Response(404, json={"error": "model not found"}))
    with pytest.raises(LLMUnavailableError, match="ollama pull"):
        await client.generate(_request())


async def test_ollama_herramientas_sin_esquema_y_sin_think():
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "message": {
                    "content": "<think>razono</think>",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "demo__calculator",
                                "arguments": {"expression": "2+2"},
                            }
                        }
                    ],
                },
                "logprobs": [{"logprob": -0.05}],
                "eval_count": 9,
            },
        )

    from jarvis.cognition.llm import ToolDefinition

    reply = await _ollama(handler).generate(
        _request(tools=[ToolDefinition("demo__calculator", "calcula", {"type": "object"})])
    )
    assert "format" not in seen and seen["tools"][0]["function"]["name"] == "demo__calculator"
    assert seen["messages"][0]["content"].endswith("/no_think")
    assert reply.text == ""
    assert reply.tool_calls[0].arguments == {"expression": "2+2"}


# ---------------------------------------------------------------------------
# Nivel 3: SDK de Anthropic falso
# ---------------------------------------------------------------------------


@dataclass
class _Usage:
    input_tokens: int = 1000
    output_tokens: int = 200
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


@dataclass
class _Block:
    type: str
    text: str = ""
    id: str = ""
    name: str = ""
    input: dict[str, Any] = field(default_factory=dict)


@dataclass
class _Message:
    content: list[_Block]
    usage: _Usage = field(default_factory=_Usage)
    stop_reason: str = "end_turn"


@dataclass
class _Event:
    type: str
    text: str = ""


class _Stream:
    def __init__(self, sdk: _FakeSDK, text: str) -> None:
        self._sdk = sdk
        self._text = text

    async def __aenter__(self) -> _Stream:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    def __aiter__(self):
        return self._events()

    async def _events(self):
        for word in self._text.split(" "):
            yield _Event("text", word + " ")

    async def get_final_message(self) -> _Message:
        return _Message([_Block("text", self._text)])


class _FakeSDK:
    def __init__(self, text: str = "Respuesta de la nube.", failures: int = 0) -> None:
        self.calls: list[dict[str, Any]] = []
        self._text = text
        self._failures = failures
        self.messages = self

    def stream(self, **params: Any) -> _Stream:
        self.calls.append(params)
        if self._failures > 0:
            self._failures -= 1
            raise anthropic.APIConnectionError(request=httpx.Request("POST", "https://api"))
        return _Stream(self, self._text)

    async def close(self) -> None:
        return None


def _claude(
    sdk: _FakeSDK,
    *,
    key: str | None = "sk-prueba",
    allowed: bool = True,
    budget: float = 1.0,
    costs: list[Any] | None = None,
) -> tuple[ClaudeClient, SpendLedger]:
    config = CognitionConfig(cloud_retry_base_s=0.001)
    ledger = SpendLedger(None, lambda: budget)

    async def on_cost(update: Any) -> None:
        if costs is not None:
            costs.append(update)

    client = ClaudeClient(
        lambda: config,
        ledger,
        lambda: key,
        lambda: allowed,
        on_cost,
        client_factory=lambda k: sdk,  # type: ignore[arg-type,return-value]
    )
    return client, ledger


async def test_claude_cobra_y_publica_el_costo():
    costs: list[Any] = []
    sdk = _FakeSDK()
    client, ledger = _claude(sdk, costs=costs)
    spoken: list[str] = []

    async def on_text(delta: str) -> None:
        spoken.append(delta)

    reply = await client.generate(
        _request(complexity=Complexity.LIGHT, on_text=on_text, correlation_id="c1")
    )
    # Haiku 5.5: 1000 * 0.10 + 200 * 0.50 por millon.
    assert reply.cost_usd == pytest.approx((1000 * 0.10 + 200 * 0.50) / 1e6)
    assert ledger.spent_today == pytest.approx(reply.cost_usd)
    assert costs[0].model == "claude-haiku-5-5" and costs[0].correlation_id == "c1"
    assert reply.streamed and "".join(spoken).strip() == "Respuesta de la nube."
    params = sdk.calls[0]
    assert params["thinking"] == {"type": "disabled"}
    assert params["output_config"] == {"effort": "low"}
    assert "temperature" not in params
    assert params["system"] == "s"


@pytest.mark.parametrize(
    ("complexity", "model"),
    [
        (Complexity.LIGHT, "claude-haiku-5-5"),
        (Complexity.STANDARD, "claude-sonnet-5-5"),
        (Complexity.DEEP, "claude-opus-5-5"),
    ],
)
async def test_claude_elige_modelo_por_complejidad(complexity, model):
    sdk = _FakeSDK()
    client, _ = _claude(sdk)
    reply = await client.generate(_request(complexity=complexity))
    assert reply.model == model == sdk.calls[0]["model"]


async def test_claude_reintenta_con_retroceso():
    sdk = _FakeSDK(failures=2)
    client, _ = _claude(sdk)
    reply = await client.generate(_request())
    assert len(sdk.calls) == 3
    assert reply.text == "Respuesta de la nube."


async def test_claude_se_rinde_tras_los_reintentos():
    sdk = _FakeSDK(failures=10)
    client, _ = _claude(sdk)
    with pytest.raises(LLMUnavailableError, match="no responde"):
        await client.generate(_request())
    assert len(sdk.calls) == 4  # 1 + 3 reintentos


async def test_claude_sin_presupuesto_no_toca_la_red():
    sdk = _FakeSDK()
    client, ledger = _claude(sdk, budget=0.01)
    ledger.record(0.02)
    with pytest.raises(LLMUnavailableError, match="presupuesto"):
        await client.generate(_request())
    assert sdk.calls == []


async def test_claude_sin_clave_no_esta_disponible():
    client, _ = _claude(_FakeSDK(), key=None)
    assert not await client.available()
    assert "clave" in (client.unavailable_reason() or "")


async def test_local_only_no_abre_ni_un_socket(monkeypatch):
    """Con la red prohibida, el cliente real del SDK ni siquiera se construye."""
    attempts: list[Any] = []
    real_connect = socket.socket.connect

    def spy(self: socket.socket, address: Any) -> Any:
        attempts.append(address)
        return real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", spy)
    config = CognitionConfig()
    client = ClaudeClient(
        lambda: config, SpendLedger(None, lambda: 1.0), lambda: "sk-x", lambda: False
    )
    with pytest.raises(LLMUnavailableError, match="solo-local"):
        await client.generate(_request())
    assert attempts == []


# ---------------------------------------------------------------------------
# Bucle de herramientas
# ---------------------------------------------------------------------------


class _ToolCaller:
    """Un modelo que pide herramientas segun un guion y despues contesta."""

    level = CascadeLevel.LOCAL_LLM

    def __init__(self, plan: list[list[ToolUse]], final: str = "Listo.") -> None:
        self._plan = list(plan)
        self._final = final
        self.requests: list[LLMRequest] = []

    async def available(self) -> bool:
        return True

    async def generate(self, request: LLMRequest) -> LLMReply:
        self.requests.append(request)
        if self._plan and request.tools:
            return LLMReply(text="", level=self.level, model="m", tool_calls=self._plan.pop(0))
        return LLMReply(text=self._final, level=self.level, model="m")


def _use(name: str, **args: Any) -> ToolUse:
    return ToolUse(id=f"id-{name}-{len(args)}-{sorted(args.items())}", name=name, arguments=args)


def _registry(bus=None) -> ToolRegistry:
    registry = ToolRegistry(bus, lambda: 0.2)
    for tool in demo_tools():
        registry.register(tool)
    return registry


async def test_bucle_ejecuta_la_calculadora_y_devuelve_el_resultado(bus):
    started: list[Any] = []

    async def on_tool(event):
        started.append(event)

    bus.subscribe("tool.*", on_tool)
    registry = _registry(bus)
    model = _ToolCaller([[_use("demo__calculator", expression="(25*4)+3")]], final="Da 103.")
    outcome = await ToolLoop(registry, lambda: 5).run(model, _request(), "cid")
    assert outcome.reply.text == "Da 103."
    assert outcome.results[0].success and outcome.results[0].data["result"] == 103
    tool_message = model.requests[1].messages[-1]
    assert tool_message.role == "tool" and "103" in tool_message.content
    await bus.wait_idle(1.0)
    assert [e.type for e in started] == [EventType.TOOL_CALL_STARTED, EventType.TOOL_CALL_FINISHED]


async def test_bucle_respeta_el_tope_de_iteraciones():
    plan = [[_use("demo__calculator", expression=f"{n}+1")] for n in range(10)]
    model = _ToolCaller(plan)
    outcome = await ToolLoop(_registry(), lambda: 3).run(model, _request(), "cid")
    assert outcome.stopped_by == "max_iterations"
    assert outcome.iterations == 3
    assert model.requests[-1].tools == ()  # la vuelta final va sin herramientas
    assert outcome.reply.text == "Listo."


async def test_bucle_detecta_llamadas_repetidas():
    same = _use("demo__calculator", expression="2+2")
    model = _ToolCaller([[same], [same], [same]])
    outcome = await ToolLoop(_registry(), lambda: 8).run(model, _request(), "cid")
    assert outcome.stopped_by == "loop"
    assert len(outcome.results) == 1


async def test_herramienta_lenta_se_reintenta_y_luego_usa_la_alterna():
    calls: list[str] = []

    async def slow(arguments: dict[str, Any]) -> Any:
        calls.append("lenta")
        await asyncio.sleep(1.0)
        return "nunca"

    async def backup(arguments: dict[str, Any]) -> Any:
        calls.append("alterna")
        return {"ok": True}

    registry = ToolRegistry(None, lambda: 0.05)
    registry.register(ToolSpec("x", "lenta", "d", {"type": "object"}, slow, fallback="x.alterna"))
    registry.register(ToolSpec("x", "alterna", "d", {"type": "object"}, backup))
    model = _ToolCaller([[_use("x__lenta")]])
    outcome = await ToolLoop(registry, lambda: 4).run(model, _request(), "cid")
    assert calls == ["lenta", "lenta", "alterna"]
    assert outcome.results[-1].success


async def test_herramienta_rota_le_pide_al_modelo_otra_estrategia():
    async def broken(arguments: dict[str, Any]) -> Any:
        raise RuntimeError("disco lleno")

    registry = ToolRegistry(None, lambda: 1.0)
    registry.register(ToolSpec("x", "rota", "d", {"type": "object"}, broken))
    model = _ToolCaller([[_use("x__rota")]], final="No se pudo: disco lleno.")
    outcome = await ToolLoop(registry, lambda: 4).run(model, _request(), "cid")
    message = model.requests[1].messages[-1]
    assert message.is_error and "No repitas" in message.content and "disco lleno" in message.content
    assert not outcome.results[0].success


async def test_herramienta_inexistente_falla_con_nombre():
    registry = ToolRegistry(None, lambda: 1.0)
    call, result = await registry.call("media.volume_change", {"delta": 10}, "cid")
    assert call is None
    assert not result.success and "media.volume_change" in (result.error or "")


def test_calculadora_segura():
    assert safe_arithmetic("2^10") == 1024
    assert safe_arithmetic("(25*4)+3") == 103
    for evil in ("__import__('os')", "open('x')", "a+1", "9**999999"):
        with pytest.raises(ValueError):
            safe_arithmetic(evil)
