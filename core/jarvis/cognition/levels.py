"""Ejecucion de los niveles 2 y 3, con escaladas y degradaciones honestas.

Los caminos posibles de una orden que no es de forma fija:

    2                    el modelo local contesta con confianza suficiente
    2 -> 3               duda (temprana o al medir la respuesta entera): escala
    2 -x-> 3             Ollama no responde y hay nube: se avisa y se escala
    2 (con advertencia)  duda y no hay nube: se contesta avisando que no hay certeza
    3                    la orden lo pedia (codigo, pasos, contexto, explicito)
    3 -x-> 2             la nube falla o se acabo el presupuesto: se avisa y
                         contesta el local
    x                    ninguno puede: se dice exactamente por que

Ningun camino termina en silencio ni finge una respuesta que no existe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from jarvis.cognition.llm import Complexity, GateDecision, LLMReply, LLMRequest, LLMUnavailableError
from jarvis.cognition.router import Route
from jarvis.contracts.enums import CascadeLevel, Emotion
from jarvis.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from jarvis.cognition.context import BuiltContext
    from jarvis.cognition.llm import LanguageModel
    from jarvis.cognition.router import Router
    from jarvis.cognition.speaker import ResponseStream
    from jarvis.cognition.tool_loop import ToolLoop, ToolRegistry
    from jarvis.config import CognitionConfig
    from jarvis.contracts.cognition import Intent
    from jarvis.contracts.tools import ToolCall

__all__ = ["Attempt", "CloudModel", "Interaction", "LevelRunner"]

_log = get_logger(__name__)

_TOOL_INTENTS = frozenset({"query.calculation", "memory.recall"})
"""Intenciones que se resuelven mejor con herramientas. Crece en la Fase 6."""


class CloudModel(Protocol):
    """Un `LanguageModel` de nivel 3 que sabe decir por que no esta disponible."""

    @property
    def level(self) -> CascadeLevel:
        """Siempre el nivel 3."""
        ...

    def unavailable_reason(self) -> str | None:
        """None si se puede usar ahora mismo."""
        ...

    async def generate(self, request: LLMRequest) -> LLMReply:
        """Ver `LanguageModel.generate`."""
        ...

    async def available(self) -> bool:
        """Ver `LanguageModel.available`."""
        ...


@dataclass(slots=True)
class Attempt:
    """Como se va resolviendo una orden. Termina en la respuesta y su registro."""

    level: CascadeLevel
    levels: list[CascadeLevel] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    text: str = ""
    emotion: Emotion = Emotion.NEUTRAL
    should_speak: bool = True
    model: str | None = None
    confidence: float | None = None
    cost_usd: float = 0.0
    degraded: bool = False
    actions: list[ToolCall] = field(default_factory=list)
    fallback_text: str = ""
    """La respuesta local que se descarto al escalar; se usa si la nube falla."""


@dataclass(slots=True)
class Interaction:
    """Todo lo que los niveles necesitan de la orden en curso."""

    text: str
    intent: Intent
    correlation_id: str
    stream: ResponseStream
    attempt: Attempt
    local_context: Callable[[], BuiltContext]
    cloud_context: Callable[[], BuiltContext]
    cloud_blocker: Callable[[], str | None]


class LevelRunner:
    """Corre los niveles 2 y 3 sobre una `Interaction`."""

    def __init__(
        self,
        *,
        config: Callable[[], CognitionConfig],
        router: Router,
        local: LanguageModel | None,
        cloud: CloudModel | None,
        tools: ToolRegistry,
        tool_loop: ToolLoop,
        announce_route: Callable[[Route, Interaction], Awaitable[None]],
    ) -> None:
        """Args: los modelos (None si no existen), las herramientas y como publicar rutas."""
        self._config = config
        self._router = router
        self._local = local
        self.cloud = cloud
        self._tools = tools
        self._tool_loop = tool_loop
        self._announce_route = announce_route

    async def run_local(self, it: Interaction, route: Route) -> None:
        """Nivel 2, con escalada o degradacion segun haga falta."""
        it.attempt.levels.append(CascadeLevel.LOCAL_LLM)
        it.attempt.reasons.append(route.reason)
        it.attempt.degraded = it.attempt.degraded or route.degraded
        it.stream.restart(CascadeLevel.LOCAL_LLM)
        if route.degraded and route.wanted_cloud_because:
            await it.stream.notice(
                f"Esto pedía la nube, pero {it.cloud_blocker()}. Respondo en local."
            )

        if self._local is None:
            await self._local_failed(it, "no hay modelo local configurado")
            return
        blocker = it.cloud_blocker()
        cfg = self._config()
        use_tools = cfg.local_tools and it.intent.name in _TOOL_INTENTS and len(self._tools) > 0
        request = LLMRequest(
            messages=it.local_context().messages,
            on_text=None if use_tools else it.stream.feed,
            gate=None
            if use_tools
            else self._router.gate(cloud_blocker=blocker, margin=cfg.confidence_stream_margin),
            correlation_id=it.correlation_id,
        )
        try:
            reply = await self._generate(self._local, request, it, use_tools=use_tools)
        except LLMUnavailableError as error:
            await self._local_failed(it, error.reason, spoke=error.spoke)
            return

        it.attempt.model = reply.model
        it.attempt.confidence = reply.confidence
        if reply.streamed:
            # Ya se esta diciendo: la confianza final queda registrada, no se escala.
            it.attempt.level = CascadeLevel.LOCAL_LLM
            it.attempt.text = reply.text
            await it.stream.finish()
            return

        escalation: Route | None
        if reply.gate is GateDecision.ABORT and blocker is None:
            early = reply.early_confidence or 0.0
            escalation = Route(
                CascadeLevel.CLOUD_LLM, f"confianza_temprana_{early:.2f}", Complexity.LIGHT
            )
        else:
            escalation = self._router.escalate_after_local(reply.confidence, cloud_blocker=blocker)
        if escalation is not None:
            it.attempt.fallback_text = reply.text
            await self.run_cloud(it, escalation, attempt_number=2)
            return

        threshold = cfg.local_confidence_threshold
        text = reply.text
        if not text:
            text = "No tengo una respuesta fiable para eso."
            it.attempt.emotion = Emotion.ERROR
        elif reply.confidence is not None and reply.confidence < threshold:
            # Sin nube a donde escalar: se contesta, pero sin fingir certeza.
            text = f"No estoy del todo seguro. {text}"
            it.attempt.degraded = True
        it.attempt.level = CascadeLevel.LOCAL_LLM
        it.attempt.text = text
        await it.stream.feed(text)
        await it.stream.finish()

    async def run_cloud(self, it: Interaction, route: Route, *, attempt_number: int = 1) -> None:
        """Nivel 3; si falla, avisa y cae al nivel 2."""
        if attempt_number > 1:
            await self._announce_route(route, it)
        it.attempt.levels.append(CascadeLevel.CLOUD_LLM)
        it.attempt.reasons.append(route.reason)
        it.stream.restart(CascadeLevel.CLOUD_LLM)
        if self.cloud is None:
            await self._cloud_failed(it, "no hay cliente de la nube", spoke=False)
            return
        use_tools = it.intent.name in _TOOL_INTENTS and len(self._tools) > 0
        request = LLMRequest(
            messages=it.cloud_context().messages,
            complexity=route.complexity,
            on_text=it.stream.feed,
            correlation_id=it.correlation_id,
        )
        try:
            reply = await self._generate(self.cloud, request, it, use_tools=use_tools)
        except LLMUnavailableError as error:
            await self._cloud_failed(it, error.reason, spoke=error.spoke)
            return
        it.attempt.level = CascadeLevel.CLOUD_LLM
        it.attempt.model = reply.model
        it.attempt.confidence = None
        it.attempt.cost_usd += reply.cost_usd
        it.attempt.text = reply.text
        if not reply.streamed and reply.text:
            await it.stream.feed(reply.text)
        await it.stream.finish()

    # ------------------------------------------------------------------

    async def _generate(
        self,
        model: LanguageModel | CloudModel,
        request: LLMRequest,
        it: Interaction,
        *,
        use_tools: bool,
    ) -> LLMReply:
        if not use_tools:
            return await model.generate(request)
        request.tools = self._tools.definitions()
        outcome = await self._tool_loop.run(model, request, it.correlation_id)
        it.attempt.actions.extend(outcome.calls)
        if outcome.stopped_by:
            it.attempt.reasons.append(f"herramientas_{outcome.stopped_by}")
        return outcome.reply

    async def _local_failed(self, it: Interaction, reason: str, *, spoke: bool = False) -> None:
        _log.warning("cascade.local_failed", reason=reason, spoke=spoke)
        it.attempt.reasons.append(f"local_caido: {reason}")
        blocker = it.cloud_blocker()
        if spoke:
            await it.stream.salvage()
            it.attempt.text = it.stream.answer_text
            it.attempt.degraded = True
            await it.stream.notice(
                "Perdí la conexión con el modelo local.", Emotion.ERROR, after=True
            )
            return
        if blocker is None and CascadeLevel.CLOUD_LLM not in it.attempt.levels:
            await it.stream.notice("El modelo local no responde; paso a la nube.", Emotion.ALERTA)
            await self.run_cloud(
                it, Route(CascadeLevel.CLOUD_LLM, "local_caido", Complexity.LIGHT), attempt_number=2
            )
            return
        it.attempt.level = CascadeLevel.LOCAL_LLM
        it.attempt.degraded = True
        it.attempt.emotion = Emotion.ERROR
        why_no_cloud = blocker or "la nube ya falló"
        await it.stream.notice(
            f"El modelo local no responde ({reason}) y la nube no está disponible: {why_no_cloud}.",
            Emotion.ERROR,
        )

    async def _cloud_failed(self, it: Interaction, reason: str, *, spoke: bool) -> None:
        _log.warning("cascade.cloud_failed", reason=reason, spoke=spoke)
        it.attempt.reasons.append(f"nube_caida: {reason}")
        it.attempt.degraded = True
        if spoke:
            it.attempt.level = CascadeLevel.CLOUD_LLM
            await it.stream.salvage()
            it.attempt.text = it.stream.answer_text
            await it.stream.notice("Se cortó la conexión con la nube.", Emotion.ERROR, after=True)
            return
        if CascadeLevel.LOCAL_LLM in it.attempt.levels:
            # El local ya se intento: se usa lo que dio, sin fingir certeza.
            it.attempt.level = CascadeLevel.LOCAL_LLM
            it.stream.restart(CascadeLevel.LOCAL_LLM)
            if it.attempt.fallback_text:
                it.attempt.text = (
                    f"La nube falló; no estoy del todo seguro. {it.attempt.fallback_text}"
                )
                await it.stream.feed(it.attempt.text)
                await it.stream.finish()
                return
            it.attempt.emotion = Emotion.ERROR
            await it.stream.notice(f"No pude resolverlo: la nube falló ({reason}).", Emotion.ERROR)
            return
        await it.stream.notice(
            f"La nube no está disponible: {reason}. Contesto en local.", Emotion.ALERTA
        )
        await self.run_local(it, Route(CascadeLevel.LOCAL_LLM, f"nube_caida: {reason}"))
