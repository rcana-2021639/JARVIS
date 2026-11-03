"""El orquestador cognitivo: de la transcripcion a la respuesta.

Escucha `voice.stt.final` (o recibe texto por `handle_text`) y por cada orden:

    1. publica PENSANDO
    2. clasifica                       -> cognition.intent.classified
    3. enruta                          -> cognition.routed
    4. resuelve en el nivel elegido, hablando frase a frase
                                       -> cognition.response.chunk
    5. responde                        -> cognition.response
    6. registra niveles, tiempos, costo -> cognition.completed

DECISION: una orden a la vez. Si llega otra mientras la anterior sigue
pensando, la anterior se cancela: el usuario ya paso a otra cosa, y dos
respuestas mezclandose en los altavoces es peor que perder una.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import TYPE_CHECKING, Any

from jarvis.cognition.context import ContextBuilder, ConversationHistory, Turn
from jarvis.cognition.control import SessionControl
from jarvis.cognition.fastpath import FastContext, FastPath, spoken_date
from jarvis.cognition.levels import Attempt, Interaction, LevelRunner
from jarvis.cognition.phrases import PhrasePicker
from jarvis.cognition.prompts import system_prompt
from jarvis.cognition.reporting import CascadePublisher, build_record, build_response
from jarvis.cognition.router import Route, Router, profile_task
from jarvis.cognition.speaker import ResponseStream
from jarvis.cognition.timers import TimerService
from jarvis.cognition.tool_loop import ToolLoop, ToolRegistry
from jarvis.contracts.base import new_correlation_id
from jarvis.contracts.cascade import CostUpdate, ResponseChunk
from jarvis.contracts.cognition import Intent, Response
from jarvis.contracts.enums import (
    CascadeLevel,
    CommandOrigin,
    CoreState,
    Emotion,
    EventType,
)
from jarvis.contracts.events import Event
from jarvis.contracts.voice import Utterance
from jarvis.logging import clear_correlation_id, get_logger, set_correlation_id

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine
    from datetime import datetime

    from jarvis.bus.base import EventBus, Subscription
    from jarvis.cognition.budget import SpendLedger
    from jarvis.cognition.classifier import IntentClassifier
    from jarvis.cognition.context import BuiltContext
    from jarvis.cognition.levels import CloudModel
    from jarvis.cognition.llm import LocalModel
    from jarvis.cognition.router import PrivacySwitch
    from jarvis.cognition.timers import Timer
    from jarvis.config import CognitionConfig

__all__ = ["CognitiveOrchestrator"]

_log = get_logger(__name__)
_BUDGET_NOTICE = "Aviso: se alcanzó el presupuesto de hoy. Sigo solo en local."


class CognitiveOrchestrator:
    """La cascada completa, conectada al bus."""

    def __init__(
        self,
        bus: EventBus,
        config: Callable[[], CognitionConfig],
        classifier: IntentClassifier,
        *,
        ledger: SpendLedger,
        privacy: PrivacySwitch,
        local: LocalModel | None,
        cloud: CloudModel | None,
        tools: ToolRegistry | None = None,
        user_name: Callable[[], str] = lambda: "Usuario",
        clock: Callable[[], datetime] | None = None,
        phrases: PhrasePicker | None = None,
    ) -> None:
        """Conecta las piezas. No se suscribe a nada hasta `start`."""
        self._bus = bus
        self._config = config
        self._classifier = classifier
        self._ledger = ledger
        self._privacy = privacy
        self._local = local
        self._cloud = cloud
        # `is None` y no `or`: un registro vacio es falso (tiene __len__) y se
        # cambiaria por otro, perdiendo lo que se registre despues.
        self._tools = (
            tools if tools is not None else ToolRegistry(bus, lambda: config().tool_timeout_s)
        )
        self._user_name = user_name
        self._clock = clock
        self._phrases = phrases or PhrasePicker()
        self._out = CascadePublisher(bus)
        self._control = SessionControl(
            user_name=user_name, privacy=privacy, ledger=ledger, local=local, cloud=cloud
        )
        self._router = Router(config)
        self._fast = FastPath()
        self._context = ContextBuilder()
        self._history = ConversationHistory(config().history_turns)
        self._timers = TimerService(self._announce_timer)
        self._runner = LevelRunner(
            config=config,
            router=self._router,
            local=local,
            cloud=cloud,
            tools=self._tools,
            tool_loop=ToolLoop(self._tools, lambda: config().max_tool_iterations),
            announce_route=self._publish_route,
        )
        self._inflight: asyncio.Task[Response] | None = None
        self._tasks: set[asyncio.Task[object]] = set()
        self._subscription: Subscription | None = None
        self._budget_notice_pending = False

    @property
    def fast_path(self) -> FastPath:
        """El nivel 1, para registrar manejadores extra."""
        return self._fast

    @property
    def tools(self) -> ToolRegistry:
        """Las herramientas disponibles."""
        return self._tools

    async def start(self, *, listen_voice: bool = True, warm_up: bool = True) -> None:
        """Se suscribe a la voz y calienta el modelo local en segundo plano."""
        if listen_voice and self._subscription is None:
            self._subscription = self._bus.subscribe(
                EventType.VOICE_STT_FINAL, self._on_final, name="cognition.orchestrator"
            )
        if warm_up and self._local is not None:
            prompt = system_prompt(self._user_name(), structured=True)
            self._spawn(self._local.warm_up(prompt))
        _log.info("cognition.started", privacy=self._privacy.mode.value)

    async def stop(self) -> None:
        """Cancela todo lo pendiente."""
        if self._subscription is not None:
            await self._subscription.unsubscribe()
            self._subscription = None
        await self._cancel_inflight()
        await self._timers.close()
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def handle_text(
        self,
        text: str,
        *,
        origin: CommandOrigin = CommandOrigin.TEXTO,
        correlation_id: str | None = None,
    ) -> Response:
        """Atiende una orden de punta a punta y devuelve la respuesta."""
        cid = correlation_id or new_correlation_id()
        cancelled = await self._cancel_inflight()
        task = asyncio.create_task(self._interact(text, origin, cid, cancelled))
        self._inflight = task
        try:
            return await task
        finally:
            if self._inflight is task:
                self._inflight = None

    @property
    def control(self) -> SessionControl:
        """Lo que la ruta rapida consulta y cambia: privacidad, gasto, salud."""
        return self._control

    async def health(self) -> str:
        """Estado de los niveles 2 y 3 en una frase."""
        return await self._control.health()

    def set_cloud(self, cloud: CloudModel | None) -> None:
        """Cambia el nivel 3 en caliente (el banco lo usa para simular la nube)."""
        self._cloud = cloud
        self._runner.cloud = cloud
        self._control.cloud = cloud

    # ------------------------------------------------------------------

    async def _on_final(self, event: Event) -> None:
        utterance = event.payload_as(Utterance)
        if utterance.is_empty:
            # "Jarvis" y silencio: no es un error, es alguien que cambio de idea.
            _log.info("cognition.empty_utterance")
            return
        self._spawn(
            self.handle_text(
                utterance.text, origin=CommandOrigin.VOZ, correlation_id=utterance.correlation_id
            )
        )

    async def _interact(
        self, text: str, origin: CommandOrigin, cid: str, cancelled_previous: bool
    ) -> Response:
        token = set_correlation_id(cid)
        started = time.perf_counter()
        cfg = self._config()
        self._history.resize(cfg.history_turns)
        stream = ResponseStream(
            self._publish_chunk, cid, max_sentences=cfg.spoken_max_sentences, started=started
        )
        attempt = Attempt(level=CascadeLevel.FAST_PATH)
        intent: Intent | None = None
        classify_ms = 0.0
        try:
            await self._out.state(CoreState.DORMIDO, CoreState.PENSANDO, "orden", cid)
            classification = self._classifier.classify(text)
            intent, classify_ms = classification.intent, classification.elapsed_ms
            await self._out.emit(EventType.COGNITION_INTENT_CLASSIFIED, intent, cid)
            local_ctx = self._build(text, structured=True, budget=cfg.local_context_tokens)
            blocker = self._control.cloud_blocker()
            route = self._router.decide(
                intent,
                profile_task(text, local_ctx.tokens),
                privacy=self._privacy.mode,
                cloud_blocker=blocker,
                fast_path_handles=self._fast.handles(intent.name),
            )
            interaction = Interaction(
                text=text,
                intent=intent,
                correlation_id=cid,
                stream=stream,
                attempt=attempt,
                local_context=lambda: local_ctx,
                cloud_context=lambda: self._build(
                    text, structured=False, budget=cfg.cloud_context_tokens
                ),
                cloud_blocker=self._control.cloud_blocker,
            )
            await self._publish_route(route, interaction)
            if route.level is CascadeLevel.FAST_PATH:
                await self._run_fast(interaction, route, cancelled_previous)
            elif route.level is CascadeLevel.LOCAL_LLM:
                await self._runner.run_local(interaction, route)
            else:
                await self._runner.run_cloud(interaction, route)
            if self._budget_notice_pending:
                self._budget_notice_pending = False
                await stream.notice(_BUDGET_NOTICE, Emotion.ALERTA)
        except asyncio.CancelledError:
            attempt.reasons.append("cancelada")
            attempt.should_speak = False
            await self._finish(text, intent, attempt, stream, cid, started, classify_ms)
            clear_correlation_id(token)
            raise
        except Exception as error:
            _log.exception("cognition.failed")
            await self._out.error(error, cid)
            attempt.text = "Algo falló al procesar eso. Quedó en el registro."
            attempt.emotion = Emotion.ERROR
            attempt.reasons.append(f"error: {type(error).__name__}")
        response = await self._finish(text, intent, attempt, stream, cid, started, classify_ms)
        clear_correlation_id(token)
        return response

    async def _run_fast(self, it: Interaction, route: Route, cancelled_previous: bool) -> None:
        it.attempt.levels.append(CascadeLevel.FAST_PATH)
        it.attempt.reasons.append(route.reason)
        now = self._clock() if self._clock else _local_now()
        outcome = await self._fast.execute(
            it.intent,
            FastContext(
                correlation_id=it.correlation_id,
                control=self._control,
                tools=self._tools,
                timers=self._timers,
                phrases=self._phrases,
                now=now,
                step_percent=self._config().fast_step_percent,
                cancelled_previous=cancelled_previous,
            ),
        )
        it.attempt.level = CascadeLevel.FAST_PATH
        it.attempt.text = outcome.text
        it.attempt.emotion = outcome.emotion
        it.attempt.should_speak = outcome.should_speak
        it.attempt.actions.extend(outcome.actions)

    async def _finish(
        self,
        text: str,
        intent: Intent | None,
        attempt: Attempt,
        stream: ResponseStream,
        cid: str,
        started: float,
        classify_ms: float,
    ) -> Response:
        response = build_response(attempt, stream, cid)
        await self._out.emit(EventType.COGNITION_RESPONSE, response, cid)
        if response.should_speak:
            self._control.spoken = stream.spoken_text or response.text
        # DECISION: el historial es la conversacion con los modelos. Las ordenes
        # del nivel 1 son acciones: medido, llenaban el contexto de "ese control
        # todavia no existe" y el modelo local copiaba ese estilo.
        conversational = attempt.level is not CascadeLevel.FAST_PATH
        if conversational and response.text and "cancelada" not in attempt.reasons:
            self._history.add(Turn(text, response.text, attempt.level, response.created_at))
        if not response.should_speak:
            await self._out.state(
                CoreState.PENSANDO, CoreState.DORMIDO, "sin_respuesta_hablada", cid
            )
        record = build_record(
            text, intent, attempt, stream, cid, started=started, classify_ms=classify_ms
        )
        await self._out.emit(EventType.COGNITION_COMPLETED, record, cid)
        _log.info(
            "cognition.completed",
            intent=record.intent,
            level=int(record.handled_by_level),
            tried=[int(level) for level in record.levels_tried],
            total_ms=record.total_ms,
            cost_usd=round(record.cost_usd, 6),
        )
        return response

    # ------------------------------------------------------------------

    def _build(self, text: str, *, structured: bool, budget: int) -> BuiltContext:
        now = self._clock() if self._clock else _local_now()
        # DECISION: el nivel 2 no recibe estado. Medido: qwen3:4b lo repetia en
        # voz alta ("Modo HYBRID activado", "Hoy es lunes... 20:26"). Fecha y
        # hora las contesta el nivel 1; la nube si las recibe.
        state = "" if structured else f"{spoken_date(now)[:-1]}, {now:%H:%M}"
        return self._context.build(
            system=system_prompt(self._user_name(), structured=structured),
            user_text=text,
            budget_tokens=budget,
            history=self._history,
            state=state,
        )

    async def on_cost(self, update: CostUpdate) -> None:
        """Recibe cada gasto del nivel 3: lo publica y prepara el aviso si se agoto."""
        await self._out.emit(EventType.COGNITION_COST_UPDATED, update, update.correlation_id)
        before = update.spent_today_usd - update.cost_usd
        if update.budget_exhausted and before < update.budget_today_usd:
            # Esta llamada cruzo el tope: se avisa una vez, al final de la respuesta.
            self._budget_notice_pending = True

    async def _announce_timer(self, timer: Timer) -> None:
        key, values = (
            ("timer_done_label", {"label": timer.label}) if timer.label else ("timer_done", {})
        )
        cid = new_correlation_id()
        response = Response(
            text=self._phrases.pick(key, **values),
            emotion=Emotion.ALERTA,
            handled_by_level=CascadeLevel.FAST_PATH,
            correlation_id=cid,
        )
        await self._out.emit(EventType.COGNITION_RESPONSE, response, cid)

    async def _publish_route(self, route: Route, it: Interaction) -> None:
        model = None
        if route.level is CascadeLevel.LOCAL_LLM and self._local is not None:
            model = self._local.model
        elif route.level is CascadeLevel.CLOUD_LLM and self._cloud is not None:
            model_for = getattr(self._cloud, "model_for", None)
            model = model_for(route.complexity) if callable(model_for) else None
        await self._out.route(route, it, model=model, privacy=self._privacy.mode)

    async def _publish_chunk(self, chunk: ResponseChunk) -> None:
        await self._out.emit(EventType.COGNITION_RESPONSE_CHUNK, chunk, chunk.correlation_id)

    async def _cancel_inflight(self) -> bool:
        task = self._inflight
        if task is None or task.done():
            return False
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        _log.info("cognition.superseded")
        return True

    def _spawn(self, coroutine: Coroutine[Any, Any, object]) -> None:
        task: asyncio.Task[object] = asyncio.create_task(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)


def _local_now() -> datetime:
    from datetime import datetime

    return datetime.now().astimezone()
