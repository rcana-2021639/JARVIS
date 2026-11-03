"""El grabador: del bus a la memoria episodica y procedimental.

Escucha a la cascada y guarda cada interaccion como episodio:

    cognition.response     la respuesta y las herramientas que uso (se guarda
                           un momento, hasta que llega su resumen)
    tool.call.finished     si cada herramienta funciono
    cognition.completed    el resumen: con el se escribe el turno, una accion
                           por herramienta, y se anota la secuencia en la
                           memoria procedimental (que puede promoverla a rutina)
    core.error             cada error, como episodio propio
    core.state.changed     la foto del sistema en la memoria de trabajo
"""

from __future__ import annotations

import asyncio
import json
from collections import OrderedDict
from typing import TYPE_CHECKING, TypeVar

from jarvis.contracts.cascade import CascadeRecord
from jarvis.contracts.cognition import Response
from jarvis.contracts.enums import CascadeLevel, EventType
from jarvis.contracts.tools import ToolResult
from jarvis.logging import get_logger
from jarvis.memory.procedural import Step
from jarvis.memory.text import clip

if TYPE_CHECKING:
    from collections.abc import Callable

    from jarvis.bus.base import EventBus, Subscription
    from jarvis.contracts.events import Event
    from jarvis.memory.episodic import Episode, EpisodicMemory
    from jarvis.memory.procedural import ProceduralMemory, Routine
    from jarvis.memory.working import WorkingMemory

__all__ = ["REDACTED_INTENTS", "EpisodeRecorder"]

_log = get_logger(__name__)
_T = TypeVar("_T")
_KEEP = 256
"""Respuestas y resultados que se recuerdan mientras llega su resumen."""
_ARGS_CHARS = 120

REDACTED_INTENTS = frozenset({"memory.forget"})
"""Intenciones cuyo texto NO se guarda: "olvida lo de X" guardado tal cual haria
que X se siguiera encontrando justo despues de olvidarlo."""

_IMPORTANCE = {"memory.remember": 0.7}


class EpisodeRecorder:
    """Convierte lo que pasa en el bus en episodios."""

    def __init__(
        self,
        bus: EventBus,
        *,
        episodic: EpisodicMemory,
        procedural: ProceduralMemory,
        working: WorkingMemory,
        on_routine: Callable[[Routine], None],
        on_episode: Callable[[Episode], None] = lambda episode: None,
    ) -> None:
        """Args: on_routine, se llama cuando una secuencia se promueve a rutina."""
        self._bus = bus
        self._episodic = episodic
        self._procedural = procedural
        self._working = working
        self._on_routine = on_routine
        self._on_episode = on_episode
        self._responses: OrderedDict[str, Response] = OrderedDict()
        self._results: OrderedDict[str, bool] = OrderedDict()
        self._subscriptions: list[Subscription] = []

    def start(self) -> None:
        """Se suscribe a los eventos que alimentan la memoria."""
        if self._subscriptions:
            return
        handlers = {
            EventType.COGNITION_RESPONSE: self._on_response,
            EventType.TOOL_CALL_FINISHED: self._on_tool,
            EventType.COGNITION_COMPLETED: self._on_completed,
            EventType.CORE_ERROR: self._on_error,
            EventType.CORE_STATE_CHANGED: self._on_state,
        }
        for event_type, handler in handlers.items():
            self._subscriptions.append(
                self._bus.subscribe(event_type, handler, name=f"memory.{event_type.value}")
            )

    async def stop(self) -> None:
        """Se da de baja."""
        for subscription in self._subscriptions:
            await subscription.unsubscribe()
        self._subscriptions.clear()

    # ------------------------------------------------------------------

    async def _on_response(self, event: Event) -> None:
        response = event.payload_as(Response)
        self._remember(self._responses, response.correlation_id, response)

    async def _on_tool(self, event: Event) -> None:
        result = event.payload_as(ToolResult)
        self._remember(self._results, result.call_id, result.success)

    async def _on_state(self, event: Event) -> None:
        self._working.observe(event)

    async def _on_completed(self, event: Event) -> None:
        record = event.payload_as(CascadeRecord)
        self._working.observe(event)
        response = self._responses.pop(record.correlation_id, None)
        if "cancelada" in record.reasons and (response is None or not response.text):
            return
        answer = response.text if response is not None else ""
        failed = record.degraded or any(reason.startswith("error") for reason in record.reasons)
        user_text: str | None = record.text
        if record.intent in REDACTED_INTENTS:
            user_text, answer = None, ""
            content = "Se pidió olvidar un tema; el contenido no se guarda."
        else:
            content = (
                f"Usuario: {record.text}\nJarvis: {answer}" if answer else f"Usuario: {record.text}"
            )
        episode = await asyncio.to_thread(
            self._episodic.add,
            "turno",
            content,
            user_text=user_text,
            answer=answer or None,
            intent=record.intent,
            level=record.handled_by_level,
            success=not failed,
            correlation_id=record.correlation_id,
            importance=_IMPORTANCE.get(record.intent, 0.3),
        )
        self._on_episode(episode)
        actions = response.actions if response is not None else []
        if not actions:
            return
        steps: list[Step] = []
        all_ok = not failed
        for call in actions:
            ok = self._results.pop(call.call_id, False)
            all_ok = all_ok and ok
            steps.append(Step(f"{call.server}.{call.tool}", dict(call.arguments)))
            arguments = clip(json.dumps(call.arguments, ensure_ascii=False), _ARGS_CHARS)
            await asyncio.to_thread(
                self._episodic.add,
                "accion",
                f"{call.server}.{call.tool}({arguments}) -> {'bien' if ok else 'falló'}",
                intent=record.intent,
                level=record.handled_by_level,
                success=ok,
                correlation_id=record.correlation_id,
            )
        # DECISION: solo se aprenden secuencias que eligio un MODELO. Las del
        # nivel 1 ya son rutas rapidas; volverlas rutina no acelera nada.
        if record.handled_by_level is CascadeLevel.FAST_PATH or user_text is None:
            return
        routine = await asyncio.to_thread(
            self._procedural.observe, user_text, steps, success=all_ok
        )
        if routine is not None:
            _log.info("memory.routine_promoted", routine=routine.intent_name, steps=len(steps))
            self._on_routine(routine)

    async def _on_error(self, event: Event) -> None:
        payload = event.payload if isinstance(event.payload, dict) else {}
        module = str(payload.get("module", event.source))
        error = str(payload.get("error", "error"))
        detail = clip(str(payload.get("detail", "")), 160)
        await asyncio.to_thread(
            self._episodic.add,
            "error",
            f"Error en {module}: {error}. {detail}".strip(),
            success=False,
            correlation_id=event.correlation_id,
        )

    @staticmethod
    def _remember(store: OrderedDict[str, _T], key: str, value: _T) -> None:
        store[key] = value
        while len(store) > _KEEP:
            store.popitem(last=False)
