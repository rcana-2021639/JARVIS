"""Todo lo que la cascada publica: eventos, estado, errores y su contabilidad.

Separado del orquestador para que este se lea como el flujo de una orden y no
como una lista de `bus.publish`.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from jarvis.contracts.cascade import CascadeRecord, RouteDecision
from jarvis.contracts.cognition import Response
from jarvis.contracts.enums import CascadeLevel, CoreState, Emotion, EventType
from jarvis.contracts.events import Event
from jarvis.contracts.system import StateChange

if TYPE_CHECKING:
    from jarvis.bus.base import EventBus
    from jarvis.cognition.levels import Attempt, Interaction
    from jarvis.cognition.phrases import PhrasePicker
    from jarvis.cognition.router import Route
    from jarvis.cognition.speaker import ResponseStream
    from jarvis.cognition.timers import Timer
    from jarvis.contracts.base import BaseContract
    from jarvis.contracts.cognition import Intent
    from jarvis.contracts.enums import PrivacyMode

__all__ = ["CascadePublisher", "build_record", "build_response", "timer_response"]

SOURCE = "cognition.orchestrator"


class CascadePublisher:
    """Publica en el bus con la fuente y el id de correlacion correctos."""

    def __init__(self, bus: EventBus) -> None:
        """Args: bus, donde se publica todo."""
        self._bus = bus

    async def emit(self, event_type: EventType, payload: BaseContract, cid: str | None) -> None:
        """Un contrato como evento."""
        await self._bus.publish(Event.of(event_type, payload, source=SOURCE, correlation_id=cid))

    async def state(self, previous: CoreState, current: CoreState, reason: str, cid: str) -> None:
        """Una transicion del nucleo."""
        change = StateChange(
            previous=previous, current=current, reason=reason, source=SOURCE, correlation_id=cid
        )
        await self.emit(EventType.CORE_STATE_CHANGED, change, cid)

    async def route(
        self, route: Route, it: Interaction, *, model: str | None, privacy: PrivacyMode
    ) -> None:
        """Una decision del enrutador; `attempt` cuenta los niveles ya probados."""
        decision = RouteDecision(
            level=route.level,
            reason=route.reason,
            intent=it.intent.name,
            model=model,
            attempt=len(it.attempt.levels) + 1,
            privacy_mode=privacy,
            correlation_id=it.correlation_id,
        )
        await self.emit(EventType.COGNITION_ROUTED, decision, it.correlation_id)

    async def error(self, error: BaseException, cid: str) -> None:
        """Nada falla en silencio: el fallo llega al bus como `core.error`."""
        await self._bus.publish(
            Event(
                type=EventType.CORE_ERROR,
                payload={"module": SOURCE, "error": type(error).__name__, "detail": str(error)},
                source=SOURCE,
                correlation_id=cid,
            )
        )


def build_response(attempt: Attempt, stream: ResponseStream, cid: str) -> Response:
    """La respuesta final: avisos, texto y lo que quedo para el chat, en orden."""
    text = " ".join(part for part in (*stream.notices, attempt.text, *stream.trailing) if part)
    return Response(
        text=text,
        should_speak=attempt.should_speak and bool(text),
        emotion=attempt.emotion,
        handled_by_level=attempt.level,
        # Solo se cobra si contesto la nube: un intento fallido que gasto y
        # termino en local se registra en el CascadeRecord, no en la respuesta.
        estimated_cost_usd=attempt.cost_usd if attempt.level is CascadeLevel.CLOUD_LLM else 0.0,
        actions=attempt.actions,
        streamed=stream.any_spoken,
        correlation_id=cid,
    )


def build_record(
    text: str,
    intent: Intent | None,
    attempt: Attempt,
    stream: ResponseStream,
    cid: str,
    *,
    started: float,
    classify_ms: float,
) -> CascadeRecord:
    """El resumen de la interaccion para `cognition.completed`."""
    return CascadeRecord(
        text=text,
        intent=intent.name if intent else "desconocida",
        intent_confidence=intent.confidence if intent else 0.0,
        suggested_level=intent.suggested_level if intent else CascadeLevel.LOCAL_LLM,
        handled_by_level=attempt.level,
        levels_tried=attempt.levels or [attempt.level],
        reasons=attempt.reasons,
        model=attempt.model,
        answer_confidence=attempt.confidence,
        classify_ms=round(classify_ms, 3),
        first_chunk_ms=None if stream.first_chunk_ms is None else round(stream.first_chunk_ms, 1),
        total_ms=round((time.perf_counter() - started) * 1000.0, 1),
        cost_usd=attempt.cost_usd,
        degraded=attempt.degraded,
        correlation_id=cid,
    )


def timer_response(timer: Timer, phrases: PhrasePicker, cid: str) -> Response:
    """El aviso de un temporizador que vencio. Es una interaccion nueva, sin orden."""
    if timer.label:
        text = phrases.pick("timer_done_label", label=timer.label)
    else:
        text = phrases.pick("timer_done")
    return Response(
        text=text,
        emotion=Emotion.ALERTA,
        handled_by_level=CascadeLevel.FAST_PATH,
        correlation_id=cid,
    )
