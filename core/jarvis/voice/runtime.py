"""Lo que comparten la escucha y el habla del pipeline de voz.

El estado del nucleo, cuando el pipeline esta ocioso, las tareas en vuelo y la
forma de publicar al bus. `VoiceListener` y `VoiceSpeaker` reciben esta pieza
en vez de conocerse entre si mas de lo necesario.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from jarvis.config import VoiceConfig
from jarvis.contracts.enums import CoreState, EventType
from jarvis.contracts.events import Event
from jarvis.contracts.system import StateChange

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

    from jarvis.bus.base import EventBus
    from jarvis.contracts.base import BaseContract, CorrelationId
    from jarvis.voice.engines import VoiceEngines

__all__ = ["PipelineRuntime"]


class PipelineRuntime:
    """Estado y utilidades compartidas por las partes del pipeline."""

    def __init__(
        self,
        bus: EventBus,
        engines: VoiceEngines,
        config: VoiceConfig | Callable[[], VoiceConfig],
    ) -> None:
        """Args: el bus, los motores y la configuracion (o una funcion que la lea)."""
        self.bus = bus
        self.engines = engines
        self.config: Callable[[], VoiceConfig] = (
            (lambda: config) if isinstance(config, VoiceConfig) else config
        )
        cfg = self.config()
        self.sample_rate = cfg.sample_rate
        self.frame_ms = cfg.frame_ms

        self.state = CoreState.DORMIDO
        self.idle = asyncio.Event()
        self.idle.set()
        self.not_listening = asyncio.Event()
        """Despejado mientras el usuario habla: Jarvis no habla encima de nadie."""
        self.not_listening.set()
        self.stopping = False
        self.tasks: set[asyncio.Task[object]] = set()

        self.busy_checks: list[Callable[[], bool]] = []
        """Cada parte registra aqui como saber si esta ocupada."""
        self.active_correlation_id: Callable[[], CorrelationId | None] = lambda: None
        """La interaccion en curso, para los cambios de estado que no traen la suya."""

    def spawn(self, coroutine: Coroutine[Any, Any, object]) -> asyncio.Task[object]:
        """Lanza una tarea y la retiene hasta que termine.

        asyncio solo guarda referencias debiles a las tareas: sin este conjunto,
        una transcripcion en curso podria desaparecer a medias.
        """
        task = asyncio.create_task(coroutine)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return task

    def refresh_idle(self) -> None:
        """Marca ocioso si ninguna parte tiene trabajo pendiente."""
        if not any(busy() for busy in self.busy_checks):
            self.idle.set()

    async def set_state(
        self,
        state: CoreState,
        reason: str,
        correlation_id: CorrelationId | None = None,
    ) -> None:
        """Cambia el estado y lo publica, si de verdad cambio."""
        previous = self.state
        if previous is state:
            return
        self.state = state
        cid = correlation_id if correlation_id is not None else self.active_correlation_id()
        await self.emit(
            EventType.CORE_STATE_CHANGED,
            StateChange(
                previous=previous,
                current=state,
                reason=reason,
                source="voice.pipeline",
                correlation_id=cid,
            ),
            "voice.pipeline",
            cid,
        )

    async def emit(
        self,
        event_type: EventType,
        payload: BaseContract,
        source: str,
        correlation_id: CorrelationId | None = None,
    ) -> None:
        """Publica un contrato en el bus."""
        await self.bus.publish(
            Event.of(event_type, payload, source=source, correlation_id=correlation_id)
        )

    async def report_error(
        self,
        module: str,
        error: BaseException,
        correlation_id: CorrelationId | None = None,
    ) -> None:
        """Nada falla en silencio: todo fallo llega al bus como `core.error`."""
        await self.bus.publish(
            Event(
                type=EventType.CORE_ERROR,
                payload={"module": module, "error": type(error).__name__, "detail": str(error)},
                source=module,
                correlation_id=correlation_id,
            )
        )
