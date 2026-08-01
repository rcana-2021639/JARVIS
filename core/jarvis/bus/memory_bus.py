"""Bus de eventos en memoria, asincrono, con contrapresion y reintentos."""

from __future__ import annotations

import asyncio
import contextlib
from fnmatch import fnmatchcase
from typing import TYPE_CHECKING

from jarvis.bus.base import (
    DEFAULT_RETRY_POLICY,
    EventBus,
    OverflowPolicy,
    RetryPolicy,
    Subscription,
)
from jarvis.bus.metrics import BusMetrics
from jarvis.contracts.enums import EventType
from jarvis.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from jarvis.contracts.events import Event

__all__ = ["InMemoryEventBus"]

_log = get_logger(__name__)

#: Cada cuantos descartes se vuelve a avisar que un consumidor va atrasado.
#: Sin esto, un consumidor atascado genera un evento de aviso por cada evento
#: descartado y el aviso se vuelve parte del problema.
_LAG_WARNING_EVERY = 100


class _Subscription:
    """Suscripcion concreta del bus en memoria.

    Cada una tiene su propia cola y su propia tarea de entrega. Es lo que hace
    que un consumidor lento no afecte a los demas: se llena SU cola, no la del
    bus.
    """

    __slots__ = (
        "_bus",
        "_closed",
        "_handler",
        "_queue",
        "_task",
        "dropped",
        "name",
        "overflow",
        "pattern",
        "retry",
    )

    def __init__(
        self,
        bus: InMemoryEventBus,
        pattern: str,
        handler: Callable[[Event], Awaitable[None]],
        name: str,
        max_queue: int,
        overflow: OverflowPolicy,
        retry: RetryPolicy,
    ) -> None:
        """Crea la suscripcion. La tarea de entrega arranca con el bus."""
        self._bus = bus
        self._handler = handler
        self._queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=max_queue)
        self._task: asyncio.Task[None] | None = None
        self._closed = False
        self.name = name
        self.pattern = pattern
        self.overflow = overflow
        self.retry = retry
        self.dropped = 0

    @property
    def pending(self) -> int:
        """Eventos encolados esperando atencion."""
        return self._queue.qsize()

    def matches(self, event_type: EventType) -> bool:
        """Si este suscriptor quiere recibir eventos de ese tipo."""
        return fnmatchcase(event_type.value, self.pattern)

    def offer(self, event: Event) -> bool:
        """Intenta encolar un evento sin bloquear jamas.

        Devuelve False si hubo que descartar algo. Nunca lanza excepcion: el
        publicador no puede verse afectado por un consumidor lento.
        """
        if self._closed:
            return False
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            return self._handle_overflow(event)
        return True

    def _handle_overflow(self, event: Event) -> bool:
        """Aplica la politica de desbordamiento configurada."""
        self.dropped += 1
        if self.overflow is OverflowPolicy.DROP_NEWEST:
            return False
        # DROP_OLDEST: saca el mas viejo sin atender y mete el nuevo.
        with contextlib.suppress(asyncio.QueueEmpty):
            self._queue.get_nowait()
            self._queue.task_done()
        with contextlib.suppress(asyncio.QueueFull):
            self._queue.put_nowait(event)
        return False

    def start(self) -> None:
        """Arranca la tarea que consume la cola. Idempotente."""
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name=f"bus-sub-{self.name}")

    async def _run(self) -> None:
        """Bucle de entrega. Solo termina cuando se cancela la tarea."""
        while True:
            event = await self._queue.get()
            try:
                await self._deliver(event)
            finally:
                self._queue.task_done()

    async def _deliver(self, event: Event) -> None:
        """Entrega un evento al manejador, con reintentos y retroceso.

        Una excepcion del manejador NUNCA sale de aqui. Se reintenta segun la
        politica y, si se agotan los intentos, se registra el fallo y el bucle
        sigue con el siguiente evento. Un manejador roto degrada su propia
        suscripcion, nunca el bus.
        """
        latency_ms = event.age_ms
        for attempt in range(1, self.retry.max_attempts + 1):
            try:
                await self._handler(event)
            except asyncio.CancelledError:
                raise
            # Capturar Exception aqui es el proposito del bus: aislar al
            # manejador. No es un except silencioso -- abajo se registra y se
            # contabiliza cada fallo.
            except Exception:
                if attempt < self.retry.max_attempts:
                    self._bus.metrics.record_retried()
                    await asyncio.sleep(self.retry.backoff_ms(attempt + 1) / 1000.0)
                    continue
                self._bus.metrics.record_failed(event.type)
                self._bus.on_handler_failed(self, event)
                return
            else:
                self._bus.metrics.record_delivered(event.type, latency_ms)
                return

    async def unsubscribe(self) -> None:
        """Da de baja la suscripcion y cancela su tarea de entrega."""
        if self._closed:
            return
        self._closed = True
        self._bus.remove_subscription(self)
        await self.cancel_task()

    async def cancel_task(self) -> None:
        """Cancela la tarea de entrega y espera a que termine de verdad."""
        task = self._task
        self._task = None
        if task is None or task.done():
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def join(self) -> None:
        """Espera a que la cola de esta suscripcion quede vacia."""
        await self._queue.join()


class InMemoryEventBus(EventBus):
    """Bus de eventos en memoria para un solo proceso.

    Es el bus de todo el nucleo. Redis o NATS solo harian falta si el nucleo se
    partiera en varios procesos, y hoy no lo esta: partirlo agregaria una
    serializacion y un salto de red por evento, justo lo contrario de lo que
    pide el principio de latencia. Cuando haga falta, `EventBus` ya define la
    frontera para cambiarlo sin tocar a un solo consumidor.
    """

    def __init__(self) -> None:
        """Crea un bus detenido. Hay que llamar a `start()`."""
        self._subscriptions: list[_Subscription] = []
        self._metrics = BusMetrics()
        self._running = False

    @property
    def metrics(self) -> BusMetrics:
        """Metricas acumuladas del bus."""
        return self._metrics

    @property
    def is_running(self) -> bool:
        """Si el bus esta entregando eventos ahora mismo."""
        return self._running

    @property
    def subscription_count(self) -> int:
        """Cuantas suscripciones vivas hay."""
        return len(self._subscriptions)

    @property
    def total_pending(self) -> int:
        """Suma de todas las colas. Es la senal de que algo se esta atrasando."""
        return sum(sub.pending for sub in self._subscriptions)

    async def start(self) -> None:
        """Arranca las tareas de entrega de todas las suscripciones."""
        self._running = True
        for sub in self._subscriptions:
            sub.start()

    async def stop(self, *, drain: bool = True) -> None:
        """Detiene el bus.

        Con `drain=True` espera a que las colas se vacien antes de cancelar, que
        es lo correcto al cerrar el nucleo: los eventos ya publicados llegan a
        su destino. Con `drain=False` corta de inmediato, para el killswitch.
        """
        if drain:
            await self.wait_idle(timeout=5.0)
        self._running = False
        for sub in list(self._subscriptions):
            await sub.cancel_task()

    def subscribe(
        self,
        pattern: str | EventType,
        handler: Callable[[Event], Awaitable[None]],
        *,
        name: str | None = None,
        max_queue: int = 256,
        overflow: OverflowPolicy = OverflowPolicy.DROP_OLDEST,
        retry: RetryPolicy = DEFAULT_RETRY_POLICY,
    ) -> Subscription:
        """Registra un manejador para un tipo exacto o un patron glob."""
        if max_queue < 1:
            msg = "max_queue debe ser al menos 1"
            raise ValueError(msg)
        raw_pattern = pattern.value if isinstance(pattern, EventType) else pattern
        sub = _Subscription(
            bus=self,
            pattern=raw_pattern,
            handler=handler,
            name=name or str(getattr(handler, "__name__", "anonimo")),
            max_queue=max_queue,
            overflow=overflow,
            retry=retry,
        )
        self._subscriptions.append(sub)
        if self._running:
            sub.start()
        return sub

    async def publish(self, event: Event) -> None:
        """Publica un evento a todos los suscriptores cuyo patron encaje.

        No bloquea y no lanza excepcion nunca. Si un consumidor no da abasto, su
        cola aplica la politica de desbordamiento y el resto no se entera.
        """
        self._metrics.record_published(event.type)
        for sub in self._subscriptions:
            if sub.matches(event.type) and not sub.offer(event):
                self._metrics.record_dropped(event.type)
                self._warn_if_lagging(sub)

    async def wait_idle(self, timeout: float | None = None) -> bool:  # noqa: ASYNC109
        """Espera a que todas las colas queden vacias. Ver `EventBus.wait_idle`."""
        if not self._subscriptions:
            return True
        joins = [sub.join() for sub in list(self._subscriptions)]
        try:
            await asyncio.wait_for(asyncio.gather(*joins), timeout=timeout)
        except TimeoutError:
            return False
        return True

    # -- ganchos internos usados por _Subscription --------------------------

    def remove_subscription(self, sub: _Subscription) -> None:
        """Saca una suscripcion de la lista de reparto."""
        with contextlib.suppress(ValueError):
            self._subscriptions.remove(sub)

    def on_handler_failed(self, sub: _Subscription, event: Event) -> None:
        """Gancho para cuando un manejador agota sus reintentos.

        DECISION: no publica un evento de error al bus. Si el manejador que
        falla estuviera suscrito a `*` -- y el timeline del HUD lo va a estar --
        cada error generaria un evento que provocaria otro error, en bucle. El
        fallo se registra y ahi muere.
        """
        _log.error(
            "bus.handler.failed",
            subscriber=sub.name,
            pattern=sub.pattern,
            event_type=event.type.value,
            event_id=event.event_id,
            attempts=sub.retry.max_attempts,
            correlation_id=event.correlation_id,
        )

    def _warn_if_lagging(self, sub: _Subscription) -> None:
        """Avisa cuando un consumidor lleva muchos descartes acumulados.

        Se avisa cada N descartes y no en cada uno: un consumidor atascado
        generaria una linea de registro por evento perdido y el aviso pasaria a
        ser parte del problema.
        """
        if sub.dropped % _LAG_WARNING_EVERY != 0:
            return
        _log.warning(
            "bus.consumer.lagging",
            subscriber=sub.name,
            pattern=sub.pattern,
            dropped=sub.dropped,
            pending=sub.pending,
            overflow=sub.overflow.value,
        )
