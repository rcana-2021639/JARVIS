"""Interfaz abstracta del bus de eventos.

DECISION: los consumidores dependen SOLO de lo que hay en este archivo, nunca de
`InMemoryEventBus`. El documento maestro pide poder cambiar a Redis o NATS mas
adelante sin tocar a los consumidores, y eso solo se cumple si la
implementacion concreta no aparece en ninguna firma fuera del bus.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, TypeAlias, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from types import TracebackType
    from typing import Self

    from jarvis.bus.metrics import BusMetrics
    from jarvis.contracts.enums import EventType
    from jarvis.contracts.events import Event

__all__ = [
    "DEFAULT_RETRY_POLICY",
    "EventBus",
    "EventHandler",
    "OverflowPolicy",
    "RetryPolicy",
    "Subscription",
]

EventHandler: TypeAlias = "Callable[[Event], Awaitable[None]]"
"""Un manejador es una corrutina que recibe un evento y no devuelve nada.

Si necesita contestar algo, publica otro evento. El bus es unidireccional a
proposito: un manejador que devuelve valores invita a acoplar productores con
consumidores, que es justo lo que el bus existe para evitar.
"""


class OverflowPolicy(StrEnum):
    """Que hacer cuando la cola de un consumidor se llena.

    El principio de la fase es que un consumidor lento no tumba nada. Por eso
    NO existe una politica que bloquee al publicador indefinidamente: si el
    pipeline de voz se frena porque el panel de telemetria del HUD se atasco,
    el diseno esta mal.
    """

    DROP_OLDEST = "DROP_OLDEST"
    """Descarta el evento mas viejo sin atender y encola el nuevo.

    Es el valor por defecto y el correcto para flujos donde lo reciente vale mas
    que lo completo: nivel de audio a 30Hz, amplitud del TTS a 60Hz, telemetria.
    Que el HUD se pierda un fotograma de amplitud no le importa a nadie.
    """

    DROP_NEWEST = "DROP_NEWEST"
    """Descarta el evento entrante y conserva la cola como esta.

    Para consumidores donde el orden importa mas que la frescura y perder el
    principio de una secuencia la volveria incoherente.
    """


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Reintento con retroceso exponencial cuando un manejador lanza excepcion.

    El retroceso evita que un manejador roto queme la CPU reintentando en bucle
    cerrado. Al agotar los intentos el bus registra el fallo y sigue: un
    manejador defectuoso nunca detiene la entrega a los demas suscriptores.
    """

    max_attempts: int = 3
    """Intentos totales, incluido el primero. 1 significa sin reintentos."""

    initial_backoff_ms: float = 20.0
    """Espera antes del segundo intento."""

    multiplier: float = 2.0
    """Cuanto se multiplica la espera en cada intento sucesivo."""

    max_backoff_ms: float = 1000.0
    """Techo de la espera. Sin esto, ocho fallos seguidos son varios minutos."""

    def backoff_ms(self, attempt: int) -> float:
        """Espera en milisegundos antes del intento numero `attempt` (base 1)."""
        if attempt <= 1:
            return 0.0
        delay = self.initial_backoff_ms * (self.multiplier ** (attempt - 2))
        return min(delay, self.max_backoff_ms)

    def __post_init__(self) -> None:
        """Una politica mal formada es un error de programacion, no de datos."""
        if self.max_attempts < 1:
            msg = "max_attempts debe ser al menos 1"
            raise ValueError(msg)
        if self.initial_backoff_ms < 0 or self.max_backoff_ms < 0:
            msg = "Los tiempos de retroceso no pueden ser negativos"
            raise ValueError(msg)
        if self.multiplier < 1.0:
            msg = "multiplier debe ser al menos 1.0 para que el retroceso crezca"
            raise ValueError(msg)


DEFAULT_RETRY_POLICY = RetryPolicy()
"""Politica por defecto: tres intentos, 20ms, 40ms, techo de 1s."""


@runtime_checkable
class Subscription(Protocol):
    """Una suscripcion viva. Se obtiene de `EventBus.subscribe`."""

    @property
    def name(self) -> str:
        """Nombre legible del consumidor. Aparece en registros y metricas."""
        ...

    @property
    def pattern(self) -> str:
        """Patron que filtra los eventos: un tipo exacto o un glob como `voice.*`."""
        ...

    @property
    def pending(self) -> int:
        """Eventos encolados esperando a que el manejador los atienda."""
        ...

    @property
    def dropped(self) -> int:
        """Cuantos eventos se han descartado por desbordamiento de esta cola."""
        ...

    async def unsubscribe(self) -> None:
        """Da de baja la suscripcion y drena su cola."""
        ...


class EventBus(ABC):
    """Publicacion y suscripcion de eventos entre modulos.

    Contrato que toda implementacion debe cumplir:

    1. `publish` nunca bloquea de forma indefinida ni lanza excepcion por culpa
       de un consumidor lento. Como maximo descarta segun la `OverflowPolicy`.
    2. Una excepcion dentro de un manejador jamas se propaga al publicador ni
       afecta la entrega a los demas suscriptores.
    3. La entrega a cada suscriptor es independiente: cada uno tiene su cola.
    """

    @abstractmethod
    async def start(self) -> None:
        """Arranca las tareas de entrega. Idempotente."""

    @abstractmethod
    async def stop(self, *, drain: bool = True) -> None:
        """Detiene el bus. Con `drain` espera a vaciar las colas antes de cortar."""

    @abstractmethod
    async def publish(self, event: Event) -> None:
        """Publica un evento a todos los suscriptores que encajen con su tipo."""

    @abstractmethod
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
        """Registra un manejador para un tipo exacto o un patron glob.

        Args:
            pattern: `EventType.VOICE_STT_FINAL` para un tipo exacto, o una
                cadena glob como `voice.*` o `*.error` para varios.
            handler: corrutina que recibe el evento.
            name: nombre legible del consumidor. Por defecto, el del manejador.
            max_queue: tamano de la cola propia de este consumidor.
            overflow: que hacer cuando esa cola se llena.
            retry: politica de reintento si el manejador lanza excepcion.

        Returns:
            La suscripcion, para poder darla de baja despues.
        """

    @abstractmethod
    async def wait_idle(self, timeout: float | None = None) -> bool:  # noqa: ASYNC109
        """Espera a que todas las colas queden vacias.

        Existe para las pruebas y para el cierre ordenado del nucleo. Devuelve
        False si vencio el tiempo con eventos aun pendientes.

        El `timeout` es parte de la firma a proposito, en vez de dejarlo al
        `asyncio.timeout()` del llamador como sugiere ASYNC109: aqui vencer el
        plazo no es una excepcion sino una respuesta valida -- "quedaron cosas
        pendientes" --, y el cierre del nucleo necesita justamente eso para
        decidir si corta igual.
        """

    @property
    @abstractmethod
    def metrics(self) -> BusMetrics:
        """Metricas acumuladas del bus."""

    async def __aenter__(self) -> Self:
        """Permite usar el bus con `async with`."""
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Cierra el bus drenando lo pendiente."""
        await self.stop()
