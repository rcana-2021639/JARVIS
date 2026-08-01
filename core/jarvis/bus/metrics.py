"""Metricas del bus de eventos.

Sin esto no hay forma de saber si el bus es el cuello de botella cuando el
pipeline de voz no llegue a los 700ms de la Fase 1. Se mide desde el primer dia
justamente para no tener que adivinar despues.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from jarvis.contracts.enums import EventType

__all__ = ["BusMetrics", "TypeMetrics"]

#: Cuantas muestras de latencia se conservan por tipo de evento.
#: Suficiente para percentiles estables sin que la memoria crezca sin limite.
_LATENCY_SAMPLES = 1000

#: Ventana en segundos sobre la que se calcula el rendimiento instantaneo.
_THROUGHPUT_WINDOW_S = 5.0


def _percentile(sorted_values: list[float], fraction: float) -> float:
    """Percentil por interpolacion lineal sobre una lista YA ordenada."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = fraction * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


@dataclass(slots=True)
class TypeMetrics:
    """Contadores y latencias de un tipo de evento concreto."""

    published: int = 0
    delivered: int = 0
    dropped: int = 0
    failed: int = 0
    _latencies_ms: deque[float] = field(
        default_factory=lambda: deque(maxlen=_LATENCY_SAMPLES),
        repr=False,
    )

    def record_latency(self, latency_ms: float) -> None:
        """Anota cuanto tardo un evento desde que se publico hasta que se atendio."""
        self._latencies_ms.append(latency_ms)

    @property
    def samples(self) -> int:
        """Cuantas muestras de latencia hay disponibles ahora mismo."""
        return len(self._latencies_ms)

    @property
    def mean_latency_ms(self) -> float:
        """Latencia media de entrega."""
        if not self._latencies_ms:
            return 0.0
        return sum(self._latencies_ms) / len(self._latencies_ms)

    @property
    def p50_latency_ms(self) -> float:
        """Mediana de la latencia de entrega."""
        return _percentile(sorted(self._latencies_ms), 0.50)

    @property
    def p95_latency_ms(self) -> float:
        """Percentil 95. Es el numero que importa: la media esconde los atascos."""
        return _percentile(sorted(self._latencies_ms), 0.95)

    @property
    def max_latency_ms(self) -> float:
        """Peor latencia observada dentro de la ventana de muestras."""
        return max(self._latencies_ms, default=0.0)


class BusMetrics:
    """Metricas agregadas del bus, con desglose por tipo de evento.

    OJO con la diferencia entre `published` y `delivered`: el primero cuenta
    EVENTOS que entraron al bus, el segundo cuenta INVOCACIONES de manejador. Un
    evento con tres suscriptores que encajan suma 1 a `published` y 3 a
    `delivered`. Que `delivered` sea mayor que `published` es lo normal, no un
    error de contabilidad.
    """

    def __init__(self) -> None:
        """Arranca con todos los contadores en cero."""
        self._by_type: dict[EventType, TypeMetrics] = {}
        self._publish_times: deque[float] = deque()
        self._started_at = time.monotonic()
        self.published = 0
        self.delivered = 0
        self.dropped = 0
        self.failed = 0
        self.retried = 0

    def for_type(self, event_type: EventType) -> TypeMetrics:
        """Devuelve (creandolas si hace falta) las metricas de un tipo."""
        metrics = self._by_type.get(event_type)
        if metrics is None:
            metrics = TypeMetrics()
            self._by_type[event_type] = metrics
        return metrics

    # -- registro de sucesos ------------------------------------------------

    def record_published(self, event_type: EventType) -> None:
        """Un evento entro al bus."""
        self.published += 1
        self.for_type(event_type).published += 1
        now = time.monotonic()
        self._publish_times.append(now)
        self._trim_window(now)

    def record_delivered(self, event_type: EventType, latency_ms: float) -> None:
        """Un manejador atendio un evento sin lanzar excepcion."""
        self.delivered += 1
        type_metrics = self.for_type(event_type)
        type_metrics.delivered += 1
        type_metrics.record_latency(latency_ms)

    def record_dropped(self, event_type: EventType) -> None:
        """Un evento se descarto porque la cola de un consumidor estaba llena."""
        self.dropped += 1
        self.for_type(event_type).dropped += 1

    def record_failed(self, event_type: EventType) -> None:
        """Un manejador agoto sus reintentos y se rindio."""
        self.failed += 1
        self.for_type(event_type).failed += 1

    def record_retried(self) -> None:
        """Un manejador fallo y se va a reintentar."""
        self.retried += 1

    # -- lectura ------------------------------------------------------------

    @property
    def uptime_s(self) -> float:
        """Segundos desde que se creo el objeto de metricas."""
        return time.monotonic() - self._started_at

    @property
    def throughput_eps(self) -> float:
        """Eventos publicados por segundo en los ultimos segundos.

        Es una ventana movil y no el promedio historico: lo que interesa saber
        es si el bus esta saturado AHORA, no cuanto movio desde que arranco.
        """
        self._trim_window(time.monotonic())
        if not self._publish_times:
            return 0.0
        span = time.monotonic() - self._publish_times[0]
        if span <= 0:
            return float(len(self._publish_times))
        return len(self._publish_times) / span

    @property
    def types(self) -> Iterable[EventType]:
        """Tipos de evento que han pasado por el bus."""
        return self._by_type.keys()

    def snapshot(self) -> dict[str, float | int]:
        """Foto plana de los contadores globales, lista para publicar al bus."""
        return {
            "published": self.published,
            "delivered": self.delivered,
            "dropped": self.dropped,
            "failed": self.failed,
            "retried": self.retried,
            "throughput_eps": round(self.throughput_eps, 2),
            "uptime_s": round(self.uptime_s, 2),
        }

    def reset(self) -> None:
        """Vuelve todo a cero. Solo para pruebas."""
        self._by_type.clear()
        self._publish_times.clear()
        self._started_at = time.monotonic()
        self.published = 0
        self.delivered = 0
        self.dropped = 0
        self.failed = 0
        self.retried = 0

    def _trim_window(self, now: float) -> None:
        """Descarta las marcas de publicacion mas viejas que la ventana."""
        cutoff = now - _THROUGHPUT_WINDOW_S
        while self._publish_times and self._publish_times[0] < cutoff:
            self._publish_times.popleft()
