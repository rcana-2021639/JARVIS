"""Demostracion del bus de eventos: 1000 eventos, tres consumidores, metricas.

Es el criterio de aceptacion de la Fase 0 que verifica que el bus entrega todo
lo que se publica y a que latencia lo hace.

Uso:

    python scripts/bus_demo.py
    python scripts/bus_demo.py --events 10000 --slow-consumer
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))

from jarvis.bus import InMemoryEventBus, OverflowPolicy
from jarvis.contracts import (
    CascadeLevel,
    Event,
    EventType,
    Language,
    Response,
    Utterance,
    new_correlation_id,
    utc_now,
)
from jarvis.logging import configure_logging

_WIDTH = 74


def _rule(char: str = "-") -> str:
    """Linea separadora del ancho de la tabla."""
    return char * _WIDTH


def _header(title: str) -> None:
    """Imprime un encabezado de seccion."""
    print(f"\n{_rule('=')}")
    print(f"  {title}")
    print(_rule("="))


class _Consumer:
    """Consumidor de prueba que cuenta lo que recibe y mide su latencia."""

    def __init__(self, name: str, delay_ms: float = 0.0) -> None:
        """Crea el consumidor. `delay_ms` simula trabajo real."""
        self.name = name
        self.delay_s = delay_ms / 1000.0
        self.received = 0
        self.latencies_ms: list[float] = []

    async def handle(self, event: Event) -> None:
        """Anota el evento y su latencia de entrega."""
        self.latencies_ms.append(event.age_ms)
        self.received += 1
        if self.delay_s:
            await asyncio.sleep(self.delay_s)

    @property
    def mean_latency_ms(self) -> float:
        """Latencia media de entrega vista por este consumidor."""
        if not self.latencies_ms:
            return 0.0
        return sum(self.latencies_ms) / len(self.latencies_ms)

    @property
    def p95_latency_ms(self) -> float:
        """Percentil 95 de la latencia de entrega."""
        if not self.latencies_ms:
            return 0.0
        ordered = sorted(self.latencies_ms)
        return ordered[min(int(0.95 * len(ordered)), len(ordered) - 1)]


def _build_events(total: int) -> list[Event]:
    """Construye eventos realistas alternando entre transcripciones y respuestas.

    Se construyen ANTES de medir a proposito: lo que se quiere medir es el
    reparto del bus, no cuanto tarda Pydantic en validar un modelo.
    """
    events: list[Event] = []
    for index in range(total):
        correlation_id = new_correlation_id()
        if index % 2 == 0:
            payload = Utterance(
                text=f"orden de prueba numero {index}",
                confidence=0.95,
                language=Language.ES,
                duration_ms=850.0,
                started_at=utc_now(),
                ended_at=utc_now(),
                is_final=True,
                correlation_id=correlation_id,
            )
            events.append(
                Event.of(
                    EventType.VOICE_STT_FINAL,
                    payload,
                    source="voice.stt",
                    correlation_id=correlation_id,
                ),
            )
        else:
            response = Response(
                text="Listo.",
                handled_by_level=CascadeLevel.FAST_PATH,
                correlation_id=correlation_id,
            )
            events.append(
                Event.of(
                    EventType.COGNITION_RESPONSE,
                    response,
                    source="cognition.router",
                    correlation_id=correlation_id,
                ),
            )
    return events


async def run_demo(total: int, *, slow_consumer: bool) -> int:
    """Corre la demostracion. Devuelve el codigo de salida del proceso."""
    _header(f"BUS DE EVENTOS -- {total} eventos, 3 consumidores")

    bus = InMemoryEventBus()

    # Tres consumidores con patrones distintos, que es justo lo que hay que
    # demostrar: reparto por tipo exacto y por glob, cada uno con su cola.
    timeline = _Consumer("timeline (*)")
    voice_panel = _Consumer("panel de voz (voice.*)")
    cognition_panel = _Consumer(
        "panel cognitivo (cognition.response)",
        delay_ms=0.5 if slow_consumer else 0.0,
    )

    events = _build_events(total)

    async with bus:
        bus.subscribe("*", timeline.handle, name="timeline", max_queue=total + 10)
        bus.subscribe("voice.*", voice_panel.handle, name="voice-panel", max_queue=total + 10)
        bus.subscribe(
            EventType.COGNITION_RESPONSE,
            cognition_panel.handle,
            name="cognition-panel",
            max_queue=64 if slow_consumer else total + 10,
            overflow=OverflowPolicy.DROP_OLDEST,
        )

        started = time.perf_counter()
        for event in events:
            await bus.publish(event)
        publish_s = time.perf_counter() - started

        drained = await bus.wait_idle(timeout=60.0)
        total_s = time.perf_counter() - started

    voice_total = sum(1 for e in events if e.type is EventType.VOICE_STT_FINAL)
    cognition_total = total - voice_total

    _header("ENTREGA")
    print(f"  {'CONSUMIDOR':<40} {'ESPERADO':>9} {'RECIBIDO':>9} {'':>6}")
    print(f"  {_rule()[:68]}")
    rows = (
        (timeline, total),
        (voice_panel, voice_total),
        (cognition_panel, cognition_total),
    )
    all_complete = True
    for consumer, expected in rows:
        ok = consumer.received == expected
        all_complete = all_complete and ok
        mark = "OK" if ok else "FALTAN"
        print(f"  {consumer.name:<40} {expected:>9} {consumer.received:>9} {mark:>6}")

    _header("LATENCIA DE ENTREGA (ms)")
    print("  Tiempo desde que el evento se publico hasta que su manejador lo tomo.")
    print("  Es sobre todo espera en cola: la demo publica los 1000 de golpe y")
    print("  los consumidores solo avanzan despues. No es sobrecosto del bus.")
    print()
    print(f"  {'CONSUMIDOR':<40} {'MEDIA':>9} {'P95':>9}")
    print(f"  {_rule()[:60]}")
    for consumer, _ in rows:
        print(
            f"  {consumer.name:<40} "
            f"{consumer.mean_latency_ms:>9.3f} {consumer.p95_latency_ms:>9.3f}",
        )

    _header("METRICAS DEL BUS")
    for key, value in bus.metrics.snapshot().items():
        print(f"  {key:<20} {value}")
    print("\n  Nota: 'published' cuenta eventos; 'delivered' cuenta invocaciones")
    print("  de manejador. Un evento con tres suscriptores suma 1 y 3.")

    print(f"\n  {'publicacion':<20} {publish_s * 1000:.1f} ms para {total} eventos")
    print(f"  {'total con drenado':<20} {total_s * 1000:.1f} ms")
    print(f"  {'rendimiento':<20} {total / total_s:,.0f} eventos/s")

    _header("LATENCIA POR TIPO DE EVENTO (ms)")
    print(f"  {'TIPO':<32} {'N':>7} {'MEDIA':>9} {'P50':>9} {'P95':>9}")
    print(f"  {_rule()[:70]}")
    for event_type in bus.metrics.types:
        stats = bus.metrics.for_type(event_type)
        print(
            f"  {event_type.value:<32} {stats.delivered:>7} "
            f"{stats.mean_latency_ms:>9.3f} {stats.p50_latency_ms:>9.3f} "
            f"{stats.p95_latency_ms:>9.3f}",
        )

    _header("RESULTADO")
    if not drained:
        print("  FALLO: quedaron eventos pendientes al vencer el plazo de drenado.")
        return 1
    if slow_consumer:
        print("  Modo consumidor lento: se ESPERAN descartes en el panel cognitivo.")
        print("  Lo que se demuestra es que el publicador nunca se bloqueo y que")
        print("  los otros dos consumidores recibieron todo igual.")
        if timeline.received != total or voice_panel.received != voice_total:
            print("  FALLO: un consumidor lento afecto a los demas.")
            return 1
        print("  OK: el consumidor lento degrado solo, sin arrastrar a nadie.")
        return 0
    if not all_complete:
        print("  FALLO: no todos los consumidores recibieron todos sus eventos.")
        return 1
    print("  OK: los tres consumidores recibieron todos sus eventos.")
    return 0


def main() -> int:
    """Punto de entrada."""
    parser = argparse.ArgumentParser(description="Demostracion del bus de eventos.")
    parser.add_argument("--events", type=int, default=1000, help="Cuantos eventos publicar.")
    parser.add_argument(
        "--slow-consumer",
        action="store_true",
        help="Agrega un consumidor lento con cola pequena para demostrar la contrapresion.",
    )
    args = parser.parse_args()

    # Registro en WARNING: la demo imprime su propia tabla y no queremos que el
    # ruido del bus la tape.
    configure_logging(level="WARNING")
    return asyncio.run(run_demo(args.events, slow_consumer=args.slow_consumer))


if __name__ == "__main__":
    raise SystemExit(main())
