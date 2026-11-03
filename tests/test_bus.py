"""Pruebas del bus de eventos.

Las tres garantias que el bus promete y que aqui se verifican:

    1. `publish` nunca bloquea ni lanza excepcion por culpa de un consumidor lento
    2. Una excepcion en un manejador no afecta a los demas suscriptores
    3. Cada suscriptor tiene su propia cola y se degrada solo
"""

from __future__ import annotations

import asyncio

import pytest

from jarvis.bus import (
    InMemoryEventBus,
    OverflowPolicy,
    RetryPolicy,
)
from jarvis.contracts import Event, EventType, Utterance


def _evento(tipo: EventType = EventType.VOICE_STT_PARTIAL, **payload: int) -> Event:
    """Construye un evento minimo para las pruebas."""
    return Event(type=tipo, payload=dict(payload), source="test")


# ---------------------------------------------------------------------------
# Entrega
# ---------------------------------------------------------------------------


async def test_entrega_a_un_suscriptor(bus: InMemoryEventBus) -> None:
    """El caso basico: lo que se publica llega."""
    recibidos: list[Event] = []

    async def manejador(evento: Event) -> None:
        recibidos.append(evento)

    bus.subscribe(EventType.VOICE_STT_PARTIAL, manejador)
    await bus.publish(_evento(i=1))
    assert await bus.wait_idle(timeout=2.0)
    assert len(recibidos) == 1


async def test_entrega_a_varios_suscriptores_independientes(bus: InMemoryEventBus) -> None:
    """Cada suscriptor recibe su copia sin enterarse de los demas."""
    a: list[Event] = []
    b: list[Event] = []

    async def primero(evento: Event) -> None:
        a.append(evento)

    async def segundo(evento: Event) -> None:
        b.append(evento)

    bus.subscribe("*", primero, name="a")
    bus.subscribe("*", segundo, name="b")
    await bus.publish(_evento())
    assert await bus.wait_idle(timeout=2.0)
    assert len(a) == len(b) == 1


async def test_filtra_por_tipo_exacto(bus: InMemoryEventBus) -> None:
    """Suscribirse a un tipo no trae los demas."""
    recibidos: list[Event] = []

    async def manejador(evento: Event) -> None:
        recibidos.append(evento)

    bus.subscribe(EventType.VOICE_STT_FINAL, manejador)
    await bus.publish(_evento(EventType.VOICE_STT_FINAL))
    await bus.publish(_evento(EventType.VOICE_STT_PARTIAL))
    await bus.publish(_evento(EventType.CORE_STARTED))
    assert await bus.wait_idle(timeout=2.0)
    assert len(recibidos) == 1
    assert recibidos[0].type is EventType.VOICE_STT_FINAL


async def test_filtra_por_patron_glob(bus: InMemoryEventBus) -> None:
    """`voice.*` recoge toda la familia del pipeline de voz."""
    recibidos: list[EventType] = []

    async def manejador(evento: Event) -> None:
        recibidos.append(evento.type)

    bus.subscribe("voice.*", manejador)
    await bus.publish(_evento(EventType.VOICE_STT_FINAL))
    await bus.publish(_evento(EventType.VOICE_WAKE_DETECTED))
    await bus.publish(_evento(EventType.COGNITION_RESPONSE))
    assert await bus.wait_idle(timeout=2.0)
    assert set(recibidos) == {EventType.VOICE_STT_FINAL, EventType.VOICE_WAKE_DETECTED}


async def test_entrega_mil_eventos_sin_perder_ninguno(bus: InMemoryEventBus) -> None:
    """El criterio de aceptacion de la fase, en version prueba."""
    total = 1000
    recibidos: list[Event] = []

    async def manejador(evento: Event) -> None:
        recibidos.append(evento)

    bus.subscribe("*", manejador, max_queue=total + 10)
    for i in range(total):
        await bus.publish(_evento(i=i))
    assert await bus.wait_idle(timeout=30.0)
    assert len(recibidos) == total
    assert [e.payload["i"] for e in recibidos] == list(range(total))


async def test_el_orden_se_conserva_por_suscriptor(bus: InMemoryEventBus) -> None:
    """Una cola FIFO por suscriptor significa orden garantizado."""
    recibidos: list[int] = []

    async def manejador(evento: Event) -> None:
        valor = evento.payload["i"]
        assert isinstance(valor, int)
        recibidos.append(valor)

    bus.subscribe("*", manejador, max_queue=100)
    for i in range(50):
        await bus.publish(_evento(i=i))
    assert await bus.wait_idle(timeout=5.0)
    assert recibidos == list(range(50))


# ---------------------------------------------------------------------------
# Contrapresion
# ---------------------------------------------------------------------------


async def test_consumidor_lento_no_bloquea_al_publicador(bus: InMemoryEventBus) -> None:
    """La garantia numero uno: publicar nunca se queda esperando."""
    bloqueado = asyncio.Event()

    async def lento(_evento: Event) -> None:
        await bloqueado.wait()

    sub = bus.subscribe("*", lento, name="lento", max_queue=3)
    for i in range(100):
        await asyncio.wait_for(bus.publish(_evento(i=i)), timeout=1.0)

    assert sub.pending <= 3
    assert sub.dropped > 0
    bloqueado.set()


async def test_consumidor_lento_no_afecta_a_los_demas(bus: InMemoryEventBus) -> None:
    """Un panel del HUD atascado no puede frenar el pipeline de voz."""
    bloqueado = asyncio.Event()
    rapidos: list[Event] = []

    async def lento(_evento: Event) -> None:
        await bloqueado.wait()

    async def rapido(evento: Event) -> None:
        rapidos.append(evento)

    bus.subscribe("*", lento, name="lento", max_queue=2)
    bus.subscribe("*", rapido, name="rapido", max_queue=100)

    for i in range(50):
        await bus.publish(_evento(i=i))

    for _ in range(100):
        if len(rapidos) == 50:
            break
        await asyncio.sleep(0.01)

    assert len(rapidos) == 50, "el consumidor rapido debio recibir todo igual"
    bloqueado.set()


async def test_drop_oldest_conserva_lo_mas_reciente(bus: InMemoryEventBus) -> None:
    """Para audio y telemetria, lo reciente vale mas que lo completo."""
    liberar = asyncio.Event()
    recibidos: list[int] = []

    async def lento(evento: Event) -> None:
        await liberar.wait()
        valor = evento.payload["i"]
        assert isinstance(valor, int)
        recibidos.append(valor)

    bus.subscribe("*", lento, name="lento", max_queue=3, overflow=OverflowPolicy.DROP_OLDEST)
    for i in range(20):
        await bus.publish(_evento(i=i))

    liberar.set()
    await bus.wait_idle(timeout=5.0)
    # El primero ya estaba en manos del manejador; los que quedan en cola son
    # los ultimos publicados, no los primeros.
    assert max(recibidos) == 19


async def test_drop_newest_conserva_lo_mas_viejo(bus: InMemoryEventBus) -> None:
    """Cuando el orden importa mas que la frescura."""
    liberar = asyncio.Event()
    recibidos: list[int] = []

    async def lento(evento: Event) -> None:
        await liberar.wait()
        valor = evento.payload["i"]
        assert isinstance(valor, int)
        recibidos.append(valor)

    sub = bus.subscribe("*", lento, name="lento", max_queue=3, overflow=OverflowPolicy.DROP_NEWEST)
    for i in range(20):
        await bus.publish(_evento(i=i))

    liberar.set()
    await bus.wait_idle(timeout=5.0)
    assert sub.dropped > 0
    assert max(recibidos) < 19


async def test_descartes_se_contabilizan(bus: InMemoryEventBus) -> None:
    """Perder eventos en silencio seria peor que perderlos."""
    bloqueado = asyncio.Event()

    async def lento(_evento: Event) -> None:
        await bloqueado.wait()

    bus.subscribe("*", lento, name="lento", max_queue=2)
    for i in range(30):
        await bus.publish(_evento(i=i))

    assert bus.metrics.dropped > 0
    bloqueado.set()


# ---------------------------------------------------------------------------
# Reintentos y aislamiento de fallos
# ---------------------------------------------------------------------------


async def test_manejador_roto_no_afecta_a_los_demas(bus: InMemoryEventBus) -> None:
    """La garantia numero dos: una excepcion se queda dentro de su suscripcion."""
    sanos: list[Event] = []

    async def roto(_evento: Event) -> None:
        msg = "fallo a proposito"
        raise RuntimeError(msg)

    async def sano(evento: Event) -> None:
        sanos.append(evento)

    bus.subscribe("*", roto, name="roto", retry=RetryPolicy(max_attempts=1))
    bus.subscribe("*", sano, name="sano")
    await bus.publish(_evento())
    assert await bus.wait_idle(timeout=2.0)

    assert len(sanos) == 1
    assert bus.metrics.failed == 1


async def test_publicar_no_lanza_aunque_el_manejador_falle(bus: InMemoryEventBus) -> None:
    """El publicador no puede enterarse de que un consumidor esta roto."""

    async def roto(_evento: Event) -> None:
        msg = "fallo"
        raise RuntimeError(msg)

    bus.subscribe("*", roto, retry=RetryPolicy(max_attempts=1))
    await bus.publish(_evento())  # no debe lanzar
    assert await bus.wait_idle(timeout=2.0)


async def test_reintenta_hasta_agotar_los_intentos(bus: InMemoryEventBus) -> None:
    """Tres intentos configurados son tres llamadas al manejador."""
    intentos = 0

    async def inestable(_evento: Event) -> None:
        nonlocal intentos
        intentos += 1
        msg = "sigue fallando"
        raise RuntimeError(msg)

    politica = RetryPolicy(max_attempts=3, initial_backoff_ms=1.0)
    bus.subscribe("*", inestable, retry=politica)
    await bus.publish(_evento())
    assert await bus.wait_idle(timeout=5.0)

    assert intentos == 3
    assert bus.metrics.retried == 2
    assert bus.metrics.failed == 1


async def test_un_reintento_exitoso_cuenta_como_entrega(bus: InMemoryEventBus) -> None:
    """Si el segundo intento funciona, el evento se entrego."""
    intentos = 0

    async def inestable(_evento: Event) -> None:
        nonlocal intentos
        intentos += 1
        if intentos == 1:
            msg = "fallo transitorio"
            raise RuntimeError(msg)

    bus.subscribe("*", inestable, retry=RetryPolicy(max_attempts=3, initial_backoff_ms=1.0))
    await bus.publish(_evento())
    assert await bus.wait_idle(timeout=5.0)

    assert intentos == 2
    assert bus.metrics.delivered == 1
    assert bus.metrics.failed == 0


def test_el_retroceso_crece_exponencialmente_con_techo() -> None:
    """Sin techo, ocho fallos seguidos serian varios minutos de espera."""
    politica = RetryPolicy(initial_backoff_ms=20.0, multiplier=2.0, max_backoff_ms=100.0)
    assert politica.backoff_ms(1) == 0.0
    assert politica.backoff_ms(2) == 20.0
    assert politica.backoff_ms(3) == 40.0
    assert politica.backoff_ms(4) == 80.0
    assert politica.backoff_ms(5) == 100.0
    assert politica.backoff_ms(9) == 100.0


def test_politica_de_reintento_rechaza_valores_invalidos() -> None:
    """Una politica mal formada es un error de programacion."""
    with pytest.raises(ValueError, match="al menos 1"):
        RetryPolicy(max_attempts=0)
    with pytest.raises(ValueError, match="negativos"):
        RetryPolicy(initial_backoff_ms=-1.0)
    with pytest.raises(ValueError, match="multiplier"):
        RetryPolicy(multiplier=0.5)


# ---------------------------------------------------------------------------
# Ciclo de vida y metricas
# ---------------------------------------------------------------------------


async def test_dar_de_baja_una_suscripcion_detiene_la_entrega(bus: InMemoryEventBus) -> None:
    """Tras `unsubscribe`, no llega nada mas."""
    recibidos: list[Event] = []

    async def manejador(evento: Event) -> None:
        recibidos.append(evento)

    sub = bus.subscribe("*", manejador)
    await bus.publish(_evento(i=1))
    assert await bus.wait_idle(timeout=2.0)
    await sub.unsubscribe()
    await bus.publish(_evento(i=2))
    await asyncio.sleep(0.05)

    assert len(recibidos) == 1
    assert bus.subscription_count == 0


async def test_el_bus_funciona_como_gestor_de_contexto() -> None:
    """`async with` arranca y cierra drenando lo pendiente."""
    recibidos: list[Event] = []

    async def manejador(evento: Event) -> None:
        recibidos.append(evento)

    instancia = InMemoryEventBus()
    async with instancia as activo:
        assert activo.is_running
        activo.subscribe("*", manejador)
        await activo.publish(_evento())
    assert len(recibidos) == 1
    assert not instancia.is_running


async def test_suscribirse_con_el_bus_ya_arrancado(bus: InMemoryEventBus) -> None:
    """Suscribirse en caliente debe funcionar, no solo antes de arrancar."""
    recibidos: list[Event] = []

    async def tardio(evento: Event) -> None:
        recibidos.append(evento)

    await bus.publish(_evento(i=1))
    bus.subscribe("*", tardio)
    await bus.publish(_evento(i=2))
    assert await bus.wait_idle(timeout=2.0)

    assert len(recibidos) == 1, "solo debe recibir lo publicado despues de suscribirse"


async def test_metricas_cuentan_publicados_y_entregados(bus: InMemoryEventBus) -> None:
    """`published` cuenta eventos, `delivered` cuenta invocaciones."""

    async def manejador(_evento: Event) -> None:
        return

    bus.subscribe("*", manejador, name="a")
    bus.subscribe("*", manejador, name="b")
    await bus.publish(_evento())
    assert await bus.wait_idle(timeout=2.0)

    assert bus.metrics.published == 1
    assert bus.metrics.delivered == 2


async def test_metricas_registran_latencia_por_tipo(bus: InMemoryEventBus) -> None:
    """La latencia por tipo es lo que permitira afinar el pipeline en la Fase 1."""

    async def manejador(_evento: Event) -> None:
        return

    bus.subscribe("*", manejador)
    for _ in range(10):
        await bus.publish(_evento(EventType.VOICE_LEVEL))
    assert await bus.wait_idle(timeout=2.0)

    stats = bus.metrics.for_type(EventType.VOICE_LEVEL)
    assert stats.delivered == 10
    assert stats.samples == 10
    assert stats.mean_latency_ms >= 0
    assert stats.p95_latency_ms >= stats.p50_latency_ms
    # La media se redondea y el maximo no: 0.97799... contra 0.978 fallaba a veces.
    assert stats.max_latency_ms >= stats.mean_latency_ms - 1e-6


async def test_wait_idle_devuelve_falso_si_vence_el_plazo(bus: InMemoryEventBus) -> None:
    """Vencer el plazo es una respuesta valida, no una excepcion."""
    bloqueado = asyncio.Event()

    async def lento(_evento: Event) -> None:
        await bloqueado.wait()

    bus.subscribe("*", lento, max_queue=10)
    await bus.publish(_evento())
    assert not await bus.wait_idle(timeout=0.2)
    bloqueado.set()


async def test_cola_de_tamano_invalido_se_rechaza(bus: InMemoryEventBus) -> None:
    """Una cola de cero no podria recibir nada."""

    async def manejador(_evento: Event) -> None:
        return

    with pytest.raises(ValueError, match="al menos 1"):
        bus.subscribe("*", manejador, max_queue=0)


async def test_evento_con_contrato_llega_completo(
    bus: InMemoryEventBus,
    utterance: Utterance,
) -> None:
    """El camino tipado de punta a punta: `Event.of` publicado y recuperado."""
    recuperadas: list[Utterance] = []

    async def manejador(evento: Event) -> None:
        recuperadas.append(evento.payload_as(Utterance))

    bus.subscribe(EventType.VOICE_STT_FINAL, manejador)
    await bus.publish(
        Event.of(
            EventType.VOICE_STT_FINAL,
            utterance,
            source="voice.stt",
            correlation_id=utterance.correlation_id,
        ),
    )
    assert await bus.wait_idle(timeout=2.0)

    assert recuperadas[0].text == utterance.text
    assert recuperadas[0].correlation_id == utterance.correlation_id
