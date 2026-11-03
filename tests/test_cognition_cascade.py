"""La cascada completa sobre el bus: los criterios de aceptacion de la Fase 2.

Todo con falsos deterministas (ver `cognition_fakes.py`): lo que se verifica
es la orquestacion -- que nivel atiende, por que, que se dice y que se
registra --, no la calidad de ningun modelo.
"""

from __future__ import annotations

import asyncio
import random
import time
from typing import Any

import pytest

from jarvis.cognition.llm import ToolUse
from jarvis.cognition.tool_loop import ToolSpec
from jarvis.contracts import (
    CascadeLevel,
    CascadeRecord,
    CoreState,
    CostUpdate,
    Emotion,
    Event,
    EventType,
    Language,
    PrivacyMode,
    Response,
    ResponseChunk,
    RouteDecision,
    StateChange,
    Utterance,
)
from tests.cognition_fakes import FakeCloud, FakeLocal, Script, build


class Capture:
    """Todos los eventos de la cascada, para inspeccionarlos."""

    def __init__(self, bus) -> None:
        self.events: list[Event] = []
        self.bus = bus
        bus.subscribe("*", self._collect, max_queue=10_000)

    async def _collect(self, event: Event) -> None:
        self.events.append(event)

    async def settle(self) -> None:
        await self.bus.wait_idle(2.0)

    def of(self, event_type: EventType) -> list[Event]:
        return [e for e in self.events if e.type is event_type]

    def record(self) -> CascadeRecord:
        return self.of(EventType.COGNITION_COMPLETED)[-1].payload_as(CascadeRecord)

    def chunks(self) -> list[str]:
        return [
            e.payload_as(ResponseChunk).text for e in self.of(EventType.COGNITION_RESPONSE_CHUNK)
        ]

    def routes(self) -> list[RouteDecision]:
        return [e.payload_as(RouteDecision) for e in self.of(EventType.COGNITION_ROUTED)]


@pytest.fixture
async def capture(bus) -> Capture:
    return Capture(bus)


REFACTOR = "Analiza este archivo de 300 líneas y refactorízalo"


# ---------------------------------------------------------------------------
# Criterios de aceptacion
# ---------------------------------------------------------------------------


async def test_sube_el_volumen_se_resuelve_en_nivel_1_en_menos_de_100_ms(bus, capture):
    rig = build(bus, local=FakeLocal(), cloud=FakeCloud())
    started = time.perf_counter()
    response = await rig.orchestrator.handle_text("Sube el volumen")
    elapsed_ms = (time.perf_counter() - started) * 1000
    await capture.settle()

    assert elapsed_ms < 100
    assert response.handled_by_level is CascadeLevel.FAST_PATH
    assert response.estimated_cost_usd == 0
    assert rig.local.requests == [] and rig.cloud.requests == []
    record = capture.record()
    assert record.levels_tried == [CascadeLevel.FAST_PATH]
    assert record.total_ms < 100
    # Sin herramienta de volumen (Fase 6) lo dice, en vez de fingir que lo hizo.
    assert "volumen" in response.text.lower()
    assert response.actions == []


async def test_con_la_herramienta_registrada_la_usa_y_confirma(bus, capture):
    rig = build(bus, rng=random.Random(1))
    seen: list[dict[str, Any]] = []

    async def volume(arguments: dict[str, Any]) -> Any:
        seen.append(arguments)
        return {"volume": 60}

    rig.tools.register(ToolSpec("media", "volume_change", "volumen", {"type": "object"}, volume))
    response = await rig.orchestrator.handle_text("súbele 20 al volumen")
    assert seen == [{"delta": 20}]
    assert response.actions[0].qualified_name == "media.volume_change"
    assert response.emotion is Emotion.AFIRMATIVA


async def test_que_opinas_va_al_nivel_2_y_habla_en_streaming(bus, capture):
    local = FakeLocal([Script("El jazz premia la escucha atenta. Tiene su encanto.", 0.9)])
    rig = build(bus, local=local, cloud=FakeCloud())
    response = await rig.orchestrator.handle_text("¿Qué opinas del jazz?")
    await capture.settle()

    assert response.handled_by_level is CascadeLevel.LOCAL_LLM
    assert response.streamed
    assert capture.chunks() == ["El jazz premia la escucha atenta.", "Tiene su encanto."]
    assert rig.cloud.requests == []
    record = capture.record()
    assert record.levels_tried == [CascadeLevel.LOCAL_LLM]
    assert record.first_chunk_ms is not None and record.first_chunk_ms < 1000
    assert record.answer_confidence == pytest.approx(0.9)
    assert capture.routes()[0].reason == "conversacion"


async def test_refactorizar_un_archivo_escala_al_nivel_3(bus, capture):
    cloud = FakeCloud([Script("Lo dejo en el chat.", cost_usd=0.004)])
    rig = build(bus, local=FakeLocal(), cloud=cloud)
    response = await rig.orchestrator.handle_text(REFACTOR)
    await capture.settle()

    assert response.handled_by_level is CascadeLevel.CLOUD_LLM
    assert response.estimated_cost_usd == pytest.approx(0.004)
    assert rig.local.requests == []
    route = capture.routes()[0]
    assert route.level is CascadeLevel.CLOUD_LLM
    assert route.model == "fake-cloud-standard"
    assert capture.record().cost_usd == pytest.approx(0.004)


async def test_local_only_nunca_llama_a_la_nube(bus, capture):
    # FakeCloud lanza si se le llama estando bloqueada.
    rig = build(
        bus,
        local=FakeLocal([Script("Version local.", 0.9)]),
        cloud=FakeCloud(),
        privacy=PrivacyMode.LOCAL_ONLY,
    )
    response = await rig.orchestrator.handle_text(REFACTOR)
    await capture.settle()

    assert rig.cloud.requests == []
    assert response.handled_by_level is CascadeLevel.LOCAL_LLM
    record = capture.record()
    assert record.degraded
    assert "solo-local" in capture.chunks()[0]
    assert capture.routes()[0].privacy_mode is PrivacyMode.LOCAL_ONLY


async def test_modo_local_por_voz_corta_la_nube(bus, capture):
    rig = build(bus, local=FakeLocal([Script("Local.", 0.9)]), cloud=FakeCloud())
    switched = await rig.orchestrator.handle_text("Jarvis, modo local")
    assert switched.handled_by_level is CascadeLevel.FAST_PATH
    assert rig.privacy.mode is PrivacyMode.LOCAL_ONLY
    await rig.orchestrator.handle_text(REFACTOR)
    assert rig.cloud.requests == []
    await rig.orchestrator.handle_text("modo normal")
    assert rig.privacy.mode is PrivacyMode.HYBRID


async def test_el_contador_refleja_el_gasto_y_se_publica(bus, capture):
    rig = build(bus, local=FakeLocal(), cloud=FakeCloud([Script("Hecho.", cost_usd=0.003)]))

    # La fabrica real conecta el cliente de la nube con `on_cost`; aqui se simula.
    async def charged(update: CostUpdate) -> None:
        await rig.orchestrator.on_cost(update)

    await rig.orchestrator.handle_text(REFACTOR)
    await charged(
        CostUpdate(
            model="m", input_tokens=10, output_tokens=5, cost_usd=0.003,
            spent_today_usd=rig.ledger.spent_today, budget_today_usd=0.5, budget_exhausted=False,
        )
    )  # fmt: skip
    await capture.settle()
    assert rig.ledger.spent_today == pytest.approx(0.003)
    assert capture.of(EventType.COGNITION_COST_UPDATED)
    spent = await rig.orchestrator.handle_text("¿Cuánto llevamos gastado hoy?")
    assert "0.00 de 0.50" in spent.text


async def test_al_llegar_al_limite_avisa_y_degrada(bus, capture):
    cloud = FakeCloud([Script("Primera respuesta de la nube.", cost_usd=0.02)])
    rig = build(bus, local=FakeLocal([Script("Respuesta local.", 0.9)]), cloud=cloud, budget=0.01)

    async def on_cost_from_ledger() -> None:
        await rig.orchestrator.on_cost(
            CostUpdate(
                model="m", input_tokens=1, output_tokens=1, cost_usd=0.02,
                spent_today_usd=rig.ledger.spent_today, budget_today_usd=0.01,
                budget_exhausted=not rig.ledger.can_spend(),
            )
        )  # fmt: skip

    original = cloud.generate

    async def generate_and_charge(request):
        reply = await original(request)
        await on_cost_from_ledger()
        return reply

    cloud.generate = generate_and_charge  # type: ignore[method-assign]

    first = await rig.orchestrator.handle_text(REFACTOR)
    assert first.handled_by_level is CascadeLevel.CLOUD_LLM
    assert "presupuesto" in first.text  # el aviso hablado
    assert len(cloud.requests) == 1

    second = await rig.orchestrator.handle_text(REFACTOR)
    assert second.handled_by_level is CascadeLevel.LOCAL_LLM
    assert len(cloud.requests) == 1, "no se gasta mas despues del tope"
    assert "presupuesto del dia agotado" in second.text


async def test_apagar_ollama_a_mitad_de_respuesta_no_crashea(bus, capture):
    local = FakeLocal(
        [Script("Uno dos tres. Cuatro cinco.", 0.9, fail="se corto", fail_after_words=3)]
    )
    rig = build(bus, local=local, cloud=FakeCloud())
    response = await rig.orchestrator.handle_text("¿Qué opinas del cine mudo?")
    await capture.settle()

    assert "Perdí la conexión con el modelo local." in response.text
    assert capture.record().degraded
    assert not capture.of(EventType.CORE_ERROR)
    # Y la siguiente orden funciona normal.
    again = await rig.orchestrator.handle_text("qué hora es")
    assert again.text == "Son las 3 y 25 de la tarde."


async def test_ollama_apagado_escala_a_la_nube_avisando(bus, capture):
    rig = build(
        bus,
        local=FakeLocal([Script(fail="Ollama no esta corriendo")]),
        cloud=FakeCloud([Script("Desde la nube.")]),
    )
    response = await rig.orchestrator.handle_text("¿Qué opinas del cine mudo?")
    await capture.settle()
    assert response.handled_by_level is CascadeLevel.CLOUD_LLM
    assert capture.chunks()[0] == "El modelo local no responde; paso a la nube."
    assert capture.record().levels_tried == [CascadeLevel.LOCAL_LLM, CascadeLevel.CLOUD_LLM]


async def test_nada_disponible_lo_dice_tal_cual(bus, capture):
    rig = build(
        bus,
        local=FakeLocal([Script(fail="Ollama no esta corriendo")]),
        cloud=FakeCloud(blocked="no hay clave"),
    )
    response = await rig.orchestrator.handle_text("¿Qué opinas del cine mudo?")
    assert response.emotion is Emotion.ERROR
    assert "no responde" in response.text and "no hay clave" in response.text


# ---------------------------------------------------------------------------
# Confianza del nivel 2
# ---------------------------------------------------------------------------


async def test_confianza_temprana_baja_escala_sin_decir_nada_local(bus, capture):
    local = FakeLocal([Script("Inventado.", confidence=0.2)])
    rig = build(bus, local=local, cloud=FakeCloud([Script("La respuesta correcta.")]))
    response = await rig.orchestrator.handle_text("¿Quién ganó el mundial de 1954?")
    await capture.settle()
    assert response.handled_by_level is CascadeLevel.CLOUD_LLM
    assert "Inventado" not in " ".join(capture.chunks())
    routes = capture.routes()
    assert [r.level for r in routes] == [CascadeLevel.LOCAL_LLM, CascadeLevel.CLOUD_LLM]
    assert routes[1].reason.startswith("confianza_temprana") and routes[1].attempt == 2


async def test_confianza_al_limite_se_mide_entera_antes_de_hablar(bus, capture):
    # Temprana 0.65 (entre umbral 0.6 y 0.75): se genera entera; final 0.4 -> escala.
    local = FakeLocal([Script("Dudoso.", confidence=0.4, early=0.65)])
    rig = build(bus, local=local, cloud=FakeCloud([Script("Seguro.")]))
    response = await rig.orchestrator.handle_text("¿Qué opinas del ajedrez?")
    assert response.handled_by_level is CascadeLevel.CLOUD_LLM
    assert response.text == "Seguro."

    ok = FakeLocal([Script("Bastante claro.", confidence=0.7, early=0.65)])
    rig2 = build(bus, local=ok, cloud=FakeCloud())
    response2 = await rig2.orchestrator.handle_text("¿Qué opinas del ajedrez?")
    assert response2.handled_by_level is CascadeLevel.LOCAL_LLM
    assert response2.text == "Bastante claro."


async def test_sin_nube_contesta_con_advertencia_en_vez_de_inventar_certeza(bus, capture):
    local = FakeLocal([Script("Creo que fue Alemania.", confidence=0.3)])
    rig = build(bus, local=local, cloud=FakeCloud(blocked="no hay clave"))
    response = await rig.orchestrator.handle_text("¿Quién ganó el mundial de 1954?")
    assert response.text == "No estoy del todo seguro. Creo que fue Alemania."
    assert capture.record().degraded if capture.events else True


async def test_si_la_nube_falla_tras_escalar_usa_la_respuesta_local(bus, capture):
    local = FakeLocal([Script("Creo que fue Alemania.", confidence=0.4, early=0.65)])
    rig = build(bus, local=local, cloud=FakeCloud([Script(fail="la API no responde")]))
    response = await rig.orchestrator.handle_text("¿Quién ganó el mundial de 1954?")
    assert response.handled_by_level is CascadeLevel.LOCAL_LLM
    assert "Creo que fue Alemania." in response.text
    assert "no estoy del todo seguro" in response.text.lower()


# ---------------------------------------------------------------------------
# Orquestacion
# ---------------------------------------------------------------------------


async def test_escucha_la_transcripcion_final_de_la_voz(bus, capture):
    rig = build(bus, local=FakeLocal(), cloud=FakeCloud())
    await rig.orchestrator.start(warm_up=False)
    utterance = Utterance(
        text="qué hora es", confidence=0.9, language=Language.ES, duration_ms=900,
        is_final=True, correlation_id="voz-1",
    )  # fmt: skip
    await bus.publish(
        Event.of(EventType.VOICE_STT_FINAL, utterance, source="test", correlation_id="voz-1")
    )
    for _ in range(50):
        await capture.settle()
        if capture.of(EventType.COGNITION_RESPONSE):
            break
        await asyncio.sleep(0.01)
    response = capture.of(EventType.COGNITION_RESPONSE)[0].payload_as(Response)
    assert response.correlation_id == "voz-1"
    assert response.text == "Son las 3 y 25 de la tarde."
    states = [e.payload_as(StateChange).current for e in capture.of(EventType.CORE_STATE_CHANGED)]
    assert states[0] is CoreState.PENSANDO
    await rig.orchestrator.stop()


async def test_jarvis_y_silencio_no_produce_respuesta(bus, capture):
    rig = build(bus, local=FakeLocal(), cloud=FakeCloud())
    await rig.orchestrator.start(warm_up=False)
    empty = Utterance(
        text="",
        confidence=0.0,
        language=Language.ES,
        duration_ms=0,
        is_final=True,
        correlation_id="v",
    )
    await bus.publish(Event.of(EventType.VOICE_STT_FINAL, empty, source="test"))
    await capture.settle()
    await asyncio.sleep(0.05)
    assert not capture.of(EventType.COGNITION_RESPONSE)
    await rig.orchestrator.stop()


async def test_una_orden_nueva_cancela_la_anterior(bus, capture):
    class Slow(FakeLocal):
        async def generate(self, request):
            await asyncio.sleep(5)
            return await super().generate(request)

    rig = build(bus, local=Slow(), cloud=FakeCloud(blocked="sin clave"))
    first = asyncio.create_task(rig.orchestrator.handle_text("¿Qué opinas del mar?"))
    await asyncio.sleep(0.05)
    second = await rig.orchestrator.handle_text("cancela")
    assert second.text in {"Cancelado.", "Olvidado.", "Descartado."}
    with pytest.raises(asyncio.CancelledError):
        await first
    await capture.settle()
    records = [e.payload_as(CascadeRecord) for e in capture.of(EventType.COGNITION_COMPLETED)]
    assert "cancelada" in records[0].reasons


async def test_repite_lo_ultimo_dicho(bus):
    rig = build(bus)
    first = await rig.orchestrator.handle_text("qué hora es")
    again = await rig.orchestrator.handle_text("repite")
    assert again.text == first.text


async def test_callate_no_contesta_nada(bus, capture):
    rig = build(bus)
    response = await rig.orchestrator.handle_text("cállate")
    assert not response.should_speak
    await capture.settle()
    states = [e.payload_as(StateChange).current for e in capture.of(EventType.CORE_STATE_CHANGED)]
    assert states[-1] is CoreState.DORMIDO


async def test_temporizador_avisa_al_vencer(bus, capture):
    rig = build(bus)
    response = await rig.orchestrator.handle_text("pon un temporizador de 1 segundo para el té")
    assert "1 segundo" in response.text and "te" in response.text  # "té" normalizado
    await asyncio.sleep(1.2)
    await capture.settle()
    texts = [e.payload_as(Response).text for e in capture.of(EventType.COGNITION_RESPONSE)]
    assert any("te" in t and t != response.text for t in texts)
    await rig.orchestrator.stop()


async def test_la_calculadora_pasa_por_el_bucle_de_herramientas(bus, capture):
    from jarvis.cognition.demo_tools import demo_tools

    call = ToolUse("t1", "demo__calculator", {"expression": "3.14159*3**2"})
    local = FakeLocal([Script("", tool_calls=[call]), Script("Unos 28 metros cuadrados.", 0.95)])
    rig = build(bus, local=local, cloud=FakeCloud(), local_tools=True)
    for tool in demo_tools():
        rig.tools.register(tool)
    response = await rig.orchestrator.handle_text("calcula el área de un círculo de radio 3")
    assert response.text == "Unos 28 metros cuadrados."
    assert response.actions[0].qualified_name == "demo.calculator"


async def test_las_respuestas_largas_dicen_dos_frases_y_el_resto_va_al_chat(bus, capture):
    long_text = "Uno. Dos. Tres. Cuatro."
    rig = build(bus, local=FakeLocal(), cloud=FakeCloud([Script(long_text)]))
    response = await rig.orchestrator.handle_text(REFACTOR)
    await capture.settle()
    assert capture.chunks() == ["Uno.", "Dos.", "El resto está en el chat."]
    assert response.text.startswith(long_text)


async def test_todo_queda_registrado_con_el_mismo_id(bus, capture):
    rig = build(bus, local=FakeLocal(), cloud=FakeCloud())
    response = await rig.orchestrator.handle_text("¿Qué opinas del té?", correlation_id="abc")
    await capture.settle()
    cascade = [e for e in capture.events if e.type.value.startswith("cognition.")]
    assert {e.correlation_id for e in cascade} == {"abc"}
    kinds = [e.type for e in cascade]
    assert kinds[0] is EventType.COGNITION_INTENT_CLASSIFIED
    assert kinds[1] is EventType.COGNITION_ROUTED
    assert kinds[-1] is EventType.COGNITION_COMPLETED
    assert response.correlation_id == "abc"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Cuánto es el quince por ciento de doscientos treinta", "Son 34,5."),
        ("cuánto es 25 por 4", "Son 100."),
        ("calcula 10 entre 4", "Son 2,5."),
        ("cuánto es 2 elevado a 10", "Son 1024."),
    ],
)
async def test_las_cuentas_simples_no_necesitan_modelo(bus, text, expected):
    rig = build(bus, local=FakeLocal(), cloud=FakeCloud())
    response = await rig.orchestrator.handle_text(text)
    assert response.handled_by_level is CascadeLevel.FAST_PATH
    assert response.text == expected
    assert rig.local.requests == []


async def test_si_nada_responde_el_aviso_no_se_repite(bus):
    rig = build(
        bus, local=FakeLocal([Script(fail="apagado")]), cloud=FakeCloud(blocked="sin clave")
    )
    response = await rig.orchestrator.handle_text("¿Qué opinas del mar?")
    assert response.text.count("no responde") == 1


async def test_lo_que_se_dijo_antes_del_corte_va_antes_del_aviso(bus):
    local = FakeLocal([Script("Uno dos tres. Cuatro.", 0.9, fail="corte", fail_after_words=3)])
    rig = build(bus, local=local, cloud=FakeCloud())
    response = await rig.orchestrator.handle_text("¿Qué opinas del mar?")
    assert response.text == "Uno dos tres. Perdí la conexión con el modelo local."
