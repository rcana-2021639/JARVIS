"""Los criterios de aceptacion de la Fase 3, de punta a punta sobre el bus.

La cascada corre con modelos falsos (`cognition_fakes`) y la memoria es la real:
SQLite, FTS5 y LanceDB en una carpeta temporal. Solo el embedder es el de
respaldo por palabras, para no depender de MiniLM.
"""

from __future__ import annotations

import asyncio
import time

import numpy as np
import pytest

from jarvis.cognition.llm import ToolUse
from jarvis.cognition.tool_loop import ToolSpec
from jarvis.contracts import (
    CascadeLevel,
    CascadeRecord,
    ConsolidationReport,
    Emotion,
    EventType,
    Response,
    RiskLevel,
    ToolCall,
    ToolResult,
)
from jarvis.contracts.events import Event
from jarvis.memory.extraction import EntityCandidate, Extraction, FactCandidate
from jarvis.memory.simulation import simulated_week
from jarvis.memory.store import dumps
from jarvis.memory.text import to_epoch
from tests.cognition_fakes import FakeLocal, Script
from tests.memory_fakes import ScriptedExtractor, SimClock, assistant, make_memory


async def _settle(bus) -> None:
    await bus.wait_idle(timeout=5.0)


# ---------------------------------------------------------------------------
# Criterio 1: decir un hecho y preguntarlo tres dias despues


async def test_un_hecho_dicho_se_recupera_tres_dias_despues(bus, tmp_path):
    clock = SimClock()
    local = FakeLocal([Script("Tu hermana se llama Ana.", 0.95)])
    rig, memory = assistant(bus, tmp_path, clock, local=local)
    await memory.start()

    saved = await rig.orchestrator.handle_text("Recuerda que mi hermana se llama Ana")
    assert saved.handled_by_level is CascadeLevel.FAST_PATH
    assert saved.text == "Anotado: tu hermana se llama Ana."
    await _settle(bus)

    clock.advance(days=3)
    answer = await rig.orchestrator.handle_text("¿Cómo se llama mi hermana?")
    assert answer.handled_by_level is CascadeLevel.LOCAL_LLM
    prompt = local.requests[-1].messages[-1].content
    assert "Lo que recuerdas" in prompt and "Tu hermana se llama Ana." in prompt

    about = await rig.orchestrator.handle_text("¿Qué sabes de mi hermana?")
    assert about.handled_by_level is CascadeLevel.FAST_PATH
    assert "tu hermana se llama Ana" in about.text
    assert "Ana" in {e.name for e in memory.graph.entities("persona")}
    await memory.stop()
    memory.close()


async def test_lo_dicho_dos_veces_no_se_duplica(bus, tmp_path):
    rig, memory = assistant(bus, tmp_path, SimClock())
    await rig.orchestrator.handle_text("Recuerda que mi jefe se llama Carlos")
    again = await rig.orchestrator.handle_text("anota que mi jefe se llama Carlos")
    assert again.text == "Eso ya lo tenía anotado."
    changed = await rig.orchestrator.handle_text("Recuerda que mi jefe se llama Raúl")
    assert changed.text == "Actualizado: tu jefe se llama Raúl."
    assert [f.content for f in memory.semantic.active()] == ["Tu jefe se llama Raúl."]
    memory.close()


# ---------------------------------------------------------------------------
# Criterio 2: recuperacion hibrida < 100 ms con 10.000 registros


def _seed(memory, facts: int, episodes: int) -> None:
    """Carga masiva directa (sin pasar por la deduplicacion, que es por hecho)."""
    words = [f"palabra{i}" for i in range(400)]
    rng = np.random.default_rng(3)
    now = to_epoch(memory.now())
    fact_rows, ids, texts = [], [], []
    for index in range(facts):
        text = " ".join(rng.choice(words, 8)) + f" hecho {index}."
        fact_id = f"f{index}"
        fact_rows.append((fact_id, text, "", 0.5, now, now, "prueba", dumps({})))
        ids.append(fact_id)
        texts.append(text)
    with memory.store.transaction() as conn:
        conn.executemany(
            "INSERT INTO facts (id, content, fact_key, importance, created_at, last_accessed_at, "
            "source, metadata) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            fact_rows,
        )
        conn.executemany(
            "INSERT INTO episodes (id, kind, content, user_text, at) VALUES (?, 'turno', ?, ?, ?)",
            [
                (f"e{i}", " ".join(rng.choice(words, 12)), "x", now - i * 60)
                for i in range(episodes)
            ],
        )
    memory.store.vectors.add(ids, memory.embedder.embed(texts))
    memory.graph.relate("María", "trabaja_en", "Proyecto Aurora")


async def test_la_recuperacion_tarda_menos_de_100_ms_con_10000_registros(bus, tmp_path):
    memory = make_memory(bus, tmp_path, SimClock())
    _seed(memory, facts=10_000, episodes=10_000)
    await memory.retriever.retrieve("calentar")  # primera llamada: hilos y caches
    timings = []
    for query in ("palabra12 palabra7", "qué hace María", "palabra300 hecho", "nada que ver"):
        started = time.perf_counter()
        result = await memory.retriever.retrieve(query)
        timings.append((time.perf_counter() - started) * 1000)
        assert result.timed_out == []
        assert result.tokens <= memory._config().retrieval_budget_tokens
    assert sorted(timings)[len(timings) // 2] < 100, timings
    maria = await memory.retriever.retrieve("¿qué hace María?")
    assert "María trabaja en Proyecto Aurora" in maria.texts
    memory.close()


async def test_la_recuperacion_respeta_el_presupuesto_de_cada_nivel(bus, tmp_path):
    memory = make_memory(bus, tmp_path, SimClock(), local_retrieval_tokens=20)
    for index in range(10):
        memory.semantic.add(f"El proyecto Aurora tiene la tarea número {index} pendiente",
                            importance=0.5, source="prueba")  # fmt: skip
    recalled = await memory.recollect("¿qué falta en el proyecto Aurora?")
    assert 0 < len(recalled.local) < len(recalled.cloud)
    assert sum(len(text) for text in recalled.local) / 3.5 <= 20
    memory.close()


async def test_una_busqueda_lenta_no_frena_la_recuperacion(bus, tmp_path):
    memory = make_memory(bus, tmp_path, SimClock(), retrieval_timeout_ms=50)
    memory.semantic.add("Te gusta el jazz", importance=0.6, source="prueba")
    original = memory.graph.mentioned_in

    def slow(text: str):
        time.sleep(0.3)
        return original(text)

    memory.graph.mentioned_in = slow  # type: ignore[method-assign]
    started = time.perf_counter()
    result = await memory.retriever.retrieve("jazz")
    assert (time.perf_counter() - started) < 0.25
    assert result.timed_out == ["graph"]
    assert "Te gusta el jazz." in result.texts
    await asyncio.sleep(0.35)  # que el hilo lento termine antes de cerrar
    memory.close()


# ---------------------------------------------------------------------------
# Criterio 3: consolidacion de 100 episodios simulados


def _load_week(memory, *, total: int = 100) -> None:
    for turn in simulated_week(memory.now(), total=total):
        memory.episodic.add(
            "turno", f"Usuario: {turn.user}\nJarvis: {turn.answer}", user_text=turn.user,
            answer=turn.answer, intent=turn.intent, level=turn.level, at=turn.at,
        )  # fmt: skip


async def test_la_consolidacion_de_100_episodios_produce_hechos_y_entidades(bus, tmp_path):
    llm = ScriptedExtractor(
        Extraction(
            facts=[FactCandidate("Tu hermana se llama Ana.", 0.9)],
            entities=[EntityCandidate("Ana", "persona"), EntityCandidate("Guadalajara", "lugar")],
            summary="Hablaron del trabajo y de la familia.",
        )
    )
    memory = make_memory(bus, tmp_path, SimClock(), extractor=llm)
    _load_week(memory)
    report = await memory.consolidator.run()

    assert report.episodes == 100 and report.extractor == "modelo local"
    assert len(llm.calls) >= 100 // memory._config().consolidation_batch
    facts = {f.content for f in memory.semantic.active()}
    for expected in (
        "Tu hermana se llama Ana.",
        "Tu jefe se llama Carlos.",
        "Tu perro se llama Toby.",
        "Vives en Guadalajara.",
        "Odias el tráfico de los lunes.",
    ):
        assert expected in facts
    names = {e.name for e in memory.graph.entities()}
    assert {"Ana", "Carlos", "Toby", "María", "Aurora", "informe.docx", "Guadalajara"} <= names
    maria = memory.graph.find("María")
    assert maria is not None
    reach = {t.text for t in memory.graph.traverse([maria.id], 2)}
    assert "María usa archivo informe.docx" in reach
    assert "Rhandy trabaja en Aurora" in reach  # a dos saltos, por Aurora
    assert report.facts_new == len(facts) and report.entities_new >= 7
    assert report.facts_reinforced > 0  # el "modelo" repitio a Ana en cada tramo
    assert report.summaries >= 1
    # El clima cada manana es un habito; las ordenes del nivel 1 nunca se proponen,
    # y de las 5 preguntas abiertas que se repiten a cualquier hora casi ninguna.
    proposals = {r.trigger: r.note for r in memory.procedural.routines("propuesta")}
    assert proposals["que clima hace"] == "La pediste 7 días hacia las 8:00."
    assert "que hora es" not in proposals and "sube el volumen" not in proposals
    assert report.routines_proposed == len(proposals) <= 2
    assert memory.episodic.unconsolidated(memory.now()) == []
    again = await memory.consolidator.run()
    assert again.episodes == 0  # nada se consolida dos veces
    memory.close()


async def test_sin_ollama_la_consolidacion_sigue_con_reglas_y_lo_dice(bus, tmp_path):
    memory = make_memory(bus, tmp_path, SimClock(), extractor=ScriptedExtractor(fail=True))
    _load_week(memory, total=30)
    report = await memory.consolidator.run()
    assert report.extractor == "modelo local y luego reglas"
    assert report.errors and "Ollama" in report.errors[0]
    assert report.facts_new > 0
    memory.close()


async def test_el_reporte_se_publica_y_se_dice_al_arrancar(bus, tmp_path):
    reports: list[ConsolidationReport] = []

    async def collect(event: Event) -> None:
        reports.append(event.payload_as(ConsolidationReport))

    bus.subscribe(EventType.MEMORY_CONSOLIDATION_FINISHED, collect)
    clock = SimClock()
    memory = make_memory(bus, tmp_path, clock, consolidation_enabled=False)
    _load_week(memory, total=20)
    report = await memory.scheduler.run_now()
    assert "Anoche aprendí" in report.spoken()
    memory.close()

    rig, reopened = assistant(bus, tmp_path, clock, consolidation_enabled=False)
    pending = await reopened.start()
    await _settle(bus)
    assert pending is not None and pending.facts_new == report.facts_new
    assert len(reports) == 2  # al terminar y otra vez al arrancar
    assert await reopened.start() is None  # ya se oyo: no se repite
    learned = await rig.orchestrator.handle_text("¿Qué aprendiste anoche?")
    assert learned.text == report.spoken()
    await reopened.stop()
    reopened.close()


async def test_el_programador_detecta_una_noche_perdida(bus, tmp_path):
    clock = SimClock()  # lunes 15:00; la consolidacion es a las 4:00
    memory = make_memory(bus, tmp_path, clock)
    assert not memory.scheduler.missed()
    memory.episodic.add("turno", "Usuario: hola", user_text="hola", at=clock().replace(hour=1))
    assert memory.scheduler.missed()
    await memory.scheduler.run_now()
    assert not memory.scheduler.missed()
    memory.close()


# ---------------------------------------------------------------------------
# Criterio 4: tres repeticiones de una secuencia generan una rutina


async def _publish_interaction(bus, text: str, cid: str, *, ok: bool = True) -> None:
    call = ToolCall(server="lights", tool="set", arguments={"room": "sala"},
                    risk_level=RiskLevel.VERDE, correlation_id=cid)  # fmt: skip
    result = ToolResult(call_id=call.call_id, success=ok, data=None,
                        error=None if ok else "sin respuesta", duration_ms=3.0)  # fmt: skip
    response = Response(text="Listo.", handled_by_level=CascadeLevel.LOCAL_LLM, actions=[call],
                        correlation_id=cid)  # fmt: skip
    record = CascadeRecord(
        text=text, intent="chat.free", intent_confidence=0.5,
        suggested_level=CascadeLevel.LOCAL_LLM, handled_by_level=CascadeLevel.LOCAL_LLM,
        levels_tried=[CascadeLevel.LOCAL_LLM], classify_ms=1.0, total_ms=900.0,
        correlation_id=cid,
    )  # fmt: skip
    for event_type, payload in (
        (EventType.TOOL_CALL_FINISHED, result),
        (EventType.COGNITION_RESPONSE, response),
        (EventType.COGNITION_COMPLETED, record),
    ):
        await bus.publish(Event.of(event_type, payload, source="prueba", correlation_id=cid))
    await _settle(bus)


async def test_tres_repeticiones_generan_una_rutina_de_ruta_rapida(bus, tmp_path):
    rig, memory = assistant(bus, tmp_path, SimClock())
    calls: list[dict[str, object]] = []

    async def lights(args: dict[str, object]) -> str:
        calls.append(args)
        return "ok"

    rig.tools.register(ToolSpec("lights", "set", "Luces.", {"type": "object"}, lights))
    await memory.start()
    for index in range(3):
        await _publish_interaction(bus, "Apaga la luz de la sala", f"c{index}")
    (routine,) = memory.procedural.routines("activa")
    assert routine.trigger == "apaga la luz de la sala"
    row = memory.store.query("SELECT successes FROM action_patterns")[0]
    assert row["successes"] == 3

    response = await rig.orchestrator.handle_text("Jarvis, apaga la luz de la sala")
    assert response.handled_by_level is CascadeLevel.FAST_PATH
    assert response.text == "Hecho."
    assert calls == [{"room": "sala"}]
    assert rig.local.requests == []  # ni un modelo
    await memory.stop()
    memory.close()


async def test_una_secuencia_que_falla_no_cuenta(bus, tmp_path):
    _, memory = assistant(bus, tmp_path, SimClock())
    await memory.start()
    for index in range(3):
        await _publish_interaction(bus, "apaga la luz", f"c{index}", ok=index != 1)
    assert memory.procedural.routines("activa") == []
    await memory.stop()
    memory.close()


async def test_un_modelo_que_usa_herramientas_alimenta_la_memoria_procedimental(bus, tmp_path):
    local = FakeLocal([
        Script("", tool_calls=[ToolUse("t1", "calc__sqrt", {"x": 2})]),
        Script("Da 1.41."),
    ])  # fmt: skip
    rig, memory = assistant(bus, tmp_path, SimClock(), local=local,
                            cognition={"local_tools": True})  # fmt: skip

    async def sqrt(args: dict[str, object]) -> float:
        return 1.41

    rig.tools.register(ToolSpec("calc", "sqrt", "Raiz.", {"type": "object"}, sqrt))
    await memory.start()
    response = await rig.orchestrator.handle_text("calcula la raíz cuadrada de 2")
    await _settle(bus)
    assert response.handled_by_level is CascadeLevel.LOCAL_LLM
    assert [a.tool for a in response.actions] == ["sqrt"]
    (pattern,) = memory.procedural.patterns()
    assert pattern.successes == 1 and pattern.steps[0].tool == "calc.sqrt"
    assert memory.episodic.counts_by_kind() == {"turno": 1, "accion": 1}
    await memory.stop()
    memory.close()


# ---------------------------------------------------------------------------
# Criterio 5: "olvida X" pide confirmacion y despues X ya no se recupera


async def test_olvidar_pide_confirmacion_y_despues_no_se_recupera(bus, tmp_path):
    clock = SimClock()
    local = FakeLocal([Script("Hablamos de la fiesta.", 0.9)])
    rig, memory = assistant(bus, tmp_path, clock, local=local)
    await memory.start()
    await rig.orchestrator.handle_text("Recuerda que la fiesta sorpresa de Ana es el sábado")
    await rig.orchestrator.handle_text("¿Qué opinas de la fiesta sorpresa?")
    await _settle(bus)
    assert (await memory.retriever.retrieve("fiesta sorpresa")).items

    asked = await rig.orchestrator.handle_text("Olvida lo de la fiesta sorpresa")
    assert asked.emotion is Emotion.INTERROGATIVA
    assert "¿Lo olvido?" in asked.text
    assert memory.semantic.count() == 1  # nada se borro todavia

    done = await rig.orchestrator.handle_text("Sí")
    assert done.text.startswith("Olvidado. Borré")
    await _settle(bus)
    assert (await memory.retriever.retrieve("fiesta sorpresa")).items == []
    assert memory.semantic.count() == 0
    assert all("fiesta" not in t.user for t in memory.working.turns)
    about = await rig.orchestrator.handle_text("¿Qué sabes de la fiesta sorpresa?")
    assert about.text == "No sé nada de fiesta sorpresa todavía."
    again = await rig.orchestrator.handle_text("Sí")
    assert again.text == "No hay nada pendiente de confirmar."
    await memory.stop()
    memory.close()


async def test_decir_no_o_dejar_pasar_el_plazo_no_borra_nada(bus, tmp_path):
    clock = SimClock()
    rig, memory = assistant(bus, tmp_path, clock)
    await rig.orchestrator.handle_text("Recuerda que mi perro se llama Toby")
    await rig.orchestrator.handle_text("Olvida lo de mi perro")
    denied = await rig.orchestrator.handle_text("No")
    assert denied.text == "Entendido, no toco nada."
    await rig.orchestrator.handle_text("Olvida lo de mi perro")
    clock.advance(seconds=20)  # el plazo es de 15 s
    late = await rig.orchestrator.handle_text("Sí")
    assert late.text == "No hay nada pendiente de confirmar."
    assert memory.semantic.count() == 1
    nothing = await rig.orchestrator.handle_text("Olvida lo de los dinosaurios")
    assert nothing.text == "No tengo nada guardado sobre dinosaurios."
    memory.close()


# ---------------------------------------------------------------------------
# Criterio 7: reiniciar el proceso conserva todo


async def test_reiniciar_conserva_absolutamente_todo(bus, tmp_path):
    clock = SimClock()
    rig, memory = assistant(bus, tmp_path, clock)
    await memory.start()
    await rig.orchestrator.handle_text("Recuerda que mi hermana se llama Ana")
    memory.graph.relate("María", "trabaja_en", "Aurora")
    for index in range(3):
        await _publish_interaction(bus, "modo cine", f"c{index}")
    await memory.scheduler.run_now()
    before = memory.stats()
    await memory.stop()
    memory.close()

    rig2, reopened = assistant(bus, tmp_path, clock)
    after = reopened.stats()
    before.pop("trabajo (turnos en RAM)")
    after.pop("trabajo (turnos en RAM)")  # la de trabajo vive en RAM: se vacia, como debe
    assert after == before
    assert after["rutinas activas"] == 1 and after["hechos activos"] >= 1
    assert "Tu hermana se llama Ana." in (await reopened.retriever.retrieve("hermana")).texts
    assert rig2.orchestrator.classifier.classify("modo cine").stage == "routine"
    assert reopened.scheduler.last_report() is not None
    reopened.close()


# ---------------------------------------------------------------------------
# Voz: linea de tiempo, privacidad del olvido, herramientas


async def test_que_hicimos_ayer(bus, tmp_path):
    clock = SimClock()
    rig, memory = assistant(bus, tmp_path, clock)
    await memory.start()
    for text in ("Pon un temporizador de 5 minutos", "¿Qué hora es?", "Sube el volumen"):
        await rig.orchestrator.handle_text(text)
    await _settle(bus)
    clock.advance(days=1)
    response = await rig.orchestrator.handle_text("¿Qué hicimos ayer?")
    assert response.text.startswith("Ayer me pediste 3 cosas. Entre ellas: «Pon un temporizador")
    empty = await rig.orchestrator.handle_text("¿Qué hicimos el jueves?")
    assert empty.text == "El jueves no hablamos de nada."
    await memory.stop()
    memory.close()


async def test_olvidar_no_deja_el_tema_en_la_episodica(bus, tmp_path):
    rig, memory = assistant(bus, tmp_path, SimClock())
    await memory.start()
    await rig.orchestrator.handle_text("Olvida lo de mi expareja")
    await _settle(bus)
    (episode,) = memory.episodic.recent(1)
    assert "expareja" not in episode.content and episode.user_text is None
    await memory.stop()
    memory.close()


async def test_las_herramientas_de_memoria(bus, tmp_path):
    rig, memory = assistant(bus, tmp_path, SimClock())
    tools = rig.orchestrator.tools
    names = {d.name for d in tools.definitions()}
    expected = {"recordar", "guardar_hecho", "olvidar", "listar_entidades", "relacionar",
                "linea_de_tiempo"}  # fmt: skip
    assert {f"memory__{name}" for name in expected} <= names
    _, saved = await tools.call("memory.guardar_hecho", {"hecho": "Te gusta el té verde"}, "c")
    assert saved.data == {"resultado": "nuevo", "hecho": "Te gusta el té verde."}
    _, found = await tools.call("memory.recordar", {"consulta": "té verde"}, "c")
    assert found.data == {"recuerdos": ["Te gusta el té verde."]}
    _, related = await tools.call(
        "memory.relacionar", {"a": "Ana", "relacion": "vive en", "b": "Lima"}, "c"
    )
    assert related.data == {"relacion": "Ana vive en Lima"}
    _, listed = await tools.call("memory.listar_entidades", {}, "c")
    assert {item["nombre"] for item in listed.data} == {"Ana", "Lima"}  # type: ignore[union-attr,index]
    _, staged = await tools.call("memory.olvidar", {"consulta": "té verde"}, "c")
    assert staged.data["pendiente"] is True  # type: ignore[index,call-overload]
    assert memory.semantic.count() == 1  # la herramienta sola no borra
    confirmed = await rig.orchestrator.handle_text("sí")
    assert confirmed.text.startswith("Olvidado")
    assert memory.semantic.count() == 0
    _, timeline = await tools.call("memory.linea_de_tiempo", {"desde": "2026-11-01"}, "c")
    assert isinstance(timeline.data, list)
    memory.close()


@pytest.mark.parametrize(
    "text",
    [
        "Recuerda que mi hermana se llama Ana",
        "Olvida lo de la reunión",
        "¿Qué sabes de María?",
        "¿Qué hicimos ayer?",
        "¿Qué aprendiste anoche?",
        "Sí",
        "No",
    ],
)
def test_las_ordenes_de_memoria_se_clasifican_en_el_nivel_1(bus, tmp_path, text):
    rig, memory = assistant(bus, tmp_path, SimClock())
    intent = rig.orchestrator.classifier.classify(text).intent
    assert intent.suggested_level is CascadeLevel.FAST_PATH
    assert rig.orchestrator.fast_path.handles(intent.name)
    memory.close()


async def test_la_consolidacion_cede_el_modelo_si_el_usuario_esta_hablando(bus, tmp_path):
    clock = SimClock()
    memory = make_memory(bus, tmp_path, clock, consolidation_idle_minutes=2.0)
    memory.episodic.add("turno", "Usuario: hola", user_text="hola", at=clock())
    memory.working.snapshot.last_interaction_at = clock()
    assert memory._user_active()
    clock.advance(minutes=3)
    assert not memory._user_active()
    memory.close()
