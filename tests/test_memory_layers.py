"""Las capas de memoria por separado: almacenamiento, episodica, semantica, grafo,
procedimental, de trabajo, y las utilidades de texto y periodos."""

from __future__ import annotations

from datetime import timedelta

import pytest

from jarvis.cognition.context import Turn
from jarvis.contracts import CascadeLevel, CoreState, Event, EventType, StateChange
from jarvis.memory.episodic import EpisodicMemory
from jarvis.memory.extraction import RuleExtractor, remembered_fact
from jarvis.memory.graph import EntityGraph, entity_key
from jarvis.memory.periods import period_bounds
from jarvis.memory.procedural import ProceduralMemory, Step
from jarvis.memory.store import MIGRATIONS, MemoryStore
from jarvis.memory.text import fact_key, fold, fts_query, raw_capture, to_second_person
from jarvis.memory.working import WorkingMemory
from tests.memory_fakes import START, SimClock, make_memory

# ---------------------------------------------------------------------------
# Almacenamiento


def test_las_migraciones_corren_una_vez_y_quedan_versionadas(tmp_path):
    store = MemoryStore(tmp_path, vector_dim=8)
    assert store.schema_version == len(MIGRATIONS)
    store.set_meta("prueba", "1")
    store.close()
    again = MemoryStore(tmp_path, vector_dim=8)
    assert again.schema_version == len(MIGRATIONS)
    assert again.get_meta("prueba") == "1"
    again.close()


def test_cambiar_la_dimension_del_modelo_recrea_el_indice(tmp_path):
    import numpy as np

    store = MemoryStore(tmp_path, vector_dim=8)
    store.vectors.add(["a"], np.ones((1, 8), dtype=np.float32) / np.sqrt(8))
    assert store.vectors.count() == 1
    store.close()
    other = MemoryStore(tmp_path, vector_dim=16)
    assert other.vectors.count() == 0
    other.close()


# ---------------------------------------------------------------------------
# Episodica


def test_episodica_busca_por_palabra_y_por_fecha(tmp_path):
    clock = SimClock()
    store = MemoryStore(tmp_path, vector_dim=8)
    episodic = EpisodicMemory(store, clock)
    episodic.add("turno", "Usuario: agenda la reunión con Laura", user_text="agenda la reunión")
    clock.advance(days=1)
    episodic.add("turno", "Usuario: pon jazz", user_text="pon jazz")
    hits = episodic.search("reunion", 5)  # sin tilde: FTS5 quita los acentos
    assert [e.user_text for e, _ in hits] == ["agenda la reunión"]
    yesterday = episodic.between(START.replace(hour=0), START.replace(hour=23, minute=59))
    assert [e.user_text for e in yesterday] == ["agenda la reunión"]
    store.close()


def test_una_consulta_con_sintaxis_fts_no_rompe_la_busqueda(tmp_path):
    store = MemoryStore(tmp_path, vector_dim=8)
    episodic = EpisodicMemory(store, SimClock())
    episodic.add("turno", "Usuario: hola")
    assert episodic.search('NEAR( "x" AND * OR', 5) == []
    assert fts_query("¿qué?") is None
    store.close()


def test_los_episodios_viejos_se_comprimen_en_uno_por_dia(tmp_path):
    clock = SimClock()
    store = MemoryStore(tmp_path, vector_dim=8)
    episodic = EpisodicMemory(store, clock)
    for text in ("pon jazz", "sube el volumen", "qué hora es"):
        episodic.add("turno", f"Usuario: {text}", user_text=text)
    episodic.add("error", "Error en voz")
    clock.advance(days=100)
    episodic.add("turno", "Usuario: hoy", user_text="hoy")
    compressed = episodic.compress_before(clock() - timedelta(days=90))
    assert compressed == 4
    kinds = episodic.counts_by_kind()
    assert kinds == {"resumen": 1, "turno": 1}
    (summary,) = [e for e, _ in episodic.search("jazz", 5)]
    assert summary.compressed and "sube el volumen" in summary.content
    store.close()


# ---------------------------------------------------------------------------
# Semantica


def test_un_hecho_nuevo_un_duplicado_y_uno_que_cambio(bus, tmp_path):
    memory = make_memory(bus, tmp_path, SimClock())
    first = memory.semantic.add("Tu hermana se llama Ana", importance=0.9, source="usuario")
    assert first.outcome == "nuevo" and first.fact.content == "Tu hermana se llama Ana."
    again = memory.semantic.add("tu hermana se llama Ana.", importance=0.5, source="usuario")
    assert again.outcome == "duplicado" and again.fact.importance == 0.9
    changed = memory.semantic.add("Tu hermana se llama Lucía", importance=0.9, source="usuario")
    assert changed.outcome == "actualizado"
    assert [old.id for old in changed.replaced] == [first.fact.id]
    assert [f.content for f in memory.semantic.active()] == ["Tu hermana se llama Lucía."]
    assert memory.semantic.count("obsoleto") == 1
    assert memory.semantic.history(changed.fact.id)[0].content == "Tu hermana se llama Ana."
    memory.close()


def test_el_decaimiento_baja_lo_olvidado_pero_no_lo_permanente(bus, tmp_path):
    clock = SimClock()
    memory = make_memory(bus, tmp_path, clock, decay_half_life_days=30.0)
    memory.semantic.add("Te gusta el jazz", importance=0.6, source="usuario")
    memory.semantic.add("Tu cumpleaños es el 4 de mayo", importance=1.0, source="usuario")
    clock.advance(days=30)
    assert memory.semantic.apply_decay() == 1
    by_content = {f.content: f for f in memory.semantic.active()}
    assert by_content["Te gusta el jazz."].decay == pytest.approx(0.5, abs=0.01)
    assert by_content["Tu cumpleaños es el 4 de mayo."].decay == 0.0
    memory.close()


# ---------------------------------------------------------------------------
# Grafo


def test_el_grafo_responde_una_consulta_de_dos_saltos(tmp_path):
    store = MemoryStore(tmp_path, vector_dim=8)
    graph = EntityGraph(store, SimClock())
    graph.relate("María", "trabaja_en", "Proyecto Aurora", source_type="persona",
                 target_type="proyecto")  # fmt: skip
    graph.relate("Proyecto Aurora", "usa_archivo", "informe.docx", target_type="archivo")
    graph.relate("Pedro", "vive_en", "Monterrey", source_type="persona", target_type="lugar")
    maria = graph.find("maria")
    assert maria is not None and maria.type == "persona"
    one_hop = {t.text for t in graph.traverse([maria.id], 1)}
    assert one_hop == {"María trabaja en Proyecto Aurora"}
    two_hops = {(t.text, t.hops) for t in graph.traverse([maria.id], 2)}
    assert ("Proyecto Aurora usa archivo informe.docx", 2) in two_hops
    assert all("Pedro" not in text for text, _ in two_hops)
    path = graph.path("María", "informe.docx", max_depth=2)
    assert path is not None and [t.relation for t in path] == ["trabaja_en", "usa_archivo"]
    assert graph.path("María", "Monterrey", max_depth=3) is None
    assert graph.find("proyecto aurorra") is not None  # error de transcripcion
    assert [e.name for e in graph.mentioned_in("¿qué hace María hoy?")] == ["María"]
    store.close()


def test_las_entidades_se_normalizan_y_cuentan_menciones(tmp_path):
    store = MemoryStore(tmp_path, vector_dim=8)
    graph = EntityGraph(store, SimClock())
    graph.upsert("el Proyecto Aurora", "proyecto")
    again = graph.upsert("Aurora", "cosa")
    assert entity_key("el Proyecto Aurora") == "proyecto aurora"
    assert again.type == "cosa"  # "Aurora" a secas es otra clave
    assert graph.upsert("proyecto aurora").mentions == 2  # "cosa" hereda el tipo existente
    assert [e.name for e in graph.entities("proyecto")] == ["el Proyecto Aurora"]
    store.close()


# ---------------------------------------------------------------------------
# Procedimental


def test_tres_repeticiones_exitosas_promueven_una_rutina(tmp_path):
    store = MemoryStore(tmp_path, vector_dim=8)
    procedural = ProceduralMemory(store, SimClock(), lambda: 3)
    steps = [Step("lights.set", {"room": "sala", "on": False}), Step("media.pause")]
    assert procedural.observe("Jarvis, modo cine", steps, success=True) is None
    assert procedural.observe("modo cine por favor", steps, success=False) is None
    assert procedural.observe("modo cine", steps, success=True) is None
    routine = procedural.observe("Modo cine", steps, success=True)
    assert routine is not None and routine.status == "activa"
    assert routine.trigger == "modo cine" and routine.steps == tuple(steps)
    (pattern,) = procedural.patterns()
    assert (pattern.successes, pattern.failures) == (3, 1)
    assert procedural.observe("modo cine", steps, success=True) is None  # ya era rutina
    store.close()


def test_una_rutina_que_falla_dos_veces_seguidas_se_retira(tmp_path):
    store = MemoryStore(tmp_path, vector_dim=8)
    procedural = ProceduralMemory(store, SimClock(), lambda: 1)
    routine = procedural.observe("modo cine", [Step("media.pause")], success=True)
    assert routine is not None
    procedural.record_use(routine.id, success=False)
    procedural.record_use(routine.id, success=True)  # un exito borra el tropiezo
    procedural.record_use(routine.id, success=False)
    assert procedural.routines("activa")
    procedural.record_use(routine.id, success=False)
    assert procedural.routines("retirada")[0].id == routine.id
    store.close()


# ---------------------------------------------------------------------------
# Trabajo


def test_la_memoria_de_trabajo_sigue_el_estado_y_olvida_en_ram():
    working = WorkingMemory(lambda: 4, SimClock())
    working.add(Turn("hablemos de la fiesta de Ana", "Claro.", CascadeLevel.LOCAL_LLM, START))
    working.add(Turn("¿qué opinas de Rust?", "Es sólido.", CascadeLevel.LOCAL_LLM, START))
    change = StateChange(previous=CoreState.DORMIDO, current=CoreState.PENSANDO, reason="t",
                         source="t")  # fmt: skip
    working.observe(Event.of(EventType.CORE_STATE_CHANGED, change, source="t"))
    assert working.snapshot.state is CoreState.PENSANDO
    assert working.forget(["fiesta"]) == 1
    assert [t.user for t in working.turns] == ["¿qué opinas de Rust?"]
    working.resize(99)  # el tope lo manda memory.working_turns, no la cascada
    for index in range(6):
        working.add(Turn(f"orden {index}", "ok", CascadeLevel.LOCAL_LLM, START))
    assert len(working.turns) <= 4


# ---------------------------------------------------------------------------
# Texto, reglas y periodos


@pytest.mark.parametrize(
    ("said", "fact"),
    [
        ("Recuerda que mi hermana se llama Ana", "Tu hermana se llama Ana."),
        ("Jarvis, anota que el wifi de casa es Casa123", "El wifi de casa es Casa123."),
        ("no olvides que prefiero el café sin azúcar", "Prefieres el café sin azúcar."),
        ("acuérdate de que mi jefe se llama Carlos", "Tu jefe se llama Carlos."),
        ("¿qué hora es?", None),
    ],
)
def test_recuerda_que_extrae_el_hecho_en_segunda_persona(said, fact):
    assert remembered_fact(said) == fact


def test_utilidades_de_texto():
    assert to_second_person("Mi perro se llama Toby y yo lo quiero") == (
        "Tu perro se llama Toby y tú lo quieres"
    )
    assert fact_key("Tu hermana se llama Ana.") == "tu hermana se llama"
    assert fact_key("Llueve mucho") == ""
    assert len(fold("Ñandú Árbol")) == len("Ñandú Árbol")
    import re

    pattern = re.compile(r"olvida lo de (?P<topic>.+)$")
    assert raw_capture("Olvida lo de la Reunión.", pattern, "topic") == "la Reunión"


def test_las_reglas_sacan_entidades_y_relaciones():
    rules = RuleExtractor(lambda: "Rhandy")
    out = rules.from_text("María trabaja en el proyecto Aurora, y mi perro se llama Toby")
    relations = {(r.source, r.relation, r.target) for r in out.relations}
    assert ("María", "trabaja_en", "Aurora") in relations
    assert ("Toby", "es_perro_de", "Rhandy") in relations
    assert "Tu perro se llama Toby." in [f.content for f in out.facts]
    files = rules.from_text("María usa el archivo informe.docx")
    assert files.relations[0].target == "informe.docx"
    assert files.relations[0].target_type == "archivo"


@pytest.mark.parametrize(
    ("when", "label", "start_day", "hours"),
    [
        ("", "Hoy", 2, None),
        ("ayer", "Ayer", 1, 24),
        ("anoche", "Anoche", 1, 12),
        ("el sabado", "El sábado", 31, 24),  # START es lunes 2 de noviembre
        ("esta semana", "Esta semana", 2, None),
    ],
)
def test_los_periodos_hablados(when, label, start_day, hours):
    start, end, said = period_bounds(when, START)
    assert said == label
    assert start.day == start_day
    if hours is not None:
        assert (end - start) == timedelta(hours=hours)


def test_del_modelo_solo_se_guardan_hechos_sobre_el_usuario():
    from jarvis.memory.extraction import about_user

    people = {"toby", "ana"}
    assert about_user("Tu hermana se llama Ana.", people)
    assert about_user("Toby está enfermo.", people)
    assert not about_user("La capital de Mongolia es Ulán Bator.", people)
    assert not about_user("Pon un temporizador de 5 minutos.", people)
    assert not about_user("Rust es un lenguaje sólido para sistemas.", people)
    assert to_second_person("Me gusta el jazz para concentrarme") == (
        "Te gusta el jazz para concentrarte"
    )


def test_una_entidad_generica_se_promueve_al_conocer_su_tipo(tmp_path):
    store = MemoryStore(tmp_path, vector_dim=8)
    graph = EntityGraph(store, SimClock())
    graph.upsert("Rhandy")  # el modelo lo nombro sin tipo
    graph.upsert("Rhandy", "persona")  # las reglas saben que es una persona
    assert [(e.name, e.type) for e in graph.entities()] == [("Rhandy", "persona")]
    store.close()
