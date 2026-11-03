"""Pruebas unitarias de las piezas puras de la cascada."""

from __future__ import annotations

import itertools
import random
from datetime import date, datetime

import pytest

from jarvis.cognition.budget import SpendLedger, cost_usd
from jarvis.cognition.catalog import build_catalog
from jarvis.cognition.classifier import IntentClassifier, strip_fillers
from jarvis.cognition.context import ContextBuilder, ConversationHistory, Turn, estimate_tokens
from jarvis.cognition.fastpath import FastPath
from jarvis.cognition.llm import Complexity, GateDecision
from jarvis.cognition.local_llm import combine_confidence
from jarvis.cognition.phrases import PHRASES, PhrasePicker
from jarvis.cognition.router import PrivacySwitch, Router, profile_task
from jarvis.cognition.spoken import spoken_date, spoken_duration, spoken_time
from jarvis.cognition.streaming import SentenceChunker, StructuredStream
from jarvis.cognition.text import duration_seconds, normalize
from jarvis.config import CognitionConfig, ModelPrice
from jarvis.contracts import CascadeLevel, Intent, PrivacyMode
from tests.cognition_fakes import FakeEmbedder

# ---------------------------------------------------------------------------
# Texto
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("¿Súbele al volumen a setenta por ciento!", "subele al volumen a 70%"),
        ("pon el volumen al cincuenta y cinco", "pon el volumen al 55"),
        ("sube a ciento veinte", "sube a 120"),
        ("Entra a YouTube.com.", "entra a youtube.com"),
        ("abre una pestaña", "abre una pestaña"),
        ("temporizador de un minuto", "temporizador de 1 minuto"),
    ],
)
def test_normalize(raw, expected):
    assert normalize(raw) == expected


def test_duraciones():
    assert duration_seconds("5 minutos y 30 segundos") == 330
    assert duration_seconds("media hora") == 1800
    assert duration_seconds("2 horas") == 7200
    assert duration_seconds("abre spotify") is None


def test_quita_muletillas_y_cortesias():
    assert strip_fillers("oye jarvis podrias subir el volumen por favor") == "subir el volumen"
    assert strip_fillers("por favor") == "por favor"


def test_duracion_hablada():
    assert spoken_duration(300) == "5 minutos"
    assert spoken_duration(61) == "1 minuto y 1 segundo"
    assert spoken_duration(3900) == "1 hora y 5 minutos"


def test_hora_y_fecha_habladas():
    assert spoken_time(datetime(2026, 11, 2, 15, 25)) == "Son las 3 y 25 de la tarde."
    assert spoken_time(datetime(2026, 11, 2, 13, 0)) == "Es la 1 en punto de la tarde."
    assert spoken_date(datetime(2026, 11, 2)) == "Hoy es lunes 2 de noviembre."


# ---------------------------------------------------------------------------
# Catalogo y clasificador
# ---------------------------------------------------------------------------


def test_el_catalogo_cumple_los_minimos_del_documento():
    catalog = build_catalog()
    assert len(catalog) >= 40
    fast = [item for item in catalog if item.level is CascadeLevel.FAST_PATH]
    handled = FastPath().intents
    assert len(handled) >= 25
    # Las de memoria las registra el sistema de memoria al conectarse (test_memory_system).
    own = {item.name for item in fast if item.domain != "memory"}
    assert own <= set(handled), "toda intencion fija tiene manejador"
    domains = {item.domain for item in catalog}
    for needed in (
        "media",
        "apps",
        "system",
        "files",
        "browser",
        "query",
        "chat",
        "memory",
        "jarvis",
    ):
        assert needed in domains


@pytest.fixture
def classifier() -> IntentClassifier:
    config = CognitionConfig()
    return IntentClassifier(build_catalog(), lambda: config)


@pytest.mark.parametrize(
    ("text", "intent", "entities"),
    [
        ("Sube el volumen", "media.volume_up", {}),
        ("Súbele al volumen a setenta por ciento", "media.volume_set", {"value": 70}),
        ("Sierra todas las ventanas", "windows.close_all", {}),
        ("Maja el brillo", "system.brightness_down", {}),
        ("oye Jarvis, ¿podrías abrir Spotify, por favor?", "apps.open", {"app": "spotify"}),
        (
            "pon un temporizador de 5 minutos para la pasta",
            "timer.set",
            {"seconds": 300, "label": "pasta"},
        ),
        ("pon una alarma en media hora", "timer.set", {"seconds": 1800}),
        ("quita la alarma", "timer.cancel", {}),
        ("¿Qué hora es?", "system.time", {}),
        ("¿Qué opinas de la música clásica?", "chat.opinion", {}),
        ("Analiza este archivo de 300 líneas y refactorízalo", "code.refactor", {}),
        ("abre el archivo presupuesto.xlsx", "files.open", {"query": "presupuesto.xlsx"}),
        ("entra a youtube.com", "browser.open", {"query": "youtube.com"}),
        ("modo local", "jarvis.privacy_local", {}),
        ("puedes repetir", "jarvis.repeat", {}),
        ("gracias jarvis", "chat.thanks", {}),
    ],
)
def test_clasifica_por_patron(classifier, text, intent, entities):
    result = classifier.classify(text)
    assert result.intent.name == intent
    assert result.intent.entities == entities
    assert result.stage == "pattern"


def test_la_difusa_rescata_errores_de_transcripcion(classifier):
    result = classifier.classify("sube el bolumen")
    assert result.stage in {"pattern", "fuzzy"}
    assert result.intent.name == "media.volume_up"


def test_sin_coincidencia_ni_embeddings_es_charla(classifier):
    result = classifier.classify("me siento raro hoy, no sé por qué")
    assert result.intent.name == "chat.free"
    assert result.intent.suggested_level is CascadeLevel.LOCAL_LLM
    assert result.stage == "fallback"


def test_la_etapa_de_embeddings_encuentra_el_ejemplo():
    config = CognitionConfig()
    clf = IntentClassifier(build_catalog(), lambda: config, FakeEmbedder())
    result = clf.classify("no se escucha nada")
    assert result.stage == "embedding"
    assert result.intent.name == "media.volume_up"
    assert result.intent.suggested_level is CascadeLevel.FAST_PATH


def test_embedding_parecido_pero_dudoso_no_va_al_nivel_1():
    config = CognitionConfig(embedding_fast_similarity=0.99, embedding_min_similarity=0.3)
    clf = IntentClassifier(build_catalog(), lambda: config, FakeEmbedder())
    result = clf.classify("no se escucha casi nada aqui")
    assert result.intent.suggested_level is CascadeLevel.LOCAL_LLM


def test_el_clasificador_por_patrones_tarda_menos_de_10_ms(classifier):
    for text in ["sube el volumen", "que opinas del cafe", "pon musica de queen"] * 20:
        assert classifier.classify(text).elapsed_ms < 10.0


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def test_json_incremental_lee_confianza_antes_que_la_respuesta():
    parser = StructuredStream()
    pieces = ['{"conf', 'ianza": 0.8', '5, "respuesta": "Hola', ', \\"amigo\\"', '. Bien."}']
    seen = [parser.feed(p) for p in pieces]
    assert parser.confidence == pytest.approx(0.85)
    assert "".join(seen) == 'Hola, "amigo". Bien.'
    assert parser.answer_closed
    assert parser.finish() == (pytest.approx(0.85), 'Hola, "amigo". Bien.')


def test_json_incremental_con_escape_partido_y_unicode():
    parser = StructuredStream()
    parser.feed('{"confianza": 1, "respuesta": "caf\\')
    parser.feed("u00e9 listo\\")
    parser.feed('n"}')
    assert parser.answer == "café listo\n"


def test_las_frases_se_cortan_y_respetan_el_tope():
    chunker = SentenceChunker(max_sentences=2)
    out = chunker.feed("Uno. Dos")
    out += chunker.feed(" sigue. Tres. Cuatro.")
    out += chunker.flush()
    assert out == ["Uno.", "Dos sigue."]
    assert chunker.overflowed


def test_el_codigo_nunca_se_dice():
    chunker = SentenceChunker(max_sentences=5)
    out = chunker.feed("Aquí está. ```python\nprint(1)\n```")
    out += chunker.flush()
    assert out == ["Aquí está."]
    assert chunker.overflowed


def test_confianza_combinada():
    assert combine_confidence([], 0.8, 0.5) == pytest.approx(0.8)
    assert combine_confidence([0.0, 0.0], None, 0.5) == pytest.approx(1.0)
    assert combine_confidence([0.0], 0.4, 0.5) == pytest.approx(0.7)
    assert combine_confidence([], None, 0.5) == 0.0


# ---------------------------------------------------------------------------
# Gasto
# ---------------------------------------------------------------------------


def test_costo_por_tabla_de_precios():
    prices = {"m": ModelPrice(input_per_mtok=2.0, output_per_mtok=10.0)}
    assert cost_usd(prices, "m", input_tokens=1_000_000, output_tokens=0) == pytest.approx(2.0)
    assert cost_usd(
        prices, "m", input_tokens=0, output_tokens=0, cache_read_tokens=1_000_000
    ) == pytest.approx(0.2)
    with pytest.raises(KeyError):
        cost_usd(prices, "desconocido", input_tokens=1, output_tokens=1)


def test_el_gasto_persiste_y_se_reinicia_al_cambiar_de_dia(tmp_path):
    path = tmp_path / "spend.json"
    day = [date(2026, 11, 2)]
    ledger = SpendLedger(path, lambda: 0.10, today=lambda: day[0])
    first = ledger.record(0.06)
    assert not first.exhausted
    second = ledger.record(0.05)
    assert second.exhausted and second.just_exhausted
    assert not ledger.can_spend()

    # Reiniciar no regala presupuesto.
    again = SpendLedger(path, lambda: 0.10, today=lambda: day[0])
    assert again.spent_today == pytest.approx(0.11)
    assert not again.can_spend()

    day[0] = date(2026, 11, 3)
    assert again.spent_today == 0.0
    assert again.can_spend()


def test_presupuesto_cero_es_sin_limite():
    ledger = SpendLedger(None, lambda: 0.0)
    ledger.record(100.0)
    assert ledger.can_spend()


# ---------------------------------------------------------------------------
# Enrutador
# ---------------------------------------------------------------------------


def _intent(name: str, level: CascadeLevel, confidence: float = 0.97) -> Intent:
    return Intent(name=name, confidence=confidence, suggested_level=level)


def test_enrutador_reglas_basicas():
    router = Router(lambda: CognitionConfig())

    def route(text, intent, *, privacy=PrivacyMode.HYBRID, blocker=None):
        return router.decide(
            intent,
            profile_task(text, 100),
            privacy=privacy,
            cloud_blocker=blocker,
            fast_path_handles=True,
        )

    fast = route("sube el volumen", _intent("media.volume_up", CascadeLevel.FAST_PATH))
    assert fast.level is CascadeLevel.FAST_PATH

    low = route("sube el volumen", _intent("media.volume_up", CascadeLevel.FAST_PATH, 0.5))
    assert low.level is CascadeLevel.LOCAL_LLM

    chat = route("que opinas del cafe", _intent("chat.opinion", CascadeLevel.LOCAL_LLM))
    assert chat.level is CascadeLevel.LOCAL_LLM

    code = route(
        "analiza este archivo de 300 lineas y refactorizalo",
        _intent("code.refactor", CascadeLevel.CLOUD_LLM),
    )
    assert code.level is CascadeLevel.CLOUD_LLM
    assert code.complexity is Complexity.STANDARD

    steps = route(
        "abre el correo, luego responde, despues archiva y ademas agenda y por ultimo avisa",
        _intent("chat.free", CascadeLevel.LOCAL_LLM),
    )
    assert steps.level is CascadeLevel.CLOUD_LLM

    deep = route("usa opus y piensa a fondo", _intent("cloud.explicit", CascadeLevel.CLOUD_LLM))
    assert deep.complexity is Complexity.DEEP

    blocked = route(
        "refactoriza esto",
        _intent("code.refactor", CascadeLevel.CLOUD_LLM),
        privacy=PrivacyMode.LOCAL_ONLY,
        blocker="modo solo-local",
    )
    assert blocked.level is CascadeLevel.LOCAL_LLM
    assert blocked.degraded

    preferred = route(
        "que opinas del cafe",
        _intent("chat.opinion", CascadeLevel.LOCAL_LLM),
        privacy=PrivacyMode.CLOUD_PREFERRED,
    )
    assert preferred.level is CascadeLevel.CLOUD_LLM


def test_contexto_enorme_escala():
    router = Router(lambda: CognitionConfig(max_context_tokens=1000))
    profile = profile_task("resume la conversacion", 5000)
    route = router.decide(
        _intent("chat.free", CascadeLevel.LOCAL_LLM),
        profile,
        privacy=PrivacyMode.HYBRID,
        cloud_blocker=None,
        fast_path_handles=False,
    )
    assert route.level is CascadeLevel.CLOUD_LLM


def test_compuerta_de_confianza():
    router = Router(lambda: CognitionConfig(local_confidence_threshold=0.6))
    gate = router.gate(cloud_blocker=None, margin=0.15)
    assert gate(0.9) is GateDecision.STREAM
    assert gate(0.65) is GateDecision.BUFFER
    assert gate(0.3) is GateDecision.ABORT
    # Sin nube a donde escalar, nunca se aborta: se mide y se contesta con advertencia.
    assert router.gate(cloud_blocker="sin clave", margin=0.15)(0.3) is GateDecision.BUFFER


def test_interruptor_de_privacidad():
    switch = PrivacySwitch(lambda: PrivacyMode.HYBRID)
    assert switch.network_allowed()
    switch.set(PrivacyMode.LOCAL_ONLY)
    assert not switch.network_allowed()


# ---------------------------------------------------------------------------
# Contexto
# ---------------------------------------------------------------------------


def _turn(n: int) -> Turn:
    return Turn(
        f"pregunta {n} " * 10, f"respuesta {n} " * 10, CascadeLevel.LOCAL_LLM, datetime.now()
    )


def test_historial_resume_lo_viejo():
    history = ConversationHistory(max_turns=2)
    for n in range(4):
        history.add(_turn(n))
    assert len(history.turns) == 2
    assert len(history.summaries) == 2
    assert history.summaries[0].startswith("Usuario: pregunta 0")


def test_contexto_poda_por_prioridad():
    history = ConversationHistory(max_turns=10)
    for n in range(10):
        history.add(_turn(n))
    built = ContextBuilder().build(
        system="sistema " * 20,
        user_text="la orden",
        budget_tokens=200,
        history=history,
        state="15:25, modo HYBRID",
    )
    assert built.messages[0].role == "system"
    assert built.messages[-1].content.endswith("la orden")
    assert "15:25, modo HYBRID" in built.messages[-1].content
    assert built.tokens <= 200
    assert "turno" in built.dropped
    # Se conservan los turnos mas recientes, no los mas viejos.
    users = [m.content for m in built.messages if m.role == "user"][:-1]
    assert all("pregunta 9" in u or "pregunta 8" in u for u in users)


def test_sistema_y_orden_entran_aunque_no_quepan():
    built = ContextBuilder().build(
        system="x" * 1000,
        user_text="y" * 1000,
        budget_tokens=10,
        history=ConversationHistory(5),
    )
    assert built.over_budget
    assert len(built.messages) == 2
    assert estimate_tokens("abcdefg") == 2


# ---------------------------------------------------------------------------
# Frases
# ---------------------------------------------------------------------------


def test_las_frases_rotan_sin_repetir_la_anterior():
    picker = PhrasePicker(random.Random(7))
    picks = [picker.pick("volume_up") for _ in range(30)]
    assert all(a != b for a, b in itertools.pairwise(picks))
    assert len(set(picks)) >= 3


def test_ninguna_frase_es_servil():
    forbidden = ("¡", "Claro que sí", "Excelente", "Con gusto", "encantado")
    for pool in PHRASES.values():
        for phrase in pool:
            assert not any(word in phrase for word in forbidden), phrase
            assert len(pool) >= 2, "cada grupo necesita variedad"
