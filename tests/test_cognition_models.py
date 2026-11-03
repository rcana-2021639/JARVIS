"""La cascada con sus modelos reales. Se salta sola si faltan.

MiniLM     python scripts/download_cognition_models.py
Ollama     instalado, corriendo y con el modelo de `local_model`
"""

from __future__ import annotations

import httpx2 as httpx
import pytest

from jarvis.cognition.factory import build_cascade, build_classifier
from jarvis.cognition.llm import ChatMessage, GateDecision, LLMRequest
from jarvis.cognition.local_llm import OllamaClient
from jarvis.cognition.prompts import system_prompt
from jarvis.config import load_config
from jarvis.contracts import CascadeLevel

pytestmark = pytest.mark.models

_CONFIG = load_config()
_MINILM = _CONFIG.resolved(_CONFIG.cognition.intent_model_dir) / "model.onnx"


def _ollama_ready() -> bool:
    try:
        tags = httpx.get(f"{_CONFIG.cognition.ollama_url}/api/tags", timeout=1.0).json()
    except (httpx.HTTPError, ValueError):
        return False
    names = {m.get("name") for m in tags.get("models", [])}
    return _CONFIG.cognition.local_model in names


needs_minilm = pytest.mark.skipif(not _MINILM.exists(), reason="falta MiniLM en models/intent")
needs_ollama = pytest.mark.skipif(not _ollama_ready(), reason="Ollama no esta listo")


@needs_minilm
@pytest.mark.parametrize(
    ("text", "intent"),
    [
        ("me podrías poner algo de música", "media.play"),
        ("está muy fuerte", "media.volume_down"),
        ("no se escucha nada", "media.volume_up"),
        ("cuéntame un chiste", "chat.free"),
    ],
)
def test_minilm_entiende_frases_sin_patron(text, intent):
    classifier = build_classifier(lambda: _CONFIG)
    result = classifier.classify(text)
    assert result.intent.name == intent
    assert result.elapsed_ms < 15  # meta 10 ms; holgura para la primera llamada en frio


@needs_ollama
async def test_ollama_contesta_con_esquema_y_confianza():
    client = OllamaClient(lambda: _CONFIG.cognition)
    request = LLMRequest(
        messages=[
            ChatMessage("system", system_prompt("Rhandy", structured=True)),
            ChatMessage("user", "¿Cuál es la capital de Francia?"),
        ],
        gate=lambda early: GateDecision.BUFFER,
    )
    reply = await client.generate(request)
    await client.close()
    assert "París" in reply.text or "Paris" in reply.text
    assert reply.confidence is not None and 0.0 <= reply.confidence <= 1.0
    assert reply.early_confidence is not None


@needs_ollama
async def test_la_cascada_real_resuelve_por_nivel_1_y_2(bus):
    cascade = build_cascade(bus, lambda: _CONFIG)
    fast = await cascade.handle_text("¿Qué hora es?")
    assert fast.handled_by_level is CascadeLevel.FAST_PATH
    talk = await cascade.handle_text("¿Qué opinas del café?")
    assert talk.handled_by_level in {CascadeLevel.LOCAL_LLM, CascadeLevel.CLOUD_LLM}
    assert talk.text
    await cascade.stop()
