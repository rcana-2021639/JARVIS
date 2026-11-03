"""Arma la cascada completa desde la configuracion.

Es el unico lugar que sabe que piezas reales existen: MiniLM, Ollama, la API
de Claude, el llavero y las herramientas de demostracion. Las pruebas no pasan
por aqui: construyen el orquestador con falsos.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from jarvis.cognition.budget import SpendLedger
from jarvis.cognition.catalog import build_catalog
from jarvis.cognition.classifier import IntentClassifier
from jarvis.cognition.cloud_llm import ClaudeClient
from jarvis.cognition.demo_tools import demo_tools
from jarvis.cognition.embeddings import EmbeddingModelMissingError, OnnxEmbedder
from jarvis.cognition.local_llm import OllamaClient
from jarvis.cognition.orchestrator import CognitiveOrchestrator
from jarvis.cognition.router import PrivacySwitch
from jarvis.cognition.tool_loop import ToolRegistry
from jarvis.logging import get_logger
from jarvis.secrets import get_secret

if TYPE_CHECKING:
    from collections.abc import Callable

    from jarvis.bus.base import EventBus
    from jarvis.cognition.embeddings import Embedder
    from jarvis.config import CognitionConfig, JarvisConfig
    from jarvis.contracts.cascade import CostUpdate

__all__ = ["build_cascade", "build_classifier"]

_log = get_logger(__name__)
_SECRET_TTL_S = 60.0
"""Cuanto se recuerda la clave leida del llavero antes de volver a consultarlo."""


def build_classifier(config: Callable[[], JarvisConfig]) -> IntentClassifier:
    """El clasificador con embeddings si el modelo esta; si no, solo patrones.

    Que falte MiniLM no impide arrancar: se avisa y se sigue con patrones, que
    cubren todas las ordenes de forma fija.
    """
    root = config()
    cog = root.cognition
    embedder: Embedder | None = None
    model_dir = root.resolved(cog.intent_model_dir)
    if cog.classifier_engine == "embeddings":
        try:
            embedder = OnnxEmbedder(model_dir)
        except EmbeddingModelMissingError as error:
            _log.warning("intent.embeddings_unavailable", detail=str(error))
    return IntentClassifier(
        build_catalog(), lambda: config().cognition, embedder, cache_dir=model_dir
    )


def _cached_secret(name: str) -> Callable[[], str | None]:
    cache: dict[str, tuple[float, str | None]] = {}

    def read() -> str | None:
        now = time.monotonic()
        hit = cache.get(name)
        if hit is None or now - hit[0] > _SECRET_TTL_S:
            hit = (now, get_secret(name))
            cache[name] = hit
        return hit[1]

    return read


def build_cascade(bus: EventBus, config: Callable[[], JarvisConfig]) -> CognitiveOrchestrator:
    """El orquestador real, con todos sus niveles."""
    root = config()

    def cog() -> CognitionConfig:
        return config().cognition

    privacy = PrivacySwitch(lambda: cog().privacy_mode)
    ledger = SpendLedger(root.resolved(root.cognition.spend_ledger), lambda: cog().daily_budget_usd)
    holder: list[CognitiveOrchestrator] = []

    async def on_cost(update: CostUpdate) -> None:
        if holder:
            await holder[0].on_cost(update)

    cloud = ClaudeClient(
        cog,
        ledger,
        _cached_secret("anthropic_api_key"),
        privacy.network_allowed,
        on_cost,
    )
    tools = ToolRegistry(bus, lambda: cog().tool_timeout_s)
    for tool in demo_tools():
        tools.register(tool)
    orchestrator = CognitiveOrchestrator(
        bus,
        cog,
        build_classifier(config),
        ledger=ledger,
        privacy=privacy,
        local=OllamaClient(cog),
        cloud=cloud,
        tools=tools,
        user_name=lambda: config().user_name,
    )
    holder.append(orchestrator)
    return orchestrator
