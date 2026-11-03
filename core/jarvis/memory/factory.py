"""Arma la memoria real desde la configuracion, y la cascada conectada a ella.

Es el unico lugar que sabe que piezas reales hay: SQLite y LanceDB en
`memory.data_dir`, MiniLM en ONNX y Ollama para consolidar. Las pruebas no
pasan por aqui.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import numpy as np

from jarvis.cognition.embeddings import EmbeddingModelMissingError, OnnxEmbedder
from jarvis.cognition.factory import build_cascade
from jarvis.cognition.text import normalize
from jarvis.logging import get_logger
from jarvis.memory.service import MemorySystem
from jarvis.memory.store import MemoryStore

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from datetime import datetime

    from numpy.typing import NDArray

    from jarvis.bus.base import EventBus
    from jarvis.cognition.embeddings import Embedder
    from jarvis.cognition.orchestrator import CognitiveOrchestrator
    from jarvis.config import JarvisConfig

__all__ = ["KeywordEmbedder", "build_assistant", "build_memory", "build_memory_embedder"]

_log = get_logger(__name__)
_CLASSIFIER_TOKENS = 32


class KeywordEmbedder:
    """Respaldo sin modelo: bolsa de palabras con hash.

    Solo se usa si falta MiniLM. Encuentra hechos que comparten palabras, no
    sinonimos: la memoria funciona, pero peor, y se avisa al arrancar.
    """

    identity = "keyword-hash-384"
    dims = 384

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Una fila normalizada por texto."""
        out = np.zeros((len(texts), self.dims), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in normalize(text).split():
                bucket = int(hashlib.md5(word[:6].encode()).hexdigest(), 16) % self.dims
                out[row, bucket] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return (out / np.maximum(norms, 1e-9)).astype(np.float32)


def build_memory_embedder(config: JarvisConfig) -> Embedder:
    """MiniLM con el truncado de la memoria; si falta, el respaldo por palabras."""
    memory = config.memory
    try:
        return OnnxEmbedder(
            config.resolved(memory.embedding_model_dir), memory.embedding_max_tokens
        )
    except EmbeddingModelMissingError as error:
        _log.warning("memory.embeddings_unavailable", detail=str(error))
        return KeywordEmbedder()


def build_memory(
    bus: EventBus,
    config: Callable[[], JarvisConfig],
    *,
    embedder: Embedder | None = None,
    clock: Callable[[], datetime] | None = None,
) -> MemorySystem:
    """El sistema de memoria sobre `memory.data_dir`."""
    root = config()
    model = embedder or build_memory_embedder(root)
    dimensions = int(model.embed(["dimension"]).shape[1])
    store = MemoryStore(root.resolved(root.memory.data_dir), vector_dim=dimensions)
    return MemorySystem(
        bus,
        lambda: config().memory,
        store,
        model,
        cognition=lambda: config().cognition,
        user_name=lambda: config().user_name,
        clock=clock,
    )


def build_assistant(
    bus: EventBus, config: Callable[[], JarvisConfig]
) -> tuple[CognitiveOrchestrator, MemorySystem]:
    """La cascada con memoria: lo que usan `jarvis_chat.py` y `voice_live.py`.

    El clasificador y la memoria comparten la sesion de MiniLM (118 MB) con
    truncados distintos: 32 tokens para ordenes, 128 para hechos.
    """
    memory = build_memory(bus, config)
    classifier_embedder = (
        memory.embedder.with_max_tokens(_CLASSIFIER_TOKENS)
        if isinstance(memory.embedder, OnnxEmbedder)
        else None
    )
    orchestrator = build_cascade(
        bus, config, memory=memory, history=memory.working, embedder=classifier_embedder
    )
    memory.attach(orchestrator)
    return orchestrator, memory
