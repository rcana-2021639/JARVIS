"""Clasificador de intencion en dos etapas. Meta: menos de 10 ms.

    a) Patrones: regex del catalogo y, si no encaja ninguno, coincidencia
       difusa con rapidfuzz contra las frases canonicas. Microsegundos.
    b) Embeddings: si (a) no encontro nada claro, MiniLM compara la frase con
       los ejemplos etiquetados de cada intencion. ~6 ms.

Si tampoco (b) se parece a nada, la orden es charla libre para el nivel 2.

El clasificador no decide el nivel: lo sugiere. El enrutador tiene la ultima
palabra y puede escalar por complejidad, privacidad o presupuesto.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from rapidfuzz import fuzz, process

from jarvis.cognition.embeddings import cached_embeddings
from jarvis.cognition.text import normalize
from jarvis.contracts.cognition import Intent
from jarvis.contracts.enums import CascadeLevel
from jarvis.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from numpy.typing import NDArray

    from jarvis.cognition.catalog import IntentSpec
    from jarvis.cognition.embeddings import Embedder
    from jarvis.config import CognitionConfig

__all__ = ["FALLBACK_INTENT", "Classification", "IntentClassifier", "strip_fillers"]

_log = get_logger(__name__)

FALLBACK_INTENT = "chat.free"
_PATTERN_CONFIDENCE = 0.97
_FALLBACK_CONFIDENCE = 0.5

Stage = Literal["pattern", "fuzzy", "embedding", "fallback"]

_LEADING_FILLERS = re.compile(
    r"^(?:(?:oye|oiga|hey|jarvis|yarvis|por favor|porfa|a ver|bueno|ok|okay|eh)\s+)+"
)
_POLITE_ASK = re.compile(
    r"^(?:(?:me )?(?:puedes|podrias|quieres)|quiero que|necesito que|te pido que)\s+"
)
_TRAILING_FILLERS = re.compile(r"\s+(?:por favor|porfa|jarvis|gracias)$")


def strip_fillers(normalized: str) -> str:
    """Quita muletillas y cortesias que no cambian la orden.

    "oye jarvis podrias subir el volumen por favor" -> "subir el volumen".
    Si no queda nada (dijo solo "por favor"), devuelve el texto original.
    """
    text = _LEADING_FILLERS.sub("", normalized)
    text = _POLITE_ASK.sub("", text)
    text = _TRAILING_FILLERS.sub("", text).strip()
    return text or normalized


@dataclass(frozen=True, slots=True)
class Classification:
    """El resultado del clasificador mas como llego a el."""

    intent: Intent
    stage: Stage
    normalized: str
    elapsed_ms: float


class IntentClassifier:
    """Patrones primero, embeddings despues, charla si nada encaja."""

    def __init__(
        self,
        catalog: Sequence[IntentSpec],
        config: Callable[[], CognitionConfig],
        embedder: Embedder | None = None,
        *,
        cache_dir: Path | None = None,
    ) -> None:
        """Prepara las tablas de busqueda.

        Args:
            catalog: las intenciones, en orden de prioridad de patrones.
            config: devuelve la seccion vigente; los umbrales se leen en caliente.
            embedder: el modelo de la etapa (b). None deja solo patrones.
            cache_dir: donde guardar los embeddings de los ejemplos para no
                recalcularlos en cada arranque. None, sin cache.
        """
        self._catalog = tuple(catalog)
        self._config = config
        self._by_name = {item.name: item for item in self._catalog}
        if FALLBACK_INTENT not in self._by_name:
            msg = f"El catalogo debe incluir la intencion comodin {FALLBACK_INTENT}"
            raise ValueError(msg)

        self._phrases: list[str] = []
        self._phrase_owner: list[IntentSpec] = []
        for item in self._catalog:
            for phrase in item.phrases:
                self._phrases.append(normalize(phrase))
                self._phrase_owner.append(item)

        self._embedder = embedder
        self._example_owner: list[int] = []
        texts: list[str] = []
        for index, item in enumerate(self._catalog):
            for text in item.training_texts:
                texts.append(normalize(text))
                self._example_owner.append(index)
        self._example_texts = tuple(texts)
        self._owners = np.array(self._example_owner, dtype=np.int64)
        self._matrix: NDArray[np.float32] | None = None
        if embedder is not None:
            self._matrix = cached_embeddings(embedder, texts, cache_dir)

    @property
    def example_texts(self) -> tuple[str, ...]:
        """Los textos que alimentan la etapa (b), para cachear sus embeddings."""
        return self._example_texts

    @property
    def catalog(self) -> tuple[IntentSpec, ...]:
        """El catalogo en uso."""
        return self._catalog

    def spec(self, name: str) -> IntentSpec | None:
        """La definicion de una intencion por nombre."""
        return self._by_name.get(name)

    def classify(self, text: str) -> Classification:
        """Clasifica una orden. Nunca lanza: lo desconocido es charla."""
        started = time.perf_counter()
        normalized = strip_fillers(normalize(text))
        intent, stage = self._classify(normalized)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        budget_ms = self._config().classifier_timeout_ms
        if elapsed_ms > budget_ms:
            # DECISION: se usa igual. Descartar un resultado ya calculado para
            # devolver "no se" no ahorra nada; pasarse queda registrado.
            _log.warning(
                "intent.slow", elapsed_ms=round(elapsed_ms, 2), budget_ms=budget_ms, stage=stage
            )
        return Classification(intent, stage, normalized, elapsed_ms)

    # ------------------------------------------------------------------

    def _classify(self, text: str) -> tuple[Intent, Stage]:
        if not text:
            return self._make(self._by_name[FALLBACK_INTENT], text, None, 0.0), "fallback"
        cfg = self._config()

        for item in self._catalog:
            for pattern in item.patterns:
                match = pattern.search(text)
                if match is not None:
                    return self._make(item, text, match, _PATTERN_CONFIDENCE), "pattern"

        if self._phrases:
            best = process.extractOne(
                text, self._phrases, scorer=fuzz.ratio, score_cutoff=cfg.fuzzy_threshold
            )
            if best is not None:
                _, score, index = best
                return self._make(self._phrase_owner[index], text, None, score / 100.0), "fuzzy"

        if self._embedder is not None and self._matrix is not None:
            return self._by_embedding(text, cfg), "embedding"

        return self._make(
            self._by_name[FALLBACK_INTENT], text, None, _FALLBACK_CONFIDENCE
        ), "fallback"

    def _by_embedding(self, text: str, cfg: CognitionConfig) -> Intent:
        assert self._embedder is not None and self._matrix is not None
        query = self._embedder.embed([text])[0]
        similarities = self._matrix @ query
        best_example = int(np.argmax(similarities))
        similarity = float(np.clip(similarities[best_example], 0.0, 1.0))
        item = self._catalog[int(self._owners[best_example])]

        if similarity < cfg.embedding_min_similarity:
            return self._make(self._by_name[FALLBACK_INTENT], text, None, _FALLBACK_CONFIDENCE)
        level = item.level
        if level is CascadeLevel.FAST_PATH and similarity < cfg.embedding_fast_similarity:
            # Se parece a una orden fija, pero no lo bastante para ejecutarla a
            # ciegas: que la vea un modelo, con la intencion como pista.
            level = CascadeLevel.LOCAL_LLM
        return self._make(item, text, None, similarity, level=level)

    @staticmethod
    def _make(
        item: IntentSpec,
        text: str,
        match: re.Match[str] | None,
        confidence: float,
        *,
        level: CascadeLevel | None = None,
    ) -> Intent:
        return Intent(
            name=item.name,
            entities=item.entities(text, match),
            confidence=round(min(1.0, max(0.0, confidence)), 4),
            suggested_level=level or item.level,
        )
