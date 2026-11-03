"""Recuperacion hibrida: cuatro busquedas en paralelo, fusionadas con RRF.

    vectorial   hechos parecidos por significado (MiniLM + LanceDB)
    BM25        episodios y hechos por palabra clave (FTS5)
    grafo       relaciones desde las entidades mencionadas, a N saltos
    recencia    lo mas reciente, con peso que decae en el tiempo

Las cuatro corren con `asyncio.gather` en hilos aparte. Se fusionan con
Reciprocal Rank Fusion (cada lista aporta 1 / (k + posicion)), se reordena por
el peso efectivo de cada recuerdo y se corta al presupuesto de tokens.

DECISION: la recencia solo REORDENA, no aporta candidatos propios. Si aportara,
cada consulta arrastraria al contexto lo ultimo que se dijo, venga o no al
caso, y eso ya esta en la memoria de trabajo. Un recuerdo tiene que aparecer
en al menos una busqueda de relevancia (vector, BM25 o grafo) para entrar.

DECISION: sin re-ordenamiento con un modelo cruzado. El documento lo deja
opcional "si la latencia lo permite", y en esta CPU un cross-encoder sobre 20
candidatos son 200-400 ms: cuatro veces el presupuesto entero.
"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jarvis.cognition.context import estimate_tokens
from jarvis.logging import get_logger
from jarvis.memory.text import clip

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import datetime

    from jarvis.config import MemoryConfig
    from jarvis.contracts.memory import MemoryRecord
    from jarvis.memory.episodic import Episode, EpisodicMemory
    from jarvis.memory.graph import EntityGraph
    from jarvis.memory.semantic import SemanticMemory

__all__ = ["HybridRetriever", "Retrieval", "RetrievedItem"]

_log = get_logger(__name__)
_EPISODE_CHARS = 160


@dataclass(slots=True)
class RetrievedItem:
    """Un candidato con lo que se sabe de el."""

    key: str
    text: str
    record: MemoryRecord
    at: datetime
    weight: float
    rrf: float = 0.0
    score: float = 0.0
    sources: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Retrieval:
    """El resultado de una recuperacion."""

    items: list[RetrievedItem]
    elapsed_ms: float
    timings_ms: dict[str, float]
    tokens: int
    timed_out: list[str] = field(default_factory=list)
    """Busquedas que no llegaron a tiempo y se dejaron fuera."""

    @property
    def texts(self) -> list[str]:
        """Lo que entra al contexto del modelo, en orden de relevancia."""
        return [item.text for item in self.items]


def _episode_text(episode: Episode) -> str:
    day = episode.at.astimezone().strftime("%Y-%m-%d")
    return f"[{day}] {clip(episode.content.replace(chr(10), ' / '), _EPISODE_CHARS)}"


class HybridRetriever:
    """Las cuatro busquedas, la fusion y el presupuesto."""

    def __init__(
        self,
        semantic: SemanticMemory,
        episodic: EpisodicMemory,
        graph: EntityGraph,
        config: Callable[[], MemoryConfig],
        clock: Callable[[], datetime],
    ) -> None:
        """Conecta las capas que se consultan."""
        self._semantic = semantic
        self._episodic = episodic
        self._graph = graph
        self._config = config
        self._clock = clock

    async def retrieve(self, query: str, *, budget_tokens: int | None = None) -> Retrieval:
        """Recupera lo relevante para una consulta, dentro del presupuesto."""
        cfg = self._config()
        budget = budget_tokens if budget_tokens is not None else cfg.retrieval_budget_tokens
        started = time.perf_counter()
        timings: dict[str, float] = {}

        async def timed(name: str, fn: Callable[[], list[RetrievedItem]]) -> list[RetrievedItem]:
            t0 = time.perf_counter()
            result = await asyncio.to_thread(fn)
            timings[name] = round((time.perf_counter() - t0) * 1000.0, 2)
            return result

        k = cfg.retrieval_candidates
        searches = {
            "vector": asyncio.ensure_future(timed("vector", lambda: self._vector(query, k))),
            "bm25": asyncio.ensure_future(timed("bm25", lambda: self._bm25(query, k))),
            "graph": asyncio.ensure_future(timed("graph", lambda: self._graph_hits(query, k))),
        }
        # DECISION: si una busqueda se pasa del objetivo duro, se fusiona lo que
        # llego. Un recuerdo a tiempo vale mas que el recuerdo perfecto tarde.
        done, pending = await asyncio.wait(
            searches.values(), timeout=cfg.retrieval_timeout_ms / 1000.0
        )
        timed_out = [name for name, task in searches.items() if task in pending]
        for task in pending:
            task.add_done_callback(_swallow)
        lists = {name: task.result() for name, task in searches.items() if task in done}

        items = self._fuse(lists, cfg)
        chosen: list[RetrievedItem] = []
        used = 0
        for item in items:
            cost = estimate_tokens(item.text) + 2
            if used + cost > budget:
                continue
            chosen.append(item)
            used += cost
        touched = [item.key.split(":", 1)[1] for item in chosen if item.key.startswith("fact:")]
        if touched:
            # Lo que se uso deja de desvanecerse. Fuera del camino critico.
            asyncio.get_running_loop().run_in_executor(None, self._touch, touched)
        elapsed = (time.perf_counter() - started) * 1000.0
        if timed_out:
            _log.warning("memory.retrieval_partial", missing=timed_out, elapsed_ms=round(elapsed))
        return Retrieval(chosen, round(elapsed, 2), timings, used, timed_out)

    def _touch(self, ids: list[str]) -> None:
        try:
            self._semantic.touch(ids)
        except Exception:
            # No afecta la respuesta: solo el decaimiento de esos hechos.
            _log.exception("memory.touch_failed")

    # ------------------------------------------------------------------
    # Las busquedas (corren en hilos)

    def _vector(self, query: str, limit: int) -> list[RetrievedItem]:
        minimum = self._config().vector_min_similarity
        vector = self._semantic.embed([query])[0]
        return [
            RetrievedItem(f"fact:{fact.id}", fact.content, fact.record(), fact.created_at,
                          fact.weight)
            for fact, similarity in self._semantic.search_vector(vector, limit)
            if similarity >= minimum
        ]  # fmt: skip

    def _bm25(self, query: str, limit: int) -> list[RetrievedItem]:
        scored: list[tuple[float, RetrievedItem]] = []
        for fact, score in self._semantic.search_text(query, limit):
            scored.append(
                (score, RetrievedItem(f"fact:{fact.id}", fact.content, fact.record(),
                                      fact.created_at, fact.weight))
            )  # fmt: skip
        for episode, score in self._episodic.search(query, limit):
            record = episode.record()
            scored.append(
                (score, RetrievedItem(f"episode:{episode.id}", _episode_text(episode), record,
                                      episode.at, record.effective_weight))
            )  # fmt: skip
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in scored[:limit]]

    def _graph_hits(self, query: str, limit: int) -> list[RetrievedItem]:
        entities = self._graph.mentioned_in(query)
        if not entities:
            return []
        now = self._clock()
        triples = self._graph.traverse([e.id for e in entities], self._config().graph_depth)
        return [
            RetrievedItem(f"relation:{t.relation_id}", t.text, t.record(now), now,
                          1.0 / t.hops)
            for t in triples[:limit]
        ]  # fmt: skip

    # ------------------------------------------------------------------

    def _fuse(
        self, lists: dict[str, list[RetrievedItem]], cfg: MemoryConfig
    ) -> list[RetrievedItem]:
        merged: dict[str, RetrievedItem] = {}
        for name, items in lists.items():
            for rank, item in enumerate(items, start=1):
                current = merged.setdefault(item.key, item)
                current.rrf += 1.0 / (cfg.rrf_k + rank)
                current.sources.append(name)
        if not merged:
            return []
        # La recencia como cuarta lista: ordena a los candidatos por antiguedad
        # y suma su aporte de RRF con peso temporal (vida media configurable).
        now = self._clock()
        by_age = sorted(merged.values(), key=lambda it: it.at, reverse=True)
        half_life_s = cfg.recency_half_life_hours * 3600.0
        for rank, item in enumerate(by_age, start=1):
            age_s = max(0.0, (now - item.at).total_seconds())
            freshness = math.pow(0.5, age_s / half_life_s)
            item.rrf += freshness / (cfg.rrf_k + rank)
            item.sources.append("recencia")
        for item in merged.values():
            # Re-orden por relevancia real: lo desvanecido baja, lo importante sube.
            item.score = item.rrf * (0.5 + 0.5 * item.weight)
        return sorted(merged.values(), key=lambda it: it.score, reverse=True)


def _swallow(task: asyncio.Future[list[RetrievedItem]]) -> None:
    """Una busqueda tardia termina sola; su resultado (o error) ya no importa."""
    if not task.cancelled() and task.exception() is not None:
        _log.warning("memory.late_search_failed", error=str(task.exception()))
