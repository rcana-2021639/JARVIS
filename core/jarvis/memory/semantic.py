"""Memoria semantica: hechos destilados sobre el usuario y su mundo.

Preferencias, personas, proyectos, rutinas, cosas que odia. Cada hecho tiene
importancia y decaimiento: lo que no se usa pierde peso con una vida media
configurable, salvo lo permanente (importancia 1.0). Se recupera por
significado (vectores en LanceDB) y por palabras (BM25 en SQLite).

Un hecho nuevo puede ser tres cosas:

    nuevo        no se parece a nada guardado
    duplicado    casi identico a uno guardado: se refuerza el viejo
    actualizado  mismo sujeto y verbo, otro valor ("tu hermana se llama
                 Lucia" despues de "...se llama Ana"): el viejo queda obsoleto
                 y apunta al nuevo, no se borra
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

from jarvis.contracts.base import new_id
from jarvis.contracts.enums import MemoryKind
from jarvis.contracts.memory import MemoryRecord
from jarvis.logging import get_logger
from jarvis.memory.store import dumps
from jarvis.memory.text import fact_key, from_epoch, fts_query, sentence, to_epoch

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Callable, Sequence
    from datetime import datetime

    import numpy as np
    from numpy.typing import NDArray

    from jarvis.cognition.embeddings import Embedder
    from jarvis.config import MemoryConfig
    from jarvis.memory.store import MemoryStore

__all__ = ["Fact", "FactWrite", "SemanticMemory"]

_log = get_logger(__name__)
_EMBEDDER_KEY = "embedder_identity"
FactStatus = Literal["activo", "obsoleto"]


@dataclass(frozen=True, slots=True)
class Fact:
    """Un hecho que Jarvis sabe."""

    id: str
    content: str
    importance: float
    decay: float
    created_at: datetime
    last_accessed_at: datetime
    source: str
    status: FactStatus = "activo"
    superseded_by: str | None = None
    fact_key: str = ""

    def record(self) -> MemoryRecord:
        """Como contrato comun de memoria."""
        return MemoryRecord(
            record_id=f"fact:{self.id}",
            kind=MemoryKind.SEMANTICA,
            content=self.content,
            importance=self.importance,
            decay=self.decay,
            timestamp=self.created_at,
            last_accessed_at=self.last_accessed_at,
            metadata={"source": self.source},
        )

    @property
    def weight(self) -> float:
        """Importancia menos lo que se desvanecio; lo permanente pesa 1."""
        return self.record().effective_weight


@dataclass(frozen=True, slots=True)
class FactWrite:
    """Que paso al guardar un hecho."""

    fact: Fact
    outcome: Literal["nuevo", "duplicado", "actualizado"]
    replaced: tuple[Fact, ...] = ()


def _row(row: sqlite3.Row) -> Fact:
    return Fact(
        id=row["id"],
        content=row["content"],
        importance=float(row["importance"]),
        decay=float(row["decay"]),
        created_at=from_epoch(row["created_at"]),
        last_accessed_at=from_epoch(row["last_accessed_at"]),
        source=row["source"],
        status=row["status"],
        superseded_by=row["superseded_by"],
        fact_key=row["fact_key"],
    )


class SemanticMemory:
    """Hechos con importancia y decaimiento, buscables por significado."""

    def __init__(
        self,
        store: MemoryStore,
        embedder: Embedder,
        config: Callable[[], MemoryConfig],
        clock: Callable[[], datetime],
    ) -> None:
        """Conecta la capa y reconstruye el indice si cambio el modelo."""
        self._store = store
        self._embedder = embedder
        self._config = config
        self._clock = clock
        self._ensure_index()

    def _ensure_index(self) -> None:
        active = self.count()
        same_model = self._store.get_meta(_EMBEDDER_KEY) == self._embedder.identity
        if same_model and self._store.vectors.count() == active:
            return
        _log.info("memory.reindex", facts=active, same_model=same_model)
        self._store.vectors.clear()
        facts = self.active()
        if facts:
            self._store.vectors.add([f.id for f in facts], self.embed([f.content for f in facts]))
        self._store.set_meta(_EMBEDDER_KEY, self._embedder.identity)

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Vectores normalizados de unos textos."""
        return self._embedder.embed(list(texts))

    # ------------------------------------------------------------------
    # Escritura

    def add(
        self,
        content: str,
        *,
        importance: float,
        source: str,
        metadata: dict[str, Any] | None = None,
    ) -> FactWrite:
        """Guarda un hecho, o refuerza/actualiza el que ya existia."""
        cfg = self._config()
        text = sentence(content)
        now = self._clock()
        vector = self.embed([text])[0]
        nearest = self._store.vectors.search(vector, 1)
        if nearest and nearest[0][1] >= cfg.duplicate_similarity:
            existing = self.get([nearest[0][0]])
            if existing:
                old = existing[0]
                stronger = max(old.importance, importance)
                self._store.execute(
                    "UPDATE facts SET importance = ?, decay = 0, last_accessed_at = ? WHERE id = ?",
                    (stronger, to_epoch(now), old.id),
                )
                return FactWrite(
                    replace(old, importance=stronger, decay=0.0, last_accessed_at=now),
                    "duplicado",
                )

        key = fact_key(text)
        fact = Fact(
            id=new_id(),
            content=text,
            importance=importance,
            decay=0.0,
            created_at=now,
            last_accessed_at=now,
            source=source,
            fact_key=key,
        )
        replaced = self._active_with_key(key) if key else []
        with self._store.transaction() as conn:
            conn.execute(
                "INSERT INTO facts (id, content, fact_key, importance, decay, created_at, "
                "last_accessed_at, source, metadata) VALUES (?, ?, ?, ?, 0, ?, ?, ?, ?)",
                (
                    fact.id, text, key, importance, to_epoch(now), to_epoch(now), source,
                    dumps(metadata or {}),
                ),
            )  # fmt: skip
            for old in replaced:
                conn.execute(
                    "UPDATE facts SET status = 'obsoleto', superseded_by = ? WHERE id = ?",
                    (fact.id, old.id),
                )
        self._store.vectors.delete([old.id for old in replaced])
        self._store.vectors.add([fact.id], vector[None, :])
        return FactWrite(fact, "actualizado" if replaced else "nuevo", tuple(replaced))

    def delete(self, ids: Sequence[str]) -> int:
        """Borra hechos de verdad (olvido explicito), de SQLite y de LanceDB."""
        if not ids:
            return 0
        with self._store.transaction() as conn:
            removed = sum(
                conn.execute("DELETE FROM facts WHERE id = ?", (item,)).rowcount for item in ids
            )
        self._store.vectors.delete(list(ids))
        return removed

    def touch(self, ids: Sequence[str]) -> None:
        """Se acaban de usar: el decaimiento vuelve a contar desde ahora."""
        if not ids:
            return
        now = to_epoch(self._clock())
        with self._store.transaction() as conn:
            conn.executemany(
                "UPDATE facts SET last_accessed_at = ?, decay = 0 WHERE id = ?",
                [(now, item) for item in ids],
            )

    def apply_decay(self) -> int:
        """Recalcula el decaimiento por desuso. Devuelve cuantos hechos cambiaron."""
        half_life = self._config().decay_half_life_days
        now = self._clock()
        changes: list[tuple[float, str]] = []
        for fact in self.active():
            decay = fact.record().decayed_at(now, half_life)
            if abs(decay - fact.decay) >= 0.01:
                changes.append((round(decay, 4), fact.id))
        if changes:
            with self._store.transaction() as conn:
                conn.executemany("UPDATE facts SET decay = ? WHERE id = ?", changes)
        return len(changes)

    # ------------------------------------------------------------------
    # Lectura

    def _active_with_key(self, key: str) -> list[Fact]:
        rows = self._store.query(
            "SELECT * FROM facts WHERE fact_key = ? AND status = 'activo'", (key,)
        )
        return [_row(row) for row in rows]

    def get(self, ids: Sequence[str]) -> list[Fact]:
        """Hechos por id, en el mismo orden."""
        if not ids:
            return []
        marks = ", ".join("?" for _ in ids)
        rows = self._store.query(f"SELECT * FROM facts WHERE id IN ({marks})", tuple(ids))
        by_id = {row["id"]: _row(row) for row in rows}
        return [by_id[item] for item in ids if item in by_id]

    def count(self, status: FactStatus = "activo") -> int:
        """Cuantos hechos hay en un estado."""
        rows = self._store.query("SELECT COUNT(*) FROM facts WHERE status = ?", (status,))
        return int(rows[0][0])

    def active(self, limit: int | None = None) -> list[Fact]:
        """Los hechos vigentes, por importancia efectiva."""
        rows = self._store.query(
            "SELECT * FROM facts WHERE status = 'activo' "
            "ORDER BY importance * (1 - decay) DESC, created_at DESC LIMIT ?",
            (limit if limit is not None else -1,),
        )
        return [_row(row) for row in rows]

    def history(self, fact_id: str) -> list[Fact]:
        """Las versiones viejas que este hecho reemplazo."""
        rows = self._store.query("SELECT * FROM facts WHERE superseded_by = ?", (fact_id,))
        return [_row(row) for row in rows]

    def search_vector(self, vector: NDArray[np.float32], limit: int) -> list[tuple[Fact, float]]:
        """Los hechos mas parecidos por significado, con su coseno."""
        hits = self._store.vectors.search(vector, limit)
        facts = {f.id: f for f in self.get([item for item, _ in hits])}
        return [(facts[item], score) for item, score in hits if item in facts]

    def search_text(self, query: str, limit: int) -> list[tuple[Fact, float]]:
        """BM25 sobre los hechos vigentes."""
        expression = fts_query(query)
        if expression is None:
            return []
        rows = self._store.query(
            "SELECT f.*, bm25(facts_fts) AS score FROM facts_fts "
            "JOIN facts f ON f.rowid = facts_fts.rowid "
            "WHERE facts_fts MATCH ? AND f.status = 'activo' ORDER BY score LIMIT ?",
            (expression, limit),
        )
        return [(_row(row), -float(row["score"])) for row in rows]

    def recent(self, limit: int) -> list[Fact]:
        """Los hechos tocados o aprendidos hace menos, primero."""
        rows = self._store.query(
            "SELECT * FROM facts WHERE status = 'activo' "
            "ORDER BY MAX(created_at, last_accessed_at) DESC LIMIT ?",
            (limit,),
        )
        return [_row(row) for row in rows]

    def matching(self, topic: str) -> list[Fact]:
        """Los hechos que "olvida lo de X" deberia abarcar: por significado o por palabra."""
        cfg = self._config()
        vector = self.embed([topic])[0]
        found: dict[str, Fact] = {}
        for fact, score in self.search_vector(vector, cfg.retrieval_candidates):
            if score >= cfg.forget_min_similarity:
                found[fact.id] = fact
        for fact, _ in self.search_text(topic, cfg.retrieval_candidates):
            found.setdefault(fact.id, fact)
        return list(found.values())

    def metadata(self, fact_id: str) -> dict[str, Any]:
        """Los datos extra de un hecho."""
        rows = self._store.query("SELECT metadata FROM facts WHERE id = ?", (fact_id,))
        return dict(json.loads(rows[0]["metadata"])) if rows else {}
