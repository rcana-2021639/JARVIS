"""Memoria episodica: todo lo ocurrido, con marca de tiempo. SQLite + FTS5.

Cada intercambio, cada accion ejecutada, cada resultado y cada error. Es lo que
permite "¿que te pedi el martes?". Se busca por palabra clave con BM25 (FTS5) y
por rango de fechas; la consolidacion nocturna la lee para destilar hechos y
comprime lo que pasa de 90 dias.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from jarvis.contracts.base import new_id
from jarvis.contracts.enums import CascadeLevel, MemoryKind
from jarvis.contracts.memory import MemoryRecord
from jarvis.memory.store import dumps
from jarvis.memory.text import clip, from_epoch, fts_query, to_epoch

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Callable, Sequence
    from datetime import datetime

    from jarvis.memory.store import MemoryStore

__all__ = ["Episode", "EpisodeKind", "EpisodicMemory"]

EpisodeKind = Literal["turno", "accion", "error", "resumen"]
_COMPRESSED_LINE_CHARS = 120


@dataclass(frozen=True, slots=True)
class Episode:
    """Algo que paso."""

    id: str
    kind: EpisodeKind
    content: str
    at: datetime
    user_text: str | None = None
    answer: str | None = None
    intent: str | None = None
    level: CascadeLevel | None = None
    success: bool = True
    correlation_id: str | None = None
    importance: float = 0.3
    compressed: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def record(self) -> MemoryRecord:
        """Como contrato comun de memoria, para la recuperacion."""
        return MemoryRecord(
            record_id=f"episode:{self.id}",
            kind=MemoryKind.EPISODICA,
            content=self.content,
            importance=self.importance,
            timestamp=self.at,
            last_accessed_at=self.at,
            metadata={"kind": self.kind, "intent": self.intent},
        )


def _row(row: sqlite3.Row) -> Episode:
    level = row["level"]
    return Episode(
        id=row["id"],
        kind=row["kind"],
        content=row["content"],
        at=from_epoch(row["at"]),
        user_text=row["user_text"],
        answer=row["answer"],
        intent=row["intent"],
        level=CascadeLevel(level) if level is not None else None,
        success=bool(row["success"]),
        correlation_id=row["correlation_id"],
        importance=float(row["importance"]),
        compressed=bool(row["compressed"]),
        metadata=json.loads(row["metadata"]),
    )


class EpisodicMemory:
    """Lo que paso, en orden, consultable por palabras y por fechas."""

    def __init__(self, store: MemoryStore, clock: Callable[[], datetime]) -> None:
        """Args: store, el almacenamiento; clock, la hora (simulable en pruebas)."""
        self._store = store
        self._clock = clock

    def add(
        self,
        kind: EpisodeKind,
        content: str,
        *,
        user_text: str | None = None,
        answer: str | None = None,
        intent: str | None = None,
        level: CascadeLevel | None = None,
        success: bool = True,
        correlation_id: str | None = None,
        importance: float = 0.3,
        at: datetime | None = None,
        metadata: dict[str, Any] | None = None,
        compressed: bool = False,
    ) -> Episode:
        """Guarda un episodio y lo devuelve."""
        episode = Episode(
            id=new_id(),
            kind=kind,
            content=content,
            at=at or self._clock(),
            user_text=user_text,
            answer=answer,
            intent=intent,
            level=level,
            success=success,
            correlation_id=correlation_id,
            importance=importance,
            compressed=compressed,
            metadata=metadata or {},
        )
        self._store.execute(
            "INSERT INTO episodes (id, kind, content, user_text, answer, intent, level, "
            "success, correlation_id, at, importance, compressed, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                episode.id, kind, content, user_text, answer, intent,
                int(level) if level is not None else None, int(success), correlation_id,
                to_epoch(episode.at), importance, int(compressed), dumps(episode.metadata),
            ),
        )  # fmt: skip
        return episode

    def count(self) -> int:
        """Episodios guardados."""
        return int(self._store.query("SELECT COUNT(*) FROM episodes")[0][0])

    def counts_by_kind(self) -> dict[str, int]:
        """Cuantos hay de cada tipo."""
        rows = self._store.query("SELECT kind, COUNT(*) AS n FROM episodes GROUP BY kind")
        return {row["kind"]: int(row["n"]) for row in rows}

    def search(self, query: str, limit: int) -> list[tuple[Episode, float]]:
        """BM25 sobre el contenido. Devuelve (episodio, puntaje), mejor primero."""
        expression = fts_query(query)
        if expression is None:
            return []
        rows = self._store.query(
            "SELECT e.*, bm25(episodes_fts) AS score FROM episodes_fts "
            "JOIN episodes e ON e.rowid = episodes_fts.rowid "
            "WHERE episodes_fts MATCH ? ORDER BY score LIMIT ?",
            (expression, limit),
        )
        # bm25() de SQLite es negativo: mas negativo, mas relevante.
        return [(_row(row), -float(row["score"])) for row in rows]

    def between(
        self,
        start: datetime,
        end: datetime,
        *,
        kinds: Sequence[EpisodeKind] = ("turno", "accion", "resumen"),
        limit: int = 200,
    ) -> list[Episode]:
        """Episodios de un periodo, del mas viejo al mas nuevo."""
        marks = ", ".join("?" for _ in kinds)
        rows = self._store.query(
            f"SELECT * FROM episodes WHERE at >= ? AND at < ? AND kind IN ({marks}) "
            "ORDER BY at LIMIT ?",
            (to_epoch(start), to_epoch(end), *kinds, limit),
        )
        return [_row(row) for row in rows]

    def recent(self, limit: int) -> list[Episode]:
        """Los ultimos episodios, del mas nuevo al mas viejo."""
        rows = self._store.query("SELECT * FROM episodes ORDER BY at DESC LIMIT ?", (limit,))
        return [_row(row) for row in rows]

    def unconsolidated(self, until: datetime) -> list[Episode]:
        """Turnos y acciones que la consolidacion todavia no leyo."""
        rows = self._store.query(
            "SELECT * FROM episodes WHERE consolidated = 0 AND at < ? "
            "AND kind IN ('turno', 'accion') ORDER BY at",
            (to_epoch(until),),
        )
        return [_row(row) for row in rows]

    def mark_consolidated(self, ids: Sequence[str]) -> None:
        """Marca episodios como ya leidos por la consolidacion."""
        with self._store.transaction() as conn:
            conn.executemany(
                "UPDATE episodes SET consolidated = 1 WHERE id = ?", [(item,) for item in ids]
            )

    def get(self, ids: Sequence[str]) -> list[Episode]:
        """Episodios por id."""
        if not ids:
            return []
        marks = ", ".join("?" for _ in ids)
        rows = self._store.query(f"SELECT * FROM episodes WHERE id IN ({marks})", tuple(ids))
        return [_row(row) for row in rows]

    def delete(self, ids: Sequence[str]) -> int:
        """Borra episodios (olvido explicito). Devuelve cuantos."""
        with self._store.transaction() as conn:
            return sum(
                conn.execute("DELETE FROM episodes WHERE id = ?", (item,)).rowcount for item in ids
            )

    def compress_before(self, cutoff: datetime) -> int:
        """Junta los episodios viejos en uno por dia. Devuelve cuantos se comprimieron.

        Cada dia viejo queda como un solo episodio `resumen` con una linea
        recortada por turno: se sigue encontrando por palabras clave y por
        fecha, pero ocupa una fraccion. Los errores viejos se descartan.
        """
        rows = self._store.query(
            "SELECT * FROM episodes WHERE at < ? AND compressed = 0 ORDER BY at",
            (to_epoch(cutoff),),
        )
        if not rows:
            return 0
        by_day: dict[str, list[Episode]] = defaultdict(list)
        for row in rows:
            episode = _row(row)
            by_day[episode.at.astimezone().date().isoformat()].append(episode)
        with self._store.transaction() as conn:
            for day, episodes in by_day.items():
                lines = [
                    clip(e.user_text or e.content, _COMPRESSED_LINE_CHARS)
                    for e in episodes
                    if e.kind != "error"
                ]
                if lines:
                    content = f"Resumen del {day}: " + " | ".join(lines)
                    conn.execute(
                        "INSERT INTO episodes (id, kind, content, at, importance, compressed, "
                        "consolidated, metadata) VALUES (?, 'resumen', ?, ?, ?, 1, 1, ?)",
                        (
                            new_id(), content, to_epoch(episodes[0].at),
                            max(e.importance for e in episodes),
                            dumps({"day": day, "episodes": len(episodes)}),
                        ),
                    )  # fmt: skip
                conn.executemany("DELETE FROM episodes WHERE id = ?", [(e.id,) for e in episodes])
        return len(rows)
