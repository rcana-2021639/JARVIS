"""Grafo de entidades: personas, proyectos, archivos, apps, sitios y lugares.

Conectadas con relaciones tipadas: Maria -trabaja_en-> Proyecto Aurora
-usa_archivo-> informe.docx. Permite razonar sobre conexiones que nunca se
dijeron juntas: desde Maria, a dos saltos, aparece informe.docx.

Vive en dos tablas de SQLite. El recorrido es en anchura, un nivel por
consulta, con profundidad configurable y en ambas direcciones: una relacion
se puede leer desde cualquiera de sus extremos.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rapidfuzz import fuzz, process

from jarvis.contracts.enums import MemoryKind
from jarvis.contracts.memory import MemoryRecord
from jarvis.memory.text import fold, from_epoch, to_epoch

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Callable, Sequence
    from datetime import datetime

    from jarvis.memory.store import MemoryStore

__all__ = ["ENTITY_TYPES", "Entity", "EntityGraph", "Triple", "entity_key"]

ENTITY_TYPES = (
    "persona", "proyecto", "archivo", "app", "sitio", "lugar", "organizacion", "cosa",
)  # fmt: skip
_FUZZY_MIN = 88
_ARTICLE = re.compile(r"^(?:el|la|los|las|un|una|mi|mis|tu|tus|su|sus)\s+")
_NON_WORD = re.compile(r"[^\w.\s-]")


def entity_key(name: str) -> str:
    """Forma canonica de un nombre: "el Proyecto Aurora" -> "proyecto aurora"."""
    key = _NON_WORD.sub(" ", fold(name)).strip()
    key = _ARTICLE.sub("", key)
    return " ".join(key.split())


@dataclass(frozen=True, slots=True)
class Entity:
    """Un nodo del grafo."""

    id: int
    name: str
    key: str
    type: str
    mentions: int
    last_seen_at: datetime


@dataclass(frozen=True, slots=True)
class Triple:
    """Una relacion leida como frase: (Maria, trabaja_en, Proyecto Aurora)."""

    relation_id: int
    source: str
    relation: str
    target: str
    hops: int
    """A cuantos saltos de la entidad de partida esta esta relacion (1 = directa)."""

    @property
    def text(self) -> str:
        """ "Maria trabaja en Proyecto Aurora"."""
        return f"{self.source} {self.relation.replace('_', ' ')} {self.target}"

    def record(self, at: datetime) -> MemoryRecord:
        """Como contrato comun de memoria."""
        return MemoryRecord(
            record_id=f"relation:{self.relation_id}",
            kind=MemoryKind.ENTIDAD,
            content=self.text,
            importance=0.6,
            timestamp=at,
            last_accessed_at=at,
            metadata={"hops": self.hops},
        )


def _entity(row: sqlite3.Row) -> Entity:
    return Entity(
        id=int(row["id"]),
        name=row["name"],
        key=row["key"],
        type=row["type"],
        mentions=int(row["mentions"]),
        last_seen_at=from_epoch(row["last_seen_at"]),
    )


def _relation_name(rel: str) -> str:
    return "_".join(entity_key(rel).replace("-", " ").split()) or "relacionado_con"


class EntityGraph:
    """Entidades y relaciones tipadas en SQLite."""

    def __init__(self, store: MemoryStore, clock: Callable[[], datetime]) -> None:
        """Args: store, el almacenamiento; clock, la hora."""
        self._store = store
        self._clock = clock

    def upsert(self, name: str, type_: str = "cosa") -> Entity:
        """Crea la entidad o anota una mencion mas. Devuelve la entidad.

        Si ya existe con otro tipo y el nuevo es el generico "cosa", se usa la
        existente: "Aurora" dicho de pasada es el mismo Proyecto Aurora.
        """
        key = entity_key(name)
        if not key:
            msg = f"Nombre de entidad vacio: {name!r}"
            raise ValueError(msg)
        kind = type_ if type_ in ENTITY_TYPES else "cosa"
        now = to_epoch(self._clock())
        if kind == "cosa":
            existing = self._store.query("SELECT * FROM entities WHERE key = ? LIMIT 1", (key,))
            if existing:
                kind = existing[0]["type"]
        else:
            # Lo que se conocio como "cosa" y ahora llega con tipo se promueve, no
            # se duplica: "Rhandy" (cosa) y "Rhandy" (persona) son la misma.
            self._store.execute(
                "UPDATE entities SET type = ? WHERE key = ? AND type = 'cosa' "
                "AND NOT EXISTS (SELECT 1 FROM entities WHERE key = ? AND type = ?)",
                (kind, key, key, kind),
            )
        with self._store.transaction() as conn:
            conn.execute(
                "INSERT INTO entities (name, key, type, created_at, last_seen_at) "
                "VALUES (?, ?, ?, ?, ?) ON CONFLICT (key, type) DO UPDATE SET "
                "mentions = mentions + 1, last_seen_at = excluded.last_seen_at",
                (name.strip(), key, kind, now, now),
            )
            row = conn.execute(
                "SELECT * FROM entities WHERE key = ? AND type = ?", (key, kind)
            ).fetchone()
        return _entity(row)

    def relate(
        self,
        source: str,
        relation: str,
        target: str,
        *,
        source_type: str = "cosa",
        target_type: str = "cosa",
        origin: str = "usuario",
    ) -> Triple:
        """Conecta dos entidades (creandolas si hace falta)."""
        a = self.upsert(source, source_type)
        b = self.upsert(target, target_type)
        rel = _relation_name(relation)
        now = to_epoch(self._clock())
        with self._store.transaction() as conn:
            conn.execute(
                "INSERT INTO relations (src, rel, dst, created_at, source) VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT (src, rel, dst) DO UPDATE SET weight = weight + 1",
                (a.id, rel, b.id, now, origin),
            )
            row = conn.execute(
                "SELECT id FROM relations WHERE src = ? AND rel = ? AND dst = ?", (a.id, rel, b.id)
            ).fetchone()
        return Triple(int(row["id"]), a.name, rel, b.name, 1)

    def find(self, name: str) -> Entity | None:
        """Una entidad por nombre, tolerando errores de transcripcion."""
        key = entity_key(name)
        if not key:
            return None
        rows = self._store.query(
            "SELECT * FROM entities WHERE key = ? ORDER BY mentions DESC LIMIT 1", (key,)
        )
        if rows:
            return _entity(rows[0])
        everyone = self._store.query("SELECT * FROM entities")
        if not everyone:
            return None
        keys = [row["key"] for row in everyone]
        best = process.extractOne(key, keys, scorer=fuzz.ratio, score_cutoff=_FUZZY_MIN)
        return _entity(everyone[best[2]]) if best is not None else None

    def mentioned_in(self, text: str) -> list[Entity]:
        """Las entidades cuyo nombre aparece en el texto, palabra completa."""
        haystack = f" {entity_key(text)} "
        rows = self._store.query("SELECT * FROM entities ORDER BY length(key) DESC")
        found: list[Entity] = []
        for row in rows:
            key = row["key"]
            if len(key) >= 3 and f" {key} " in haystack:
                found.append(_entity(row))
        return found

    def traverse(self, start: Sequence[int], depth: int) -> list[Triple]:
        """Las relaciones alcanzables desde unas entidades, hasta `depth` saltos.

        Una consulta por nivel: con profundidad 2 son dos consultas indexadas.
        """
        seen_entities = set(start)
        frontier = set(start)
        seen_relations: dict[int, Triple] = {}
        for hop in range(1, depth + 1):
            if not frontier:
                break
            marks = ", ".join("?" for _ in frontier)
            rows = self._store.query(
                "SELECT r.id, r.src, r.dst, r.rel, s.name AS sname, d.name AS dname "
                "FROM relations r JOIN entities s ON s.id = r.src JOIN entities d ON d.id = r.dst "
                f"WHERE r.src IN ({marks}) OR r.dst IN ({marks}) ORDER BY r.weight DESC",
                (*frontier, *frontier),
            )
            next_frontier: set[int] = set()
            for row in rows:
                relation_id = int(row["id"])
                if relation_id not in seen_relations:
                    seen_relations[relation_id] = Triple(
                        relation_id, row["sname"], row["rel"], row["dname"], hop
                    )
                for node in (int(row["src"]), int(row["dst"])):
                    if node not in seen_entities:
                        seen_entities.add(node)
                        next_frontier.add(node)
            frontier = next_frontier
        return sorted(seen_relations.values(), key=lambda t: t.hops)

    def path(self, source: str, target: str, max_depth: int) -> list[Triple] | None:
        """El camino mas corto entre dos entidades, o None si no se conectan."""
        a, b = self.find(source), self.find(target)
        if a is None or b is None:
            return None
        if a.id == b.id:
            return []
        parents: dict[int, tuple[int, Triple]] = {}
        visited = {a.id}
        frontier = [a.id]
        for _ in range(max_depth):
            next_frontier: list[int] = []
            for node in frontier:
                for triple, other in self._edges(node):
                    if other in visited:
                        continue
                    visited.add(other)
                    parents[other] = (node, triple)
                    if other == b.id:
                        steps: list[Triple] = []
                        cursor = other
                        while cursor in parents:
                            cursor, edge = parents[cursor]
                            steps.append(edge)
                        return list(reversed(steps))
                    next_frontier.append(other)
            frontier = next_frontier
        return None

    def _edges(self, node: int) -> list[tuple[Triple, int]]:
        rows = self._store.query(
            "SELECT r.id, r.src, r.dst, r.rel, s.name AS sname, d.name AS dname "
            "FROM relations r JOIN entities s ON s.id = r.src JOIN entities d ON d.id = r.dst "
            "WHERE r.src = ? OR r.dst = ?",
            (node, node),
        )
        return [
            (
                Triple(int(r["id"]), r["sname"], r["rel"], r["dname"], 1),
                int(r["dst"]) if int(r["src"]) == node else int(r["src"]),
            )
            for r in rows
        ]

    def entities(self, type_: str | None = None) -> list[Entity]:
        """Todas las entidades, o las de un tipo, por menciones."""
        if type_:
            rows = self._store.query(
                "SELECT * FROM entities WHERE type = ? ORDER BY mentions DESC, name", (type_,)
            )
        else:
            rows = self._store.query("SELECT * FROM entities ORDER BY type, mentions DESC, name")
        return [_entity(row) for row in rows]

    def relations(self) -> list[Triple]:
        """Todas las relaciones, para el inspector."""
        rows = self._store.query(
            "SELECT r.id, r.rel, s.name AS sname, d.name AS dname FROM relations r "
            "JOIN entities s ON s.id = r.src JOIN entities d ON d.id = r.dst ORDER BY s.name"
        )
        return [Triple(int(r["id"]), r["sname"], r["rel"], r["dname"], 1) for r in rows]

    def counts(self) -> tuple[int, int]:
        """(entidades, relaciones)."""
        entities = self._store.query("SELECT COUNT(*) FROM entities")[0][0]
        relations = self._store.query("SELECT COUNT(*) FROM relations")[0][0]
        return int(entities), int(relations)

    def delete(self, ids: Sequence[int]) -> int:
        """Borra entidades; sus relaciones se van en cascada."""
        with self._store.transaction() as conn:
            return sum(
                conn.execute("DELETE FROM entities WHERE id = ?", (item,)).rowcount for item in ids
            )
