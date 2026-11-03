"""Memoria procedimental: secuencias de acciones que funcionaron.

Cada vez que una orden se resuelve con herramientas, se anota la secuencia
(que herramientas, con que argumentos) bajo la orden que la disparo. Si la
misma orden produce la misma secuencia `routine_promotion_threshold` veces con
exito, se promueve a rutina: desde ese momento la orden va por la ruta rapida
y ejecuta los pasos sin pasar por ningun modelo. El sistema se acelera solo.

Ademas la consolidacion puede PROPONER rutinas (ordenes que se repiten sin
herramientas, como "que clima hace" cada manana). Esas quedan como
`propuesta` y no se ejecutan solas.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from jarvis.cognition.classifier import strip_fillers
from jarvis.cognition.text import normalize
from jarvis.contracts.base import new_id
from jarvis.memory.store import dumps
from jarvis.memory.text import from_epoch, to_epoch

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Callable, Sequence
    from datetime import datetime

    from jarvis.memory.store import MemoryStore

__all__ = ["ActionPattern", "ProceduralMemory", "Routine", "Step", "routine_trigger"]

RoutineStatus = Literal["activa", "propuesta", "retirada"]


def routine_trigger(text: str) -> str:
    """La forma con la que se compara una orden: la misma que ve el clasificador."""
    return strip_fillers(normalize(text))


@dataclass(frozen=True, slots=True)
class Step:
    """Una llamada a herramienta dentro de una rutina."""

    tool: str
    """`servidor.nombre`."""
    arguments: dict[str, Any] = field(default_factory=dict)

    def as_json(self) -> dict[str, Any]:
        """Forma serializable y estable."""
        return {"tool": self.tool, "arguments": self.arguments}


@dataclass(frozen=True, slots=True)
class Routine:
    """Una secuencia promovida (o propuesta) para ejecutarse por ruta rapida."""

    id: str
    trigger: str
    steps: tuple[Step, ...]
    status: RoutineStatus
    origin: str
    note: str
    uses: int
    created_at: datetime

    @property
    def intent_name(self) -> str:
        """La intencion dinamica con la que el clasificador la reconoce."""
        return f"routine.{self.id[:8]}"


@dataclass(frozen=True, slots=True)
class ActionPattern:
    """Una secuencia observada y cuantas veces funciono."""

    trigger: str
    steps: tuple[Step, ...]
    successes: int
    failures: int


def _steps(raw: str) -> tuple[Step, ...]:
    return tuple(Step(item["tool"], item.get("arguments", {})) for item in json.loads(raw))


def _routine(row: sqlite3.Row) -> Routine:
    return Routine(
        id=row["id"],
        trigger=row["trigger"],
        steps=_steps(row["steps"]),
        status=row["status"],
        origin=row["origin"],
        note=row["note"],
        uses=int(row["uses"]),
        created_at=from_epoch(row["created_at"]),
    )


class ProceduralMemory:
    """Patrones de acciones y las rutinas que salen de ellos."""

    def __init__(
        self,
        store: MemoryStore,
        clock: Callable[[], datetime],
        threshold: Callable[[], int],
    ) -> None:
        """Args: store; clock; threshold, repeticiones exitosas para promover."""
        self._store = store
        self._clock = clock
        self._threshold = threshold

    def observe(self, text: str, steps: Sequence[Step], *, success: bool) -> Routine | None:
        """Anota una secuencia ejecutada. Devuelve la rutina si con esta se promovio."""
        trigger = routine_trigger(text)
        if not trigger or not steps:
            return None
        payload = dumps([step.as_json() for step in steps])
        now = to_epoch(self._clock())
        ok, failed = (1, 0) if success else (0, 1)
        with self._store.transaction() as conn:
            conn.execute(
                "INSERT INTO action_patterns (trigger, signature, steps, successes, failures, "
                "first_seen, last_seen) VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT (trigger, signature) DO UPDATE SET "
                "successes = successes + excluded.successes, "
                "failures = failures + excluded.failures, last_seen = excluded.last_seen",
                (trigger, payload, payload, ok, failed, now, now),
            )
            row = conn.execute(
                "SELECT successes FROM action_patterns WHERE trigger = ? AND signature = ?",
                (trigger, payload),
            ).fetchone()
        if not success or int(row["successes"]) < self._threshold():
            return None
        existing = self.by_trigger(trigger)
        if existing is not None and existing.status == "activa":
            return None
        return self._save(trigger, payload, "activa", "repeticion", "", existing)

    def propose(self, text: str, steps: Sequence[Step], note: str) -> Routine | None:
        """Deja una rutina propuesta (no se ejecuta sola). None si ya habia una."""
        trigger = routine_trigger(text)
        if not trigger or self.by_trigger(trigger) is not None:
            return None
        payload = dumps([step.as_json() for step in steps])
        return self._save(trigger, payload, "propuesta", "consolidacion", note, None)

    def _save(
        self,
        trigger: str,
        payload: str,
        status: RoutineStatus,
        origin: str,
        note: str,
        existing: Routine | None,
    ) -> Routine:
        routine_id = existing.id if existing is not None else new_id()
        self._store.execute(
            "INSERT INTO routines (id, trigger, steps, status, origin, note, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT (trigger) DO UPDATE SET "
            "steps = excluded.steps, status = excluded.status, origin = excluded.origin",
            (routine_id, trigger, payload, status, origin, note, to_epoch(self._clock())),
        )
        routine = self.by_trigger(trigger)
        assert routine is not None
        return routine

    def by_trigger(self, text: str) -> Routine | None:
        """La rutina de una orden, en cualquier estado."""
        rows = self._store.query(
            "SELECT * FROM routines WHERE trigger = ?", (routine_trigger(text),)
        )
        return _routine(rows[0]) if rows else None

    def routines(self, status: RoutineStatus | None = None) -> list[Routine]:
        """Las rutinas, o solo las de un estado."""
        if status is None:
            rows = self._store.query("SELECT * FROM routines ORDER BY created_at")
        else:
            rows = self._store.query(
                "SELECT * FROM routines WHERE status = ? ORDER BY created_at", (status,)
            )
        return [_routine(row) for row in rows]

    def patterns(self) -> list[ActionPattern]:
        """Todas las secuencias observadas, las mas repetidas primero."""
        rows = self._store.query(
            "SELECT * FROM action_patterns ORDER BY successes DESC, last_seen DESC"
        )
        return [
            ActionPattern(row["trigger"], _steps(row["steps"]), row["successes"], row["failures"])
            for row in rows
        ]

    def record_use(self, routine_id: str, *, success: bool) -> None:
        """Se ejecuto por ruta rapida. Una rutina que falla dos veces seguidas se retira."""
        now = to_epoch(self._clock())
        if success:
            self._store.execute(
                "UPDATE routines SET uses = uses + 1, last_used_at = ?, note = '' WHERE id = ?",
                (now, routine_id),
            )
            return
        # DECISION: un fallo suelto puede ser la red o una app cerrada; dos
        # seguidos quieren decir que el mundo cambio y la rutina ya no sirve.
        rows = self._store.query("SELECT note FROM routines WHERE id = ?", (routine_id,))
        if rows and rows[0]["note"] == "fallo":
            self.retire(routine_id)
        else:
            self._store.execute("UPDATE routines SET note = 'fallo' WHERE id = ?", (routine_id,))

    def retire(self, routine_id: str) -> None:
        """Deja de ejecutarse por ruta rapida."""
        self._store.execute("UPDATE routines SET status = 'retirada' WHERE id = ?", (routine_id,))

    def activate(self, routine_id: str) -> None:
        """Una propuesta aceptada pasa a ejecutarse."""
        self._store.execute(
            "UPDATE routines SET status = 'activa', note = '' WHERE id = ?", (routine_id,)
        )
