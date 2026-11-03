"""Piezas para probar la memoria sin modelos: reloj simulable, extractor con guion.

SimClock            un reloj que se adelanta a mano ("tres dias despues")
ScriptedExtractor   un "modelo local" que devuelve extracciones fijas o falla
make_memory()       un MemorySystem completo sobre una carpeta temporal
assistant()         la cascada con falsos + memoria conectada, como en produccion
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from jarvis.cognition.confirm import PendingConfirmations
from jarvis.config import CognitionConfig, MemoryConfig
from jarvis.memory.extraction import Extraction, ExtractionError
from jarvis.memory.factory import KeywordEmbedder
from jarvis.memory.service import MemorySystem
from jarvis.memory.store import MemoryStore
from tests.cognition_fakes import FakeLocal, Rig, build

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from jarvis.bus.base import EventBus
    from jarvis.memory.episodic import Episode

START = datetime(2026, 11, 2, 15, 0).astimezone()


@dataclass
class SimClock:
    """Hora simulada. `advance(days=3)` es "tres dias despues"."""

    now: datetime = START

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **delta: float) -> None:
        self.now = self.now + timedelta(**delta)


@dataclass
class ScriptedExtractor:
    """Extractor falso: devuelve `result` o lanza si `fail`."""

    result: Extraction = field(default_factory=Extraction)
    fail: bool = False
    name: str = "modelo local"
    calls: list[list[str]] = field(default_factory=list)

    async def extract(self, episodes: Sequence[Episode]) -> Extraction:
        self.calls.append([e.user_text or "" for e in episodes])
        if self.fail:
            raise ExtractionError("Ollama no extrajo: ConnectError")
        return self.result


def make_memory(
    bus: EventBus,
    data_dir: Path,
    clock: SimClock,
    *,
    extractor: Any = None,
    **overrides: Any,
) -> MemorySystem:
    """La memoria real (SQLite + LanceDB) con el embedder por palabras."""
    config = MemoryConfig(data_dir=data_dir, **overrides)
    embedder = KeywordEmbedder()
    store = MemoryStore(data_dir, vector_dim=embedder.dims)
    return MemorySystem(
        bus,
        lambda: config,
        store,
        embedder,
        cognition=CognitionConfig,
        user_name=lambda: "Rhandy",
        clock=clock,
        extractor=extractor or ScriptedExtractor(),
    )


def assistant(
    bus: EventBus,
    data_dir: Path,
    clock: SimClock,
    *,
    local: FakeLocal | None = None,
    cognition: dict[str, Any] | None = None,
    **memory_overrides: Any,
) -> tuple[Rig, MemorySystem]:
    """Cascada con falsos y memoria real, conectadas como en `build_assistant`."""
    memory = make_memory(bus, data_dir, clock, **memory_overrides)
    rig = build(
        bus,
        local=local or FakeLocal(),
        memory=memory,
        history=memory.working,
        confirmations=PendingConfirmations(lambda: 15.0, clock=lambda: clock().timestamp()),
        clock=clock,
        **(cognition or {}),
    )
    memory.attach(rig.orchestrator)
    return rig, memory
