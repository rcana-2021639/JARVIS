"""Memoria de trabajo: los ultimos turnos y el estado actual. Solo RAM.

Es el historial de la conversacion viva (el que ya usaba la cascada en la Fase
2) mas una foto del sistema: en que estado esta el nucleo, cuando fue la ultima
interaccion y, desde la Fase 6, que ventana esta al frente y que suena. Se
vacia al reiniciar, como pide el documento: lo que deba durar pasa a la
episodica.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from jarvis.cognition.context import ConversationHistory
from jarvis.contracts.enums import CoreState, EventType
from jarvis.contracts.system import StateChange
from jarvis.memory.text import fold

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import datetime

    from jarvis.contracts.events import Event

__all__ = ["SystemSnapshot", "WorkingMemory"]


@dataclass(slots=True)
class SystemSnapshot:
    """Lo que esta pasando ahora mismo."""

    state: CoreState = CoreState.DORMIDO
    state_since: datetime | None = None
    last_interaction_at: datetime | None = None
    active_window: str | None = None
    """La ventana al frente. La llena la Fase 6 (control del sistema)."""
    now_playing: str | None = None
    """Lo que suena. La llena la Fase 6."""


class WorkingMemory(ConversationHistory):
    """El historial en RAM de la cascada, con la foto del sistema al lado."""

    def __init__(self, max_turns: Callable[[], int], clock: Callable[[], datetime]) -> None:
        """Args: max_turns, el tope vigente (`memory.working_turns`); clock, la hora."""
        super().__init__(max_turns())
        self._max_turns_source = max_turns
        self._clock = clock
        self.snapshot = SystemSnapshot()

    def resize(self, max_turns: int) -> None:
        """El tope lo manda la seccion de memoria, no la de cognicion."""
        super().resize(self._max_turns_source())

    def observe(self, event: Event) -> None:
        """Actualiza la foto con un evento del bus."""
        if event.type == EventType.CORE_STATE_CHANGED:
            self.snapshot.state = event.payload_as(StateChange).current
            self.snapshot.state_since = self._clock()
        elif event.type == EventType.COGNITION_COMPLETED:
            self.snapshot.last_interaction_at = self._clock()

    def forget(self, words: list[str]) -> int:
        """Saca de la RAM los turnos que mencionan alguna de las palabras.

        Al olvidar algo, no basta con borrarlo del disco: si sigue en los
        ultimos turnos, el modelo lo seguiria viendo en esta conversacion.
        """
        if not words:
            return 0
        kept = [
            turn
            for turn in self._turns
            if not any(word in fold(f"{turn.user} {turn.assistant}") for word in words)
        ]
        removed = len(self._turns) - len(kept)
        self._turns.clear()
        self._turns.extend(kept)
        summaries = [s for s in self._summaries if not any(word in fold(s) for word in words)]
        removed += len(self._summaries) - len(summaries)
        self._summaries.clear()
        self._summaries.extend(summaries)
        return removed

    def state_line(self) -> str:
        """El estado en una linea, para el inspector y el contexto."""
        parts = [f"estado {self.snapshot.state.value}"]
        if self.snapshot.active_window:
            parts.append(f"ventana: {self.snapshot.active_window}")
        if self.snapshot.now_playing:
            parts.append(f"sonando: {self.snapshot.now_playing}")
        return ", ".join(parts)
