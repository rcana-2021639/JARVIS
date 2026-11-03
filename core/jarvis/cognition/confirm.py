"""Confirmaciones habladas: "¿Lo olvido? Di «Jarvis, sí»".

Una accion delicada no se ejecuta al pedirla: se deja pendiente, Jarvis
pregunta, y solo corre si la siguiente orden es "si" antes del plazo. "No",
"cancela" o el plazo vencido la descartan sin tocar nada.

DECISION: una sola accion pendiente a la vez. Pedir otra reemplaza a la
anterior, porque "si" tiene que referirse sin ambiguedad a la ultima pregunta
que se oyo. El Guardian de la Fase 8 generaliza esto a toda accion AMBAR.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from jarvis.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

__all__ = ["PendingAction", "PendingConfirmations"]

_log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class PendingAction:
    """Algo que espera un "si"."""

    description: str
    """Que se va a hacer, para el registro y para "¿que estaba pendiente?"."""
    run: Callable[[], Awaitable[str]]
    """La accion. Devuelve lo que Jarvis dice al terminarla."""
    expires_at: float


class PendingConfirmations:
    """La accion que espera confirmacion, con su plazo."""

    def __init__(
        self,
        timeout_s: Callable[[], float],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Args: timeout_s, el plazo vigente; clock, reloj monotono (simulable)."""
        self._timeout_s = timeout_s
        self._clock = clock
        self._pending: PendingAction | None = None

    def ask(self, description: str, run: Callable[[], Awaitable[str]]) -> None:
        """Deja una accion esperando confirmacion; reemplaza la anterior."""
        if self._pending is not None:
            _log.info("confirm.replaced", previous=self._pending.description)
        self._pending = PendingAction(description, run, self._clock() + self._timeout_s())
        _log.info("confirm.asked", action=description)

    @property
    def pending(self) -> PendingAction | None:
        """La accion pendiente, si sigue en plazo."""
        if self._pending is not None and self._clock() > self._pending.expires_at:
            _log.info("confirm.expired", action=self._pending.description)
            self._pending = None
        return self._pending

    async def confirm(self) -> str | None:
        """Ejecuta lo pendiente. None si no habia nada (o vencio)."""
        action = self.pending
        if action is None:
            return None
        self._pending = None
        _log.info("confirm.accepted", action=action.description)
        return await action.run()

    def deny(self) -> PendingAction | None:
        """Descarta lo pendiente. Devuelve lo que se descarto."""
        action = self.pending
        self._pending = None
        if action is not None:
            _log.info("confirm.denied", action=action.description)
        return action
