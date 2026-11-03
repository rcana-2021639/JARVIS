"""Temporizadores en proceso: el unico "control del sistema" real de la Fase 2.

No necesitan ninguna herramienta del sistema operativo: son tareas de asyncio
que, al vencer, anuncian por el mismo camino que cualquier respuesta. Viven en
RAM; reiniciar el nucleo los pierde (la persistencia es de la Fase 3).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jarvis.contracts.base import new_id
from jarvis.logging import get_logger, user_text

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

__all__ = ["Timer", "TimerService"]

_log = get_logger(__name__)


@dataclass(slots=True, eq=False)
class Timer:
    """Un temporizador corriendo."""

    seconds: float
    label: str | None
    deadline: float
    timer_id: str = field(default_factory=new_id)
    task: asyncio.Task[None] | None = None

    def remaining(self, now: float | None = None) -> float:
        """Segundos que faltan."""
        return max(0.0, self.deadline - (time.monotonic() if now is None else now))


class TimerService:
    """Programa, cancela y anuncia temporizadores."""

    def __init__(self, announce: Callable[[Timer], Awaitable[None]]) -> None:
        """Args: announce, lo que se hace cuando uno vence (decirlo en voz alta)."""
        self._announce = announce
        self._timers: dict[str, Timer] = {}

    @property
    def active(self) -> list[Timer]:
        """Los que siguen corriendo, el mas proximo primero."""
        return sorted(self._timers.values(), key=lambda t: t.deadline)

    def start(self, seconds: float, label: str | None = None) -> Timer:
        """Programa uno nuevo."""
        timer = Timer(seconds=seconds, label=label, deadline=time.monotonic() + seconds)
        timer.task = asyncio.create_task(self._run(timer), name=f"timer.{timer.timer_id[:8]}")
        self._timers[timer.timer_id] = timer
        _log.info("timer.started", seconds=seconds, label=user_text(label))
        return timer

    def cancel_all(self) -> int:
        """Cancela todos. Devuelve cuantos habia."""
        count = len(self._timers)
        for timer in list(self._timers.values()):
            if timer.task is not None:
                timer.task.cancel()
        self._timers.clear()
        return count

    async def close(self) -> None:
        """Cancela todo y espera a que las tareas mueran."""
        tasks = [t.task for t in self._timers.values() if t.task is not None]
        self.cancel_all()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _run(self, timer: Timer) -> None:
        try:
            await asyncio.sleep(timer.seconds)
        except asyncio.CancelledError:
            return
        self._timers.pop(timer.timer_id, None)
        _log.info("timer.fired", seconds=timer.seconds, label=user_text(timer.label))
        try:
            await self._announce(timer)
        except Exception:
            # Nada falla en silencio, pero un anuncio roto no debe tumbar nada.
            _log.exception("timer.announce_failed")
