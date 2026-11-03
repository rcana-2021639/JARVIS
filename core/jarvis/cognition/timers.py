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
from jarvis.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

__all__ = ["Timer", "TimerService", "spoken_duration"]

_log = get_logger(__name__)


def spoken_duration(seconds: float) -> str:
    """Una duracion como se dice: "5 minutos", "1 hora y 10 minutos", "30 segundos"."""
    total = max(0, round(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    parts: list[str] = []
    if hours:
        parts.append(f"{hours} hora" + ("s" if hours != 1 else ""))
    if minutes:
        parts.append(f"{minutes} minuto" + ("s" if minutes != 1 else ""))
    if secs and not hours:
        parts.append(f"{secs} segundo" + ("s" if secs != 1 else ""))
    if not parts:
        return "0 segundos"
    return " y ".join(parts) if len(parts) <= 2 else f"{parts[0]}, {parts[1]} y {parts[2]}"


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
        _log.info("timer.started", seconds=seconds, label=label)
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
        _log.info("timer.fired", seconds=timer.seconds, label=timer.label)
        try:
            await self._announce(timer)
        except Exception:
            # Nada falla en silencio, pero un anuncio roto no debe tumbar nada.
            _log.exception("timer.announce_failed")
