"""Arrancar el canal del HUD desde un script, sin que su falla tumbe a Jarvis."""

from __future__ import annotations

from typing import TYPE_CHECKING

from jarvis.hudlink.server import HudServer
from jarvis.logging import get_logger

if TYPE_CHECKING:
    from jarvis.bus.base import EventBus
    from jarvis.config import UIConfig

__all__ = ["start_hud"]

_log = get_logger(__name__)


async def start_hud(bus: EventBus, ui: UIConfig) -> HudServer | None:
    """El servidor del HUD si esta habilitado y el puerto esta libre; si no, None.

    Nada falla en silencio: si el puerto esta ocupado se registra y se sigue,
    porque Jarvis sin esfera sigue siendo Jarvis.
    """
    if not ui.hud_enabled:
        return None
    server = HudServer(bus, host=ui.hud_host, port=ui.hud_port)
    try:
        await server.start()
    except OSError as error:
        await server.stop()
        _log.warning("hud.unavailable", port=ui.hud_port, error=str(error))
        return None
    return server
