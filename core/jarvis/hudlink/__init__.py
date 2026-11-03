"""El canal local del nucleo al HUD (Fase 4).

    feed.py    del bus a cuatro mensajes: estado, audio (nivel + 32 bandas), carga
    server.py  WebSocket en 127.0.0.1 con una cola corta por HUD conectado

No es el puente remoto (Fase 9): este solo escucha en la interfaz local.
"""

from jarvis.hudlink.feed import PROTOCOL_VERSION, HudFeed
from jarvis.hudlink.launch import start_hud
from jarvis.hudlink.server import HudServer

__all__ = ["PROTOCOL_VERSION", "HudFeed", "HudServer", "start_hud"]
