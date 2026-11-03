"""El servidor WebSocket local que alimenta al HUD.

Escucha SOLO en la interfaz local (127.0.0.1): el HUD corre en la misma
maquina. El acceso remoto es el puente cifrado y autenticado de la Fase 9, no
este servidor, y por eso aqui una direccion que no sea local se rechaza.

Cada HUD conectado tiene su propia cola corta: si uno se atrasa (la ventana
minimizada deja de leer), pierde los mensajes mas viejos y los demas no lo
notan. Al conectarse recibe un saludo con el estado actual del nucleo.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import TYPE_CHECKING

from websockets.asyncio.server import Server, ServerConnection, serve
from websockets.exceptions import ConnectionClosed

from jarvis.hudlink.feed import HudFeed, Message
from jarvis.logging import get_logger

if TYPE_CHECKING:
    from jarvis.bus.base import EventBus

__all__ = ["LOOPBACK_HOSTS", "HudServer"]

_log = get_logger(__name__)
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
_CLIENT_QUEUE = 64


class _Client:
    """Un HUD conectado y su cola de salida."""

    def __init__(self, connection: ServerConnection) -> None:
        self.connection = connection
        self.queue: asyncio.Queue[str] = asyncio.Queue(maxsize=_CLIENT_QUEUE)

    def offer(self, text: str) -> None:
        if self.queue.full():
            # DECISION: se descarta lo mas viejo. Un cuadro de audio de hace
            # medio segundo ya no le sirve a nadie; el estado mas nuevo si.
            self.queue.get_nowait()
        self.queue.put_nowait(text)

    async def pump(self) -> None:
        while True:
            await self.connection.send(await self.queue.get())


class HudServer:
    """WebSocket en 127.0.0.1 que reparte el `HudFeed` a cada HUD conectado."""

    def __init__(self, bus: EventBus, *, host: str = "127.0.0.1", port: int = 8766) -> None:
        """Args: host, interfaz local; port, 0 elige uno libre (pruebas)."""
        if host not in LOOPBACK_HOSTS:
            msg = f"El HUD solo se sirve en la interfaz local, no en {host!r}"
            raise ValueError(msg)
        self._host = host
        self._port = port
        self._clients: set[_Client] = set()
        self._server: Server | None = None
        self.feed = HudFeed(bus, self._broadcast)

    @property
    def port(self) -> int:
        """El puerto real (util si se pidio el 0)."""
        if self._server is None:
            return self._port
        sockets = list(self._server.sockets)
        return int(sockets[0].getsockname()[1]) if sockets else self._port

    @property
    def clients(self) -> int:
        """Cuantos HUD hay conectados."""
        return len(self._clients)

    async def start(self) -> None:
        """Abre el puerto y empieza a traducir el bus."""
        self.feed.start()
        self._server = await serve(self._handle, self._host, self._port)
        _log.info("hud.listening", host=self._host, port=self.port)

    async def stop(self) -> None:
        """Cierra conexiones y puerto."""
        await self.feed.stop()
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    def _broadcast(self, message: Message) -> None:
        if not self._clients:
            return
        text = json.dumps(message, separators=(",", ":"))
        for client in self._clients:
            client.offer(text)

    async def _handle(self, connection: ServerConnection) -> None:
        client = _Client(connection)
        client.offer(json.dumps(self.feed.hello(), separators=(",", ":")))
        self._clients.add(client)
        _log.info("hud.connected", clients=len(self._clients))
        pump = asyncio.create_task(client.pump())
        try:
            # El HUD no manda nada todavia; se lee para notar el cierre.
            async for _ in connection:
                pass
        except ConnectionClosed:
            pass
        finally:
            self._clients.discard(client)
            pump.cancel()
            with contextlib.suppress(asyncio.CancelledError, ConnectionClosed):
                await pump
            _log.info("hud.disconnected", clients=len(self._clients))
