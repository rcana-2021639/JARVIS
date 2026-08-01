"""Recarga en caliente de la configuracion.

Cambiar un valor en `config/jarvis.toml` se refleja sin reiniciar el nucleo. No
es una comodidad de desarrollo: afinar el umbral de la palabra clave o el
presupuesto diario implica probar valores en vivo, y reiniciar un asistente
residente para mover un decimal es exactamente lo que la Fase 11 prohibe.

DECISION: la configuracion es inmutable y la recarga la SUSTITUYE entera en vez
de mutar campos. Un consumidor que guardo `config.voice` seguira viendo la
version vieja, que es correcto y predecible; quien quiera el valor vivo consulta
`manager.current`. Mutar en sitio produciria lecturas a medias justo mientras se
recarga.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING

from pydantic import JsonValue
from watchfiles import awatch

from jarvis.config import JarvisConfig, default_config_path, load_config
from jarvis.contracts.enums import EventType
from jarvis.contracts.events import Event
from jarvis.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from jarvis.bus.base import EventBus

__all__ = ["ConfigManager", "ReloadCallback"]

_log = get_logger(__name__)

ReloadCallback = "Callable[[JarvisConfig, JarvisConfig], None]"
"""Recibe (configuracion_nueva, configuracion_anterior)."""

#: Espera tras detectar un cambio antes de releer.
#: Los editores escriben en dos pasos (truncar y luego volcar); leer en medio da
#: un TOML incompleto. 50ms cubre eso sin que se sienta lento.
_DEBOUNCE_MS = 50

#: Margen que se da al vigilante para registrarse en el sistema de archivos.
#: `awatch` monta el observador dentro de su primera iteracion, y hasta que eso
#: ocurre los cambios pasan desapercibidos. Ver `start`.
_STARTUP_SETTLE_S = 0.25


class ConfigManager:
    """Mantiene la configuracion viva y avisa cuando cambia.

    Uso:

        manager = ConfigManager(bus=bus)
        await manager.start()
        ...
        manager.current.voice.wake_threshold   # siempre el valor vigente
    """

    def __init__(
        self,
        path: Path | None = None,
        *,
        bus: EventBus | None = None,
    ) -> None:
        """Carga la configuracion inicial sin arrancar todavia el vigilante."""
        self._path = path if path is not None else default_config_path()
        self._bus = bus
        self._config = load_config(self._path)
        self._callbacks: list[Callable[[JarvisConfig, JarvisConfig], None]] = []
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()
        self._ready = asyncio.Event()

    @property
    def current(self) -> JarvisConfig:
        """La configuracion vigente en este instante."""
        return self._config

    @property
    def path(self) -> Path:
        """Archivo TOML vigilado."""
        return self._path

    @property
    def is_watching(self) -> bool:
        """Si el vigilante esta corriendo."""
        return self._task is not None and not self._task.done()

    def on_reload(self, callback: Callable[[JarvisConfig, JarvisConfig], None]) -> None:
        """Registra una funcion que se llama tras cada recarga con exito.

        Es para los modulos que necesitan reaccionar de verdad a un cambio --
        reabrir el microfono si cambio el dispositivo, por ejemplo -- y no solo
        leer el valor nuevo la proxima vez.
        """
        self._callbacks.append(callback)

    async def start(self) -> None:
        """Arranca el vigilante del archivo y espera a que este operativo.

        No basta con crear la tarea: `asyncio.create_task` no la ejecuta hasta
        el siguiente punto de espera, y `awatch` solo registra el observador del
        sistema de archivos dentro de su primera iteracion. Devolver antes de
        eso hace que los cambios inmediatos se pierdan en silencio -- que es
        justo el fallo que tenia esta clase.

        Por eso `start` espera a que el vigilante avise que arranco y despues
        relee una vez, por si el archivo cambio durante ese arranque.
        """
        if self.is_watching:
            return
        self._stop.clear()
        self._ready.clear()
        self._task = asyncio.create_task(self._watch(), name="config-watcher")
        await self._ready.wait()
        await self.reload()

    async def stop(self) -> None:
        """Detiene el vigilante."""
        self._stop.set()
        task = self._task
        self._task = None
        if task is None or task.done():
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def reload(self) -> bool:
        """Relee el archivo y sustituye la configuracion.

        Devuelve True si se aplico un cambio. Si el TOML quedo mal escrito, se
        registra el error y se CONSERVA la configuracion anterior: un archivo a
        medio guardar no puede tumbar un asistente que esta corriendo.
        """
        previous = self._config
        try:
            fresh = load_config(self._path)
        except Exception:
            _log.exception("config.reload.failed", path=str(self._path))
            return False

        if fresh == previous:
            return False

        self._config = fresh
        changed = self._changed_sections(previous, fresh)
        _log.info("config.reloaded", path=str(self._path), sections=changed)
        self._notify(fresh, previous)
        await self._publish_reloaded(changed)
        return True

    def _notify(self, fresh: JarvisConfig, previous: JarvisConfig) -> None:
        """Llama a los interesados. Un callback roto no impide avisar al resto."""
        for callback in self._callbacks:
            try:
                callback(fresh, previous)
            except Exception:
                name = str(getattr(callback, "__name__", "anonimo"))
                _log.exception("config.callback.failed", callback=name)

    async def _publish_reloaded(self, sections: list[str]) -> None:
        """Publica CONFIG_RELOADED para que el HUD refresque sus paneles."""
        if self._bus is None:
            return
        # La anotacion explicita es lo que hace que la lista encaje en JsonValue:
        # list[str] no es list[JsonValue] porque las listas son invariantes.
        changed: list[JsonValue] = [str(section) for section in sections]
        await self._bus.publish(
            Event(
                type=EventType.CONFIG_RELOADED,
                payload={"path": str(self._path), "sections": changed},
                source="config.watcher",
            ),
        )

    @staticmethod
    def _changed_sections(previous: JarvisConfig, fresh: JarvisConfig) -> list[str]:
        """Que secciones cambiaron entre dos configuraciones."""
        before = previous.model_dump()
        after = fresh.model_dump()
        return sorted(key for key in after if before.get(key) != after.get(key))

    async def _watch(self) -> None:
        """Bucle del vigilante. Solo termina al cancelarse o al pedir stop.

        El aviso de "listo" corre EN PARALELO al bucle, no antes: `awatch` monta
        el observador del sistema de archivos dentro de su primera iteracion, asi
        que avisar antes de entrar al bucle seria mentir. Lanzandolo en paralelo,
        el margen de espera transcurre mientras el observador ya se esta
        registrando.
        """
        signal = asyncio.create_task(self._signal_ready(), name="config-ready")
        try:
            async for _changes in awatch(
                self._path.parent,
                stop_event=self._stop,
                debounce=_DEBOUNCE_MS,
            ):
                await self.reload()
        except asyncio.CancelledError:
            raise
        except Exception:
            _log.exception("config.watcher.crashed", path=str(self._path))
        finally:
            signal.cancel()
            self._ready.set()

    async def _signal_ready(self) -> None:
        """Espera el margen de arranque y marca el vigilante como operativo."""
        await asyncio.sleep(_STARTUP_SETTLE_S)
        self._ready.set()
        _log.info("config.watching", path=str(self._path))
