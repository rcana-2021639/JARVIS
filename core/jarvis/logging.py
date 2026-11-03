"""Registro estructurado y trazado de duraciones.

DECISION: este modulo NO importa el bus en tiempo de ejecucion. El bus registra
sus fallos con `get_logger()` y el decorador `@traced` publica al bus; si ambos
se importaran mutuamente habria un ciclo. Se resuelve con inyeccion: el nucleo
llama a `bind_bus(bus)` una vez al arrancar y `@traced` usa esa referencia. Sin
bus enlazado, `@traced` sigue midiendo y registrando, simplemente no publica.
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import logging
import logging.handlers
import sys
import time
from typing import TYPE_CHECKING, ParamSpec, TypeVar, cast

import structlog

from jarvis.contracts.base import CorrelationId
from jarvis.contracts.enums import EventType
from jarvis.contracts.events import Event

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

    from structlog.typing import EventDict, WrappedLogger

    from jarvis.bus.base import EventBus

__all__ = [
    "bind_bus",
    "clear_correlation_id",
    "configure_logging",
    "current_correlation_id",
    "get_logger",
    "set_correlation_id",
    "traced",
    "user_text",
]

P = ParamSpec("P")
R = TypeVar("R")

#: Id de correlacion de la interaccion en curso.
#: Es un ContextVar y no un global: asyncio corre varias interacciones a la vez
#: (una orden por voz mientras llega otra del movil) y cada tarea necesita el
#: suyo. Un global las mezclaria y la bitacora quedaria inservible.
_correlation_id: contextvars.ContextVar[CorrelationId | None] = contextvars.ContextVar(
    "jarvis_correlation_id",
    default=None,
)

#: Bus enlazado por el nucleo al arrancar. Ver `bind_bus`.
_bus: EventBus | None = None

_configured = False

_log_user_content = False
"""Si el registro puede guardar lo que dice el usuario. Ver `user_text`."""


def set_correlation_id(correlation_id: CorrelationId) -> contextvars.Token[CorrelationId | None]:
    """Fija el id de correlacion de la interaccion en curso.

    Devuelve el token para poder restaurarlo despues con `clear_correlation_id`.
    """
    return _correlation_id.set(correlation_id)


def current_correlation_id() -> CorrelationId | None:
    """Id de correlacion de la interaccion en curso, si hay una."""
    return _correlation_id.get()


def clear_correlation_id(token: contextvars.Token[CorrelationId | None]) -> None:
    """Restaura el id de correlacion anterior."""
    _correlation_id.reset(token)


def bind_bus(bus: EventBus | None) -> None:
    """Enlaza el bus al que `@traced` publica sus mediciones.

    Se llama una vez al arrancar el nucleo. Pasar None lo desenlaza, que es lo
    que hacen las pruebas para no ensuciarse entre si.
    """
    global _bus
    _bus = bus


def _add_correlation_id(
    _logger: WrappedLogger,
    _method: str,
    event_dict: EventDict,
) -> EventDict:
    """Procesador de structlog que inyecta el id de correlacion en cada linea.

    Es lo que permite reconstruir despues una interaccion completa filtrando por
    un solo campo, aunque sus lineas vengan de seis modulos distintos.
    """
    correlation_id = _correlation_id.get()
    if correlation_id is not None:
        event_dict["correlation_id"] = correlation_id
    return event_dict


def configure_logging(
    *,
    level: str = "INFO",
    development: bool = True,
    log_file: Path | None = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    log_user_content: bool = False,
) -> None:
    """Configura structlog para todo el proceso.

    Args:
        level: nivel minimo (DEBUG, INFO, WARNING, ERROR).
        development: salida coloreada y legible. En produccion, JSON puro.
        log_file: si se indica, ademas escribe a archivo con rotacion.
        max_bytes: tamano al que rota el archivo.
        backup_count: cuantos archivos rotados se conservan.
        log_user_content: si el registro guarda lo que dice el usuario. Apagado,
            `user_text` lo reemplaza por su longitud.
    """
    global _configured, _log_user_content
    _log_user_content = log_user_content

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(
            logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            ),
        )

    logging.basicConfig(format="%(message)s", level=numeric_level, handlers=handlers, force=True)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            _add_correlation_id,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(colors=development)
            if development
            else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _configured = True


def user_text(text: str | None) -> str:
    """Lo que el usuario dijo, listo para el registro.

    DECISION: por defecto el registro guarda `<N caracteres>`, no el contenido.
    El archivo de registro vive en texto plano en el disco y rota durante
    semanas: guardar ahi cada orden seria un diario de la vida del usuario sin
    cifrar. La memoria (Fase 3) y la auditoria (Fase 8) tienen su propio
    almacenamiento, con sus propias reglas. Para depurar,
    `log_user_content = true` en jarvis.toml.
    """
    if text is None:
        return "<nada>"
    return text if _log_user_content else f"<{len(text)} caracteres>"


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Devuelve un logger estructurado para un modulo.

    Si nadie llamo a `configure_logging`, lo configura con valores de desarrollo
    para que importar un modulo suelto en un REPL no reviente.
    """
    if not _configured:
        configure_logging()
    return cast("structlog.stdlib.BoundLogger", structlog.get_logger(name))


async def _publish_span(operation: str, duration_ms: float, *, ok: bool) -> None:
    """Publica una medicion de duracion al bus, si hay bus enlazado."""
    bus = _bus
    if bus is None:
        return
    await bus.publish(
        Event(
            type=EventType.TRACE_SPAN,
            payload={"operation": operation, "duration_ms": round(duration_ms, 3), "ok": ok},
            source="logging.traced",
            correlation_id=_correlation_id.get(),
        ),
    )


def traced(
    name: str | None = None,
    *,
    publish: bool = True,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Mide cuanto tarda una funcion, lo registra y lo publica al bus.

    Funciona sobre corrutinas y sobre funciones normales. En el caso sincrono no
    se publica al bus (no hay bucle de eventos donde hacerlo con seguridad); solo
    se registra.

    Una excepcion dentro de la funcion decorada se registra con su duracion y se
    vuelve a lanzar. El decorador mide, no traga errores.

    Args:
        name: nombre de la operacion. Por defecto `modulo.funcion`.
        publish: si publicar el evento TRACE_SPAN al bus.

    Example:
        >>> @traced("stt.transcribe")
        ... async def transcribe(audio: bytes) -> str: ...
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        operation = name or f"{func.__module__.rsplit('.', 1)[-1]}.{func.__name__}"
        logger = get_logger(func.__module__)

        if asyncio.iscoroutinefunction(func):
            # `func` esta tipada como Callable[P, R]; en esta rama sabemos que R
            # es en realidad una corrutina, pero el sistema de tipos no puede
            # deducirlo solo del `iscoroutinefunction`.
            async_func = cast("Callable[P, Awaitable[R]]", func)

            @functools.wraps(func)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                started = time.perf_counter()
                try:
                    result = await async_func(*args, **kwargs)
                except Exception:
                    failed_ms = (time.perf_counter() - started) * 1000.0
                    logger.exception("span.failed", operation=operation, duration_ms=failed_ms)
                    if publish:
                        await _publish_span(operation, failed_ms, ok=False)
                    raise
                duration_ms = (time.perf_counter() - started) * 1000.0
                logger.debug("span", operation=operation, duration_ms=duration_ms)
                if publish:
                    await _publish_span(operation, duration_ms, ok=True)
                return result

            return cast("Callable[P, R]", async_wrapper)

        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            started = time.perf_counter()
            try:
                result = func(*args, **kwargs)
            except Exception:
                failed_ms = (time.perf_counter() - started) * 1000.0
                logger.exception("span.failed", operation=operation, duration_ms=failed_ms)
                raise
            duration_ms = (time.perf_counter() - started) * 1000.0
            logger.debug("span", operation=operation, duration_ms=duration_ms)
            return result

        return sync_wrapper

    return decorator
