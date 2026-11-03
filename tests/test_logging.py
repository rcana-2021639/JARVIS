"""Pruebas del registro estructurado y del decorador @traced."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from jarvis.contracts import Event, EventType, new_correlation_id
from jarvis.logging import (
    bind_bus,
    clear_correlation_id,
    configure_logging,
    current_correlation_id,
    get_logger,
    set_correlation_id,
    traced,
)

if TYPE_CHECKING:
    from pathlib import Path

    from jarvis.bus import InMemoryEventBus


# ---------------------------------------------------------------------------
# Id de correlacion
# ---------------------------------------------------------------------------


def test_el_id_de_correlacion_se_fija_y_se_restaura() -> None:
    """Es lo que permite reconstruir una interaccion completa despues."""
    assert current_correlation_id() is None
    identificador = new_correlation_id()
    token = set_correlation_id(identificador)
    try:
        assert current_correlation_id() == identificador
    finally:
        clear_correlation_id(token)
    assert current_correlation_id() is None


async def test_cada_tarea_tiene_su_propio_id_de_correlacion() -> None:
    """Dos ordenes en paralelo -- una por voz y otra del movil -- no se mezclan."""
    vistos: dict[str, str | None] = {}

    async def interaccion(nombre: str) -> None:
        propio = new_correlation_id()
        set_correlation_id(propio)
        await asyncio.sleep(0.01)
        vistos[nombre] = current_correlation_id()
        assert vistos[nombre] == propio

    await asyncio.gather(interaccion("voz"), interaccion("movil"))
    assert vistos["voz"] != vistos["movil"]


# ---------------------------------------------------------------------------
# Configuracion del registro
# ---------------------------------------------------------------------------


def test_get_logger_funciona_sin_configurar_antes() -> None:
    """Importar un modulo suelto en un REPL no debe reventar."""
    logger = get_logger("prueba")
    logger.info("evento.de.prueba", dato=1)


def test_el_registro_escribe_a_archivo_con_rotacion(tmp_path: Path) -> None:
    """La Fase 11 necesita registros en disco, no solo en consola."""
    log_file = tmp_path / "logs" / "jarvis.log"
    configure_logging(level="INFO", development=False, log_file=log_file)
    get_logger("prueba").info("linea.de.prueba", valor=42)

    assert log_file.exists()
    contenido = log_file.read_text(encoding="utf-8")
    assert "linea.de.prueba" in contenido
    configure_logging()


def test_el_registro_en_produccion_es_json(tmp_path: Path) -> None:
    """En produccion el registro se parsea con herramientas, no con los ojos."""
    import json

    log_file = tmp_path / "jarvis.log"
    configure_logging(level="INFO", development=False, log_file=log_file)
    identificador = new_correlation_id()
    token = set_correlation_id(identificador)
    try:
        get_logger("prueba").info("evento.json", numero=7)
    finally:
        clear_correlation_id(token)

    primera = log_file.read_text(encoding="utf-8").strip().splitlines()[0]
    datos = json.loads(primera)
    assert datos["event"] == "evento.json"
    assert datos["numero"] == 7
    assert datos["correlation_id"] == identificador
    configure_logging()


# ---------------------------------------------------------------------------
# Decorador @traced
# ---------------------------------------------------------------------------


async def test_traced_mide_una_corrutina_y_publica_al_bus(bus: InMemoryEventBus) -> None:
    """El decorador es la fuente de las metricas de latencia de todo el sistema."""
    recibidos: list[Event] = []

    async def manejador(evento: Event) -> None:
        recibidos.append(evento)

    bus.subscribe(EventType.TRACE_SPAN, manejador)
    bind_bus(bus)

    @traced("prueba.operacion")
    async def operacion() -> str:
        await asyncio.sleep(0.02)
        return "listo"

    assert await operacion() == "listo"
    assert await bus.wait_idle(timeout=2.0)

    assert len(recibidos) == 1
    carga = recibidos[0].payload
    assert carga["operation"] == "prueba.operacion"
    assert carga["ok"] is True
    duracion = carga["duration_ms"]
    assert isinstance(duracion, float)
    # En Windows el temporizador de asyncio puede despertar un pelo antes: con
    # la CPU ocupada se midio 19.7 ms para un sleep de 20.
    assert duracion >= 18.0


async def test_traced_publica_tambien_cuando_falla(bus: InMemoryEventBus) -> None:
    """Una operacion que revienta tambien tardo algo, y eso importa medirlo."""
    recibidos: list[Event] = []

    async def manejador(evento: Event) -> None:
        recibidos.append(evento)

    bus.subscribe(EventType.TRACE_SPAN, manejador)
    bind_bus(bus)

    @traced("prueba.rota")
    async def rota() -> None:
        msg = "fallo esperado"
        raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="fallo esperado"):
        await rota()
    assert await bus.wait_idle(timeout=2.0)

    assert recibidos[0].payload["ok"] is False


async def test_traced_vuelve_a_lanzar_la_excepcion() -> None:
    """El decorador mide, no traga errores."""
    bind_bus(None)

    @traced("prueba.propaga")
    async def rota() -> None:
        msg = "debe propagarse"
        raise ValueError(msg)

    with pytest.raises(ValueError, match="debe propagarse"):
        await rota()


async def test_traced_funciona_sin_bus_enlazado() -> None:
    """Antes de que el nucleo arranque el bus, @traced no puede reventar."""
    bind_bus(None)

    @traced("prueba.sin_bus")
    async def operacion() -> int:
        return 7

    assert await operacion() == 7


def test_traced_tambien_envuelve_funciones_sincronas() -> None:
    """No todo el nucleo es asincrono; el clasificador de la Fase 2 no lo sera."""

    @traced("prueba.sincrona")
    def suma(a: int, b: int) -> int:
        return a + b

    assert suma(2, 3) == 5


def test_traced_conserva_la_identidad_de_la_funcion() -> None:
    """Sin `functools.wraps`, los nombres en los registros serian todos 'wrapper'."""

    @traced()
    def mi_operacion() -> None:
        """Docstring original."""

    assert mi_operacion.__name__ == "mi_operacion"
    assert mi_operacion.__doc__ == "Docstring original."


async def test_traced_propaga_el_id_de_correlacion(bus: InMemoryEventBus) -> None:
    """La medicion debe poder atribuirse a la interaccion que la provoco."""
    recibidos: list[Event] = []

    async def manejador(evento: Event) -> None:
        recibidos.append(evento)

    bus.subscribe(EventType.TRACE_SPAN, manejador)
    bind_bus(bus)
    identificador = new_correlation_id()
    token = set_correlation_id(identificador)

    @traced("prueba.correlacionada")
    async def operacion() -> None:
        return

    try:
        await operacion()
    finally:
        clear_correlation_id(token)
    assert await bus.wait_idle(timeout=2.0)

    assert recibidos[0].correlation_id == identificador


async def test_traced_puede_no_publicar(bus: InMemoryEventBus) -> None:
    """Para operaciones muy frecuentes, publicar cada medicion seria ruido."""
    recibidos: list[Event] = []

    async def manejador(evento: Event) -> None:
        recibidos.append(evento)

    bus.subscribe(EventType.TRACE_SPAN, manejador)
    bind_bus(bus)

    @traced("prueba.silenciosa", publish=False)
    async def operacion() -> None:
        return

    await operacion()
    await bus.wait_idle(timeout=1.0)
    assert recibidos == []


def test_lo_que_dice_el_usuario_no_va_al_registro_por_defecto() -> None:
    from jarvis.logging import user_text

    configure_logging(level="WARNING", log_file=None)
    assert user_text("mi contraseña del banco es 1234") == "<31 caracteres>"
    assert user_text(None) == "<nada>"
    configure_logging(level="WARNING", log_file=None, log_user_content=True)
    try:
        assert user_text("hola") == "hola"
    finally:
        configure_logging(level="WARNING", log_file=None)
