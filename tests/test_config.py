"""Pruebas de la configuracion y su recarga en caliente."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from jarvis.config import JarvisConfig, config_sections, load_config
from jarvis.config_watcher import ConfigManager
from jarvis.contracts import Event, EventType, PrivacyMode

if TYPE_CHECKING:
    from pathlib import Path

    from jarvis.bus import InMemoryEventBus

_TOML_MINIMO = """
user_name = "Rhandy"

[voice]
wake_threshold = 0.42

[cognition]
local_model = "qwen3:4b"
daily_budget_usd = 0.50
"""


@pytest.fixture
def toml_file(tmp_path: Path) -> Path:
    """Un jarvis.toml de prueba en un directorio temporal."""
    path = tmp_path / "jarvis.toml"
    path.write_text(_TOML_MINIMO, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Jerarquia de carga
# ---------------------------------------------------------------------------


def test_los_defaults_bastan_para_arrancar(tmp_path: Path) -> None:
    """Sin archivo de configuracion, el sistema tiene que arrancar igual."""
    config = load_config(tmp_path / "no-existe.toml")
    assert config.user_name == "Usuario"
    assert config.voice.wake_word == "jarvis"
    assert config.voice.sample_rate == 16000


def test_el_toml_gana_a_los_defaults(toml_file: Path) -> None:
    """Lo que el usuario escribe en el archivo manda sobre el valor de fabrica."""
    config = load_config(toml_file)
    assert config.user_name == "Rhandy"
    assert config.voice.wake_threshold == pytest.approx(0.42)


def test_el_entorno_gana_al_toml(toml_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """El nivel mas alto de la jerarquia son las variables de entorno."""
    monkeypatch.setenv("JARVIS_VOICE__WAKE_THRESHOLD", "0.91")
    config = load_config(toml_file)
    assert config.voice.wake_threshold == pytest.approx(0.91)


def test_el_entorno_llega_a_secciones_anidadas(monkeypatch: pytest.MonkeyPatch) -> None:
    """El delimitador doble guion bajo es lo que permite anidar."""
    monkeypatch.setenv("JARVIS_COGNITION__LOCAL_MODEL", "qwen3:14b")
    monkeypatch.setenv("JARVIS_UI__PARTICLE_COUNT", "5000")
    config = load_config()
    assert config.cognition.local_model == "qwen3:14b"
    assert config.ui.particle_count == 5000


def test_las_siete_secciones_existen() -> None:
    """El documento maestro pide exactamente estas siete."""
    config = load_config()
    for seccion in config_sections():
        assert hasattr(config, seccion), f"falta la seccion {seccion}"
    assert len(config_sections()) == 7


# ---------------------------------------------------------------------------
# Validacion
# ---------------------------------------------------------------------------


def test_rechaza_valores_fuera_de_rango(tmp_path: Path) -> None:
    """Un umbral de 5.0 no tiene sentido y debe fallar al cargar, no al usarse."""
    path = tmp_path / "jarvis.toml"
    path.write_text("[voice]\nwake_threshold = 5.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="wake_threshold"):
        load_config(path)


def test_rechaza_secciones_desconocidas(tmp_path: Path) -> None:
    """Un typo en el nombre de una clave debe avisarse, no ignorarse."""
    path = tmp_path / "jarvis.toml"
    path.write_text("[voice]\nwake_treshold = 0.5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="wake_treshold"):
        load_config(path)


def test_la_configuracion_es_inmutable() -> None:
    """Nadie muta la configuracion viva por debajo; se sustituye entera."""
    config = load_config()
    with pytest.raises(ValueError, match=r"frozen|immutable"):
        config.user_name = "otro"  # type: ignore[misc]


def test_el_modo_de_privacidad_es_un_enum() -> None:
    """Ningun estado viaja como string suelto."""
    config = load_config()
    assert isinstance(config.cognition.privacy_mode, PrivacyMode)


def test_la_configuracion_se_serializa_para_el_hud() -> None:
    """El HUD recibe la configuracion por WebSocket, asi que debe ser JSON."""
    cable = load_config().to_wire()
    assert isinstance(cable["voice"], dict)
    assert isinstance(cable["cognition"]["privacy_mode"], str)


def test_resuelve_rutas_relativas_contra_la_raiz() -> None:
    """`config/data` debe apuntar al proyecto, no al directorio de trabajo."""
    config = load_config()
    resuelta = config.resolved(config.memory.data_dir)
    assert resuelta.is_absolute()
    assert resuelta.name == "data"


def test_el_puente_no_escucha_en_todas_las_interfaces() -> None:
    """REGLA DURA: nunca 0.0.0.0. El puente solo va por Tailscale."""
    config = load_config()
    assert config.bridge.host != "0.0.0.0"
    assert not config.bridge.enabled, "el puente es de la Fase 9"


# ---------------------------------------------------------------------------
# Recarga en caliente
# ---------------------------------------------------------------------------


async def test_recarga_refleja_el_cambio_sin_reiniciar(toml_file: Path) -> None:
    """El criterio de aceptacion: cambiar un valor se refleja en vivo."""
    manager = ConfigManager(toml_file)
    assert manager.current.voice.wake_threshold == pytest.approx(0.42)

    toml_file.write_text(_TOML_MINIMO.replace("0.42", "0.77"), encoding="utf-8")
    assert await manager.reload()
    assert manager.current.voice.wake_threshold == pytest.approx(0.77)


async def test_el_vigilante_detecta_un_cambio_real(toml_file: Path) -> None:
    """La misma prueba, pero pasando por el vigilante del sistema de archivos."""
    manager = ConfigManager(toml_file)
    await manager.start()
    try:
        assert manager.is_watching
        toml_file.write_text(_TOML_MINIMO.replace("0.42", "0.66"), encoding="utf-8")

        for _ in range(100):
            await asyncio.sleep(0.05)
            if manager.current.voice.wake_threshold != pytest.approx(0.42):
                break

        assert manager.current.voice.wake_threshold == pytest.approx(0.66)
    finally:
        await manager.stop()
    assert not manager.is_watching


async def test_un_toml_roto_conserva_la_configuracion_anterior(toml_file: Path) -> None:
    """Un archivo a medio guardar no puede tumbar un asistente que esta corriendo."""
    manager = ConfigManager(toml_file)
    original = manager.current.voice.wake_threshold

    toml_file.write_text("[voice\nwake_threshold = ", encoding="utf-8")
    assert not await manager.reload()
    assert manager.current.voice.wake_threshold == pytest.approx(original)


async def test_recargar_sin_cambios_no_avisa(toml_file: Path) -> None:
    """Reescribir el mismo contenido no debe disparar callbacks ni eventos."""
    manager = ConfigManager(toml_file)
    toml_file.write_text(_TOML_MINIMO, encoding="utf-8")
    assert not await manager.reload()


async def test_los_callbacks_reciben_la_version_nueva_y_la_vieja(toml_file: Path) -> None:
    """Los modulos que deben reaccionar de verdad necesitan comparar."""
    manager = ConfigManager(toml_file)
    vistos: list[tuple[float, float]] = []

    def al_recargar(nueva: JarvisConfig, vieja: JarvisConfig) -> None:
        vistos.append((nueva.voice.wake_threshold, vieja.voice.wake_threshold))

    manager.on_reload(al_recargar)
    toml_file.write_text(_TOML_MINIMO.replace("0.42", "0.55"), encoding="utf-8")
    await manager.reload()

    assert vistos == [(pytest.approx(0.55), pytest.approx(0.42))]  # type: ignore[comparison-overlap]


async def test_un_callback_roto_no_impide_avisar_al_resto(toml_file: Path) -> None:
    """Nada falla en silencio, pero un fallo tampoco arrastra a los demas."""
    manager = ConfigManager(toml_file)
    llamado = False

    def roto(_nueva: JarvisConfig, _vieja: JarvisConfig) -> None:
        msg = "callback roto a proposito"
        raise RuntimeError(msg)

    def sano(_nueva: JarvisConfig, _vieja: JarvisConfig) -> None:
        nonlocal llamado
        llamado = True

    manager.on_reload(roto)
    manager.on_reload(sano)
    toml_file.write_text(_TOML_MINIMO.replace("0.42", "0.51"), encoding="utf-8")
    await manager.reload()

    assert llamado


async def test_la_recarga_publica_un_evento_con_las_secciones(
    toml_file: Path,
    bus: InMemoryEventBus,
) -> None:
    """El HUD necesita saber que refrescar, no solo que algo cambio."""
    recibidos: list[Event] = []

    async def manejador(evento: Event) -> None:
        recibidos.append(evento)

    bus.subscribe(EventType.CONFIG_RELOADED, manejador)
    manager = ConfigManager(toml_file, bus=bus)
    toml_file.write_text(_TOML_MINIMO.replace("0.42", "0.61"), encoding="utf-8")
    await manager.reload()
    assert await bus.wait_idle(timeout=2.0)

    assert len(recibidos) == 1
    assert recibidos[0].payload["sections"] == ["voice"]
