"""Pruebas de los esquemas JSON compartidos con el HUD y la app movil.

La que de verdad importa aqui es la de desactualizacion: sin ella, alguien
cambia un contrato, se olvida de regenerar, y el HUD queda hablando un idioma
que el nucleo dejo de entender. El bug aparecera semanas despues y muy lejos de
su causa.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from export_schemas import (
    PROTOCOL_DIR,
    build_schemas,
    check_schemas,
)
from jarvis.contracts import EXPORTED_CONTRACTS, CoreState, RiskLevel


def test_los_esquemas_en_disco_estan_al_dia() -> None:
    """Si esta falla, corre: python scripts/export_schemas.py"""
    stale = check_schemas()
    assert not stale, (
        f"Esquemas desactualizados: {stale}. Regenera con 'python scripts/export_schemas.py'."
    )


def test_existe_un_esquema_por_cada_contrato_exportado() -> None:
    """Un contrato sin esquema es un contrato que el HUD nunca vera."""
    generados = build_schemas()
    # Los contratos mas los dos archivos auxiliares.
    assert len(generados) == len(EXPORTED_CONTRACTS) + 2
    assert "index.json" in generados
    assert "enums.json" in generados


def test_los_diez_contratos_del_documento_estan_presentes() -> None:
    """Los que el documento maestro enumera, ni uno menos."""
    nombres = {contrato.__name__ for contrato in EXPORTED_CONTRACTS}
    esperados = {
        "Utterance",
        "Intent",
        "Command",
        "ToolCall",
        "ToolResult",
        "Response",
        "MemoryRecord",
        "SystemState",
        "AuditEntry",
        "Event",
    }
    assert esperados <= nombres


@pytest.mark.parametrize("contrato", EXPORTED_CONTRACTS, ids=lambda c: c.__name__)
def test_cada_esquema_es_json_valido_y_declara_su_version(contrato: type) -> None:
    """Un esquema sin `$schema` no lo puede consumir un generador de tipos."""
    nombre = contrato.__name__
    archivo = next(
        path
        for path in PROTOCOL_DIR.glob("*.schema.json")
        if json.loads(
            path.read_text(encoding="utf-8"),
        ).get("title")
        == nombre
    )
    datos = json.loads(archivo.read_text(encoding="utf-8"))
    assert datos["$schema"].startswith("https://json-schema.org/")
    assert datos["title"] == nombre
    assert "properties" in datos


def test_los_enums_exportados_traen_todos_sus_valores() -> None:
    """El HUD pinta la esfera segun CoreState; le faltan valores, falla en vivo."""
    enums = json.loads((PROTOCOL_DIR / "enums.json").read_text(encoding="utf-8"))["enums"]
    assert set(enums["CoreState"]) == {estado.value for estado in CoreState}
    assert set(enums["RiskLevel"]) == {nivel.value for nivel in RiskLevel}
    assert len(enums["CoreState"]) == 8


def test_el_indice_apunta_a_archivos_que_existen() -> None:
    """Un indice con rutas rotas es peor que no tener indice."""
    indice = json.loads((PROTOCOL_DIR / "index.json").read_text(encoding="utf-8"))
    for nombre, archivo in indice["contracts"].items():
        assert (PROTOCOL_DIR / archivo).exists(), f"falta {archivo} para {nombre}"
    assert (PROTOCOL_DIR / indice["enums"]).exists()


def test_los_valores_de_los_enums_son_ascii() -> None:
    """Estos valores acaban en claves de TOML y en JSON que cruza un WebSocket.

    Un acento ahi es una fuente de errores de codificacion a cambio de nada.
    """
    enums = json.loads((PROTOCOL_DIR / "enums.json").read_text(encoding="utf-8"))["enums"]
    for nombre, valores in enums.items():
        for valor in valores:
            assert str(valor).isascii(), f"{nombre} tiene un valor no ASCII: {valor}"
