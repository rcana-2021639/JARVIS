"""Exporta los contratos como esquemas JSON a `shared/protocol/`.

Sin esto, el HUD (TypeScript) y la app movil (React Native) tendrian que
redeclarar a mano los mismos tipos que ya existen en Python. Redeclarar a mano
significa que algun dia se desincronizan, y ese dia el bug aparece en el sitio
mas lejano posible de la causa.

Uso:

    python scripts/export_schemas.py            # escribe los esquemas
    python scripts/export_schemas.py --check    # falla si estan desactualizados

El modo `--check` es lo que evita que alguien cambie un contrato y se olvide de
regenerar. Corre en el arnes de pruebas.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))

from jarvis.contracts import EXPORTED_CONTRACTS
from jarvis.contracts import enums as contract_enums

PROTOCOL_DIR = Path(__file__).resolve().parents[1] / "shared" / "protocol"

#: Enums que el HUD y el movil necesitan literalmente: la esfera 3D cambia de
#: color segun CoreState y el timeline pinta cada accion segun RiskLevel.
EXPORTED_ENUMS = (
    contract_enums.CoreState,
    contract_enums.RiskLevel,
    contract_enums.CascadeLevel,
    contract_enums.CommandOrigin,
    contract_enums.MemoryKind,
    contract_enums.Language,
    contract_enums.Emotion,
    contract_enums.PrivacyMode,
    contract_enums.EventType,
)


def _schema_filename(name: str) -> str:
    """Convierte `SystemState` en `system-state.schema.json`."""
    parts: list[str] = []
    for index, char in enumerate(name):
        if char.isupper() and index > 0:
            parts.append("-")
        parts.append(char.lower())
    return f"{''.join(parts)}.schema.json"


def build_schemas() -> dict[str, str]:
    """Genera el contenido de todos los archivos a exportar.

    Returns:
        Un diccionario de nombre de archivo a contenido JSON ya formateado.
    """
    files: dict[str, str] = {}

    for contract in EXPORTED_CONTRACTS:
        schema = contract.model_json_schema(mode="serialization")
        schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        schema["title"] = contract.__name__
        files[_schema_filename(contract.__name__)] = _dump(schema)

    enums_payload = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "JarvisEnums",
        "description": (
            "Valores canonicos de los enums del nucleo. El HUD y la app movil "
            "los consumen para no redeclarar constantes a mano."
        ),
        "enums": {
            enum_cls.__name__: [member.value for member in enum_cls]
            for enum_cls in EXPORTED_ENUMS
        },
    }
    files["enums.json"] = _dump(enums_payload)

    index = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "JarvisProtocolIndex",
        "description": "Indice de los contratos compartidos entre el nucleo, el HUD y el movil.",
        "generatedBy": "scripts/export_schemas.py",
        "contracts": {
            contract.__name__: _schema_filename(contract.__name__)
            for contract in EXPORTED_CONTRACTS
        },
        "enums": "enums.json",
    }
    files["index.json"] = _dump(index)

    return files


def _dump(payload: object) -> str:
    """Serializa con formato estable para que el diff sea legible."""
    return json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def write_schemas(target: Path = PROTOCOL_DIR) -> list[Path]:
    """Escribe los esquemas al disco. Devuelve las rutas escritas."""
    target.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for filename, content in build_schemas().items():
        path = target / filename
        path.write_text(content, encoding="utf-8")
        written.append(path)
    return written


def check_schemas(target: Path = PROTOCOL_DIR) -> list[str]:
    """Compara lo que hay en disco con lo que deberia haber.

    Returns:
        Lista de archivos desactualizados o ausentes. Vacia si todo esta al dia.
    """
    stale: list[str] = []
    for filename, expected in build_schemas().items():
        path = target / filename
        if not path.exists():
            stale.append(f"{filename} (no existe)")
        elif path.read_text(encoding="utf-8") != expected:
            stale.append(f"{filename} (desactualizado)")
    return stale


def main() -> int:
    """Punto de entrada."""
    parser = argparse.ArgumentParser(description="Exporta los contratos a JSON Schema.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="No escribe nada; falla si los esquemas en disco no estan al dia.",
    )
    args = parser.parse_args()

    if args.check:
        stale = check_schemas()
        if stale:
            print("Esquemas desactualizados:")
            for item in stale:
                print(f"  - {item}")
            print("\nRegenera con: python scripts/export_schemas.py")
            return 1
        print(f"Esquemas al dia ({len(build_schemas())} archivos).")
        return 0

    written = write_schemas()
    print(f"Escritos {len(written)} archivos en {PROTOCOL_DIR}:")
    for path in written:
        print(f"  - {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
