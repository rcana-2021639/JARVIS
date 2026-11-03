"""Guarda un secreto en el llavero del sistema operativo.

En Windows queda en el Administrador de credenciales, bajo el servicio
"jarvis". El valor se escribe sin eco y nunca toca un archivo ni un registro.

Uso:

    python scripts/set_secret.py anthropic_api_key
    python scripts/set_secret.py anthropic_api_key --delete
"""

from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))

from jarvis.secrets import KEYRING_SERVICE, get_secret, set_secret


def main() -> int:
    """Punto de entrada."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name", help="Nombre del secreto, p. ej. anthropic_api_key")
    parser.add_argument("--delete", action="store_true", help="Borrarlo del llavero.")
    args = parser.parse_args()

    if args.delete:
        import keyring

        keyring.delete_password(KEYRING_SERVICE, args.name)
        print(f"Borrado: {args.name}")
        return 0

    value = getpass.getpass(f"Valor de {args.name} (no se muestra): ").strip()
    if not value:
        print("Vacio: no se guardo nada.")
        return 1
    set_secret(args.name, value)
    stored = get_secret(args.name)
    print("Guardado en el llavero." if stored == value else "No se pudo verificar el guardado.")
    return 0 if stored == value else 1


if __name__ == "__main__":
    sys.exit(main())
