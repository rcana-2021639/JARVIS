"""Secretos: solo en el llavero del sistema operativo.

En Windows es el Administrador de credenciales; en macOS el Llavero; en Linux
Secret Service. Nunca un archivo, nunca el registro, nunca el contexto de un
modelo. Para guardar uno: `python scripts/set_secret.py anthropic_api_key`.

La Fase 8 construye la boveda completa encima; esto es lo minimo para que la
Fase 2 lea la clave de la API sin violar la regla.
"""

from __future__ import annotations

import os

from jarvis.logging import get_logger

__all__ = ["KEYRING_SERVICE", "get_secret", "set_secret"]

_log = get_logger(__name__)

KEYRING_SERVICE = "jarvis"

_ENV_FALLBACK = {"anthropic_api_key": "ANTHROPIC_API_KEY"}
"""Variables de entorno aceptadas como respaldo. Viven en memoria, no en disco."""


def get_secret(name: str) -> str | None:
    """Lee un secreto. None si no existe. Nunca registra su valor."""
    try:
        import keyring
        from keyring.errors import KeyringError

        try:
            value = keyring.get_password(KEYRING_SERVICE, name)
        except KeyringError as error:
            _log.warning("secrets.keyring_failed", name=name, error=type(error).__name__)
            value = None
    except ImportError:
        value = None
    if value:
        return value
    env_name = _ENV_FALLBACK.get(name)
    return os.environ.get(env_name) or None if env_name else None


def set_secret(name: str, value: str) -> None:
    """Guarda un secreto en el llavero del sistema."""
    import keyring

    keyring.set_password(KEYRING_SERVICE, name, value)
