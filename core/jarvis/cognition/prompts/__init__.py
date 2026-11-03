"""Las instrucciones de sistema de Jarvis, como archivos de texto editables.

    jarvis.md       la personalidad: compartida por los niveles 2 y 3
    local_json.md   el formato con autoevaluacion que exige el nivel 2

Viven fuera del codigo para poder afinarlas sin tocar Python.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

__all__ = ["load_prompt", "system_prompt"]

_DIR = Path(__file__).resolve().parent


@cache
def load_prompt(name: str) -> str:
    """El texto de `prompts/<name>.md`."""
    return (_DIR / f"{name}.md").read_text(encoding="utf-8").strip()


def system_prompt(user_name: str, *, structured: bool = False) -> str:
    """La instruccion de sistema completa.

    Args:
        user_name: como se llama el usuario.
        structured: si agrega el formato JSON con autoevaluacion del nivel 2.

    DECISION: no lleva hora ni estado. Un prefijo identico entre peticiones es
    lo que permite a Ollama reutilizar su cache de prompt y a la API de Claude
    su cache de prefijo. El estado viaja al final, junto a la orden.
    """
    text = load_prompt("jarvis").format(user_name=user_name)
    if structured:
        text = f"{text}\n\n{load_prompt('local_json')}"
    return text
