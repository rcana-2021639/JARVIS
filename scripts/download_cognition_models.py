"""Descarga los modelos de la cascada cognitiva.

    MiniLM multilingue (ONNX, ~118 MB)  -> models/intent/   clasificador
    el modelo local de Ollama           -> via `ollama pull` nivel 2

Lee `config/jarvis.toml` para saber cuales. Es seguro correrlo varias veces.

Uso:

    python scripts/download_cognition_models.py
    python scripts/download_cognition_models.py --skip-ollama
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))

from jarvis.cognition.embeddings import MODEL_FILE, TOKENIZER_FILE
from jarvis.config import load_config

REPO = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# Cuantizado para AVX2: es el que corre en cualquier x86 de los ultimos 10 anos.
FILES = {"onnx/model_quint8_avx2.onnx": MODEL_FILE, "tokenizer.json": TOKENIZER_FILE}


def download_minilm(target: Path) -> None:
    """Baja el modelo y el tokenizador a `target`."""
    from huggingface_hub import hf_hub_download

    target.mkdir(parents=True, exist_ok=True)
    for remote, local in FILES.items():
        destination = target / local
        if destination.exists():
            print(f"[intent] ya esta: {destination}")
            continue
        print(f"[intent] bajando {remote} ...")
        shutil.copy(hf_hub_download(REPO, remote), destination)
        print(f"[intent] listo: {destination} ({destination.stat().st_size // 1_000_000} MB)")


def ollama_executable() -> str | None:
    """Donde esta ollama: en el PATH o en la instalacion por defecto de Windows."""
    found = shutil.which("ollama")
    if found:
        return found
    default = Path.home() / "AppData/Local/Programs/Ollama/ollama.exe"
    return str(default) if default.exists() else None


def main() -> int:
    """Punto de entrada."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip-ollama", action="store_true", help="No tocar Ollama.")
    args = parser.parse_args()

    config = load_config()
    download_minilm(config.resolved(config.cognition.intent_model_dir))

    if args.skip_ollama:
        return 0
    ollama = ollama_executable()
    if ollama is None:
        print("[local] Ollama no esta instalado: https://ollama.com/download")
        return 1
    model = config.cognition.local_model
    print(f"[local] ollama pull {model} ...")
    return subprocess.call([ollama, "pull", model])


if __name__ == "__main__":
    sys.exit(main())
