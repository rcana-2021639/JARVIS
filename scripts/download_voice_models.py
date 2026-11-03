"""Descarga los modelos que necesita el pipeline de voz.

Lee `config/jarvis.toml` para saber cuales: el modelo de faster-whisper
(`stt_model`), la voz de Piper (`tts_voice`) y el de openWakeWord
(`wake_model`). Todo va a `models/`, que no se versiona.

Uso:

    python scripts/download_voice_models.py            # lo que falte
    python scripts/download_voice_models.py --only stt # solo uno

Es seguro correrlo varias veces: lo que ya esta en disco no se vuelve a bajar.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))

from jarvis.config import load_config, project_root
from jarvis.voice.models import (
    download_piper_voice,
    download_wake_models,
    download_whisper,
    resolve_model_paths,
)


def main() -> int:
    """Punto de entrada."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", choices=["stt", "tts", "wake"], help="Bajar solo un modelo.")
    args = parser.parse_args()

    voice = load_config().voice
    paths = resolve_model_paths(voice, project_root())
    wanted = {args.only} if args.only else {"stt", "tts", "wake"}

    if "stt" in wanted:
        if (paths.whisper_dir / "model.bin").exists():
            print(f"[stt]  ya esta: {paths.whisper_dir}")
        else:
            print(f"[stt]  bajando faster-whisper '{voice.stt_model}' ...")
            download_whisper(voice.stt_model, paths.whisper_dir)
            print(f"[stt]  listo: {paths.whisper_dir}")

    if "tts" in wanted:
        if paths.piper_model.exists() and paths.piper_config.exists():
            print(f"[tts]  ya esta: {paths.piper_model}")
        else:
            print(f"[tts]  bajando la voz de Piper '{voice.tts_voice}' ...")
            download_piper_voice(voice.tts_voice, paths.piper_model.parent)
            print(f"[tts]  listo: {paths.piper_model}")

    if "wake" in wanted:
        # Con un modelo propio solo hacen falta los de rasgos y Silero. Un nombre
        # que no existe en el catalogo hace que openWakeWord no baje ninguno mas;
        # una lista vacia, en cambio, le pediria el catalogo entero.
        catalog_name = voice.wake_model
        if Path(voice.wake_model).suffix == ".onnx":
            print(f"[wake] modelo propio, no se descarga: {paths.wake_model}")
            catalog_name = "__ninguno__"
        print(f"[wake] comprobando openWakeWord '{voice.wake_model}' y Silero VAD ...")
        download_wake_models(catalog_name, paths.wake_dir)
        print(f"[wake] listo: {paths.wake_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
