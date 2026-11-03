"""Donde viven los modelos de voz y como se consiguen.

Los tres motores locales necesitan pesos que no se versionan (`models/` esta en
.gitignore): faster-whisper, la voz de Piper y openWakeWord. Este modulo fija
la ruta de cada uno a partir de la configuracion y sabe descargarlos.

DECISION: la descarga es un paso explicito (`scripts/download_voice_models.py`)
y nunca ocurre sola al arrancar. Un asistente residente que baja 500 MB de
internet porque falto un archivo, sin que nadie se lo pidiera, viola el
principio de "local por defecto". Si falta un modelo, el arranque falla con un
`ModelMissingError` que dice exactamente que comando correr.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jarvis.config import VoiceConfig

__all__ = [
    "DOWNLOAD_HINT",
    "ModelMissingError",
    "VoiceModelPaths",
    "download_piper_voice",
    "download_wake_models",
    "download_whisper",
    "resolve_model_paths",
]

DOWNLOAD_HINT = ".venv/Scripts/python scripts/download_voice_models.py"
"""Comando que el mensaje de error recomienda cuando falta un modelo."""


class ModelMissingError(RuntimeError):
    """Falta un modelo en disco. El mensaje dice cual y como conseguirlo."""

    def __init__(self, what: str, path: Path) -> None:
        """Construye el error con el modelo que falta y donde se esperaba."""
        super().__init__(f"Falta el modelo de {what} en {path}. Descargalo con: {DOWNLOAD_HINT}")
        self.what = what
        self.path = path


@dataclass(frozen=True, slots=True)
class VoiceModelPaths:
    """Rutas absolutas de todos los modelos de voz."""

    root: Path
    whisper_dir: Path
    piper_model: Path
    piper_config: Path
    wake_dir: Path
    wake_model: Path
    melspec_model: Path
    embedding_model: Path
    silero_vad: Path


def resolve_model_paths(config: VoiceConfig, project_root: Path) -> VoiceModelPaths:
    """Calcula donde deberia estar cada modelo segun la configuracion.

    No comprueba que existan: eso lo hace cada adaptador al cargar, porque solo
    el adaptador sabe si de verdad lo necesita (con `vad_engine = "energy"`,
    que falte Silero no es un problema).
    """
    root = (
        config.models_dir if config.models_dir.is_absolute() else project_root / config.models_dir
    )
    wake_dir = root / "openwakeword"

    custom_wake = Path(config.wake_model)
    if custom_wake.suffix == ".onnx":
        wake_model = custom_wake if custom_wake.is_absolute() else project_root / custom_wake
    else:
        wake_model = wake_dir / f"{config.wake_model}_v0.1.onnx"

    piper_dir = root / "piper"
    return VoiceModelPaths(
        root=root,
        whisper_dir=root / "whisper" / config.stt_model,
        piper_model=piper_dir / f"{config.tts_voice}.onnx",
        piper_config=piper_dir / f"{config.tts_voice}.onnx.json",
        wake_dir=wake_dir,
        wake_model=wake_model,
        melspec_model=wake_dir / "melspectrogram.onnx",
        embedding_model=wake_dir / "embedding_model.onnx",
        silero_vad=wake_dir / "silero_vad.onnx",
    )


def download_whisper(model: str, target: Path) -> Path:
    """Descarga un modelo de faster-whisper (CTranslate2) a `target`."""
    from faster_whisper.utils import download_model

    target.mkdir(parents=True, exist_ok=True)
    return Path(download_model(model, output_dir=str(target)))


def download_piper_voice(voice: str, target_dir: Path) -> None:
    """Descarga una voz de Piper (el .onnx y su .onnx.json) a `target_dir`."""
    from piper.download_voices import download_voice

    target_dir.mkdir(parents=True, exist_ok=True)
    download_voice(voice, target_dir)


def download_wake_models(model_name: str, target_dir: Path) -> None:
    """Descarga los modelos de openWakeWord: los de rasgos, Silero y la palabra clave.

    Las descargas de openWakeWord bajan tambien la variante .tflite, que en
    Windows no sirve. Se borran al terminar para no dejar peso muerto.
    """
    from openwakeword.utils import download_models

    target_dir.mkdir(parents=True, exist_ok=True)
    download_models(model_names=[model_name], target_directory=str(target_dir))
    for leftover in target_dir.glob("*.tflite"):
        leftover.unlink()
