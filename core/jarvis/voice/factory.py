"""Construye los seis motores de voz a partir de la configuracion.

Es el unico sitio que importa los adaptadores reales. Sus dependencias son el
extra `[voice]` de pyproject y se importan aqui dentro, de modo que el resto
del nucleo (y todas las pruebas) funciona sin tenerlas instaladas.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from jarvis.config import project_root
from jarvis.logging import get_logger
from jarvis.voice.engines import VoiceEngines
from jarvis.voice.models import resolve_model_paths
from jarvis.voice.vad import EnergyVAD

if TYPE_CHECKING:
    from pathlib import Path

    from jarvis.config import VoiceConfig
    from jarvis.voice.audio import AudioArray
    from jarvis.voice.engines import (
        AudioSink,
        AudioSource,
        SpeechToText,
        TextToSpeech,
        VoiceActivityDetector,
        WakeWordDetector,
    )

__all__ = ["NeverWake", "build_engines", "build_stt", "build_tts", "build_vad", "build_wake"]

_log = get_logger(__name__)


class NeverWake:
    """Detector que nunca dispara. Para `wake_engine = "none"`."""

    @property
    def keyword(self) -> str:
        """Nada."""
        return "none"

    def score(self, frame: AudioArray) -> float:
        """Siempre cero."""
        return 0.0

    def reset(self) -> None:
        """Nada que reiniciar."""


def build_wake(config: VoiceConfig, root: Path | None = None) -> WakeWordDetector:
    """El detector de palabra clave configurado."""
    if config.wake_engine == "none":
        return NeverWake()
    from jarvis.voice.adapters.openwakeword_wake import OpenWakeWordDetector

    paths = resolve_model_paths(config, root or project_root())
    return OpenWakeWordDetector(
        model_path=paths.wake_model,
        melspec_path=paths.melspec_model,
        embedding_path=paths.embedding_model,
        keyword=config.wake_model,
    )


def build_vad(config: VoiceConfig, root: Path | None = None) -> VoiceActivityDetector:
    """El VAD configurado. Si Silero no carga, cae a energia y lo dice."""
    if config.vad_engine == "energy":
        return EnergyVAD()
    paths = resolve_model_paths(config, root or project_root())
    try:
        from jarvis.voice.adapters.openwakeword_wake import SileroVAD

        return SileroVAD(model_path=paths.silero_vad)
    except Exception as error:
        _log.warning("voice.vad_fallback", engine="energy", reason=str(error))
        return EnergyVAD()


def build_stt(config: VoiceConfig, root: Path | None = None) -> SpeechToText:
    """El motor de transcripcion configurado."""
    if config.stt_engine != "local":
        msg = f"Motor de transcripcion '{config.stt_engine}' no disponible todavia; usa 'local'."
        raise ValueError(msg)
    from jarvis.voice.adapters.whisper_stt import WhisperSTT

    paths = resolve_model_paths(config, root or project_root())
    return WhisperSTT(
        model_dir=paths.whisper_dir,
        device=config.stt_device,
        compute_type=config.stt_compute_type,
        beam_size=config.stt_beam_size,
        prompt=config.stt_prompt,
    )


def build_tts(config: VoiceConfig, root: Path | None = None) -> TextToSpeech:
    """El motor de sintesis configurado."""
    if config.tts_engine != "local":
        msg = f"Motor de sintesis '{config.tts_engine}' no disponible todavia; usa 'local'."
        raise ValueError(msg)
    from jarvis.voice.adapters.piper_tts import PiperTTS

    paths = resolve_model_paths(config, root or project_root())
    return PiperTTS(model_path=paths.piper_model, config_path=paths.piper_config)


def build_engines(
    config: VoiceConfig,
    *,
    source: AudioSource | None = None,
    sink: AudioSink | None = None,
    root: Path | None = None,
) -> VoiceEngines:
    """Los seis motores. `source` y `sink` permiten cambiar el hardware por archivos.

    Raises:
        ModelMissingError: falta un modelo; el mensaje dice como descargarlo.
        DeviceNotFoundError: no hay microfono o altavoces utilizables.
    """
    if source is None:
        from jarvis.voice.adapters.sounddevice_io import MicrophoneSource

        source = MicrophoneSource(
            device=config.input_device,
            sample_rate=config.sample_rate,
            frame_ms=config.frame_ms,
        )
    if sink is None:
        from jarvis.voice.adapters.sounddevice_io import SpeakerSink

        sink = SpeakerSink(device=config.output_device)

    return VoiceEngines(
        source=source,
        wake=build_wake(config, root),
        vad=build_vad(config, root),
        stt=build_stt(config, root),
        tts=build_tts(config, root),
        sink=sink,
    )
