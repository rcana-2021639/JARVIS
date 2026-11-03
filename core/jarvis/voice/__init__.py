"""Pipeline de voz (Fase 1): palabra clave, escucha, transcripcion y habla.

    audio.py      buffer circular, medidores y conversiones
    engines.py    las interfaces de los seis motores
    vad.py        VAD por energia y detector de fin de frase
    text.py       limpieza de la palabra clave en el texto transcrito
    io.py         fuente y salida sin hardware (pruebas y banco)
    models.py     donde viven los modelos y como se descargan
    factory.py    construye los motores reales desde la configuracion
    runtime.py    estado compartido del pipeline y publicacion al bus
    listening.py  captura, palabra clave, VAD y transcripcion
    speaking.py   sintesis, amplitud y barge-in
    pipeline.py   la fachada que une todo: VoicePipeline
    adapters/     PortAudio, openWakeWord, Silero, faster-whisper y Piper

Uso tipico:

    engines = build_engines(config.voice)
    pipeline = VoicePipeline(bus, engines, lambda: manager.current.voice)
    await pipeline.start()
"""

from jarvis.voice.engines import (
    AudioSink,
    AudioSource,
    SpeechToText,
    TextToSpeech,
    Transcription,
    VoiceActivityDetector,
    VoiceEngines,
    WakeWordDetector,
)
from jarvis.voice.pipeline import VoicePipeline

__all__ = [
    "AudioSink",
    "AudioSource",
    "SpeechToText",
    "TextToSpeech",
    "Transcription",
    "VoiceActivityDetector",
    "VoiceEngines",
    "VoicePipeline",
    "WakeWordDetector",
]
