"""Las interfaces de los seis motores que el pipeline orquesta.

El pipeline (`pipeline.py`) no sabe que hay faster-whisper, Piper o un
microfono Realtek debajo: solo conoce estos protocolos. Es lo que permite
probarlo entero con audio sintetico y sin hardware, y cambiar un motor (Deepgram
en lugar de Whisper, ElevenLabs en lugar de Piper) sin tocar la orquestacion.

DECISION: los motores de inferencia (`WakeWordDetector`, `VoiceActivityDetector`,
`SpeechToText`, `TextToSpeech`) son SINCRONOS. Todos envuelven bibliotecas de
CPU sincronas, y fingir una interfaz asincrona encima solo esconde donde se
bloquea. El pipeline decide que correr en un hilo con `asyncio.to_thread`; el
motor no tiene que adivinarlo. Los de E/S (`AudioSource`, `AudioSink`) si son
asincronos, porque esperar al hardware es justo lo que no debe frenar al bucle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from jarvis.contracts.enums import Language

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from jarvis.voice.audio import AudioArray

__all__ = [
    "AudioSink",
    "AudioSource",
    "SpeechToText",
    "TextToSpeech",
    "Transcription",
    "VoiceActivityDetector",
    "VoiceEngines",
    "WakeWordDetector",
]


@runtime_checkable
class AudioSource(Protocol):
    """De donde sale el audio: el microfono, un WAV o un arreglo en una prueba."""

    @property
    def sample_rate(self) -> int:
        """Frecuencia de los bloques que entrega. Debe coincidir con `voice.sample_rate`."""
        ...

    def frames(self) -> AsyncIterator[AudioArray]:
        """Bloques de `frame_ms` de duracion, hasta que la fuente se cierre."""
        ...

    async def close(self) -> None:
        """Libera el dispositivo. Termina el iterador de `frames`."""
        ...


@runtime_checkable
class WakeWordDetector(Protocol):
    """Detector de palabra clave. Corre sobre cada bloque mientras Jarvis duerme."""

    @property
    def keyword(self) -> str:
        """Nombre legible de lo que detecta. Viaja en `WakeDetection.keyword`."""
        ...

    def score(self, frame: AudioArray) -> float:
        """Confianza en [0, 1] de que la palabra clave acaba de sonar."""
        ...

    def reset(self) -> None:
        """Olvida el contexto acumulado. Se llama tras cada disparo."""
        ...


@runtime_checkable
class VoiceActivityDetector(Protocol):
    """Detector de actividad de voz. Decide cuando empieza y termina una frase."""

    def probability(self, frame: AudioArray) -> float:
        """Probabilidad en [0, 1] de que el bloque contenga habla."""
        ...

    def reset(self) -> None:
        """Olvida el estado interno. Se llama al empezar cada escucha."""
        ...


@dataclass(frozen=True, slots=True)
class Transcription:
    """Lo que devuelve un motor de transcripcion."""

    text: str
    confidence: float
    language: Language


@runtime_checkable
class SpeechToText(Protocol):
    """Motor de transcripcion."""

    def transcribe(self, audio: AudioArray, *, language: Language | None) -> Transcription:
        """Transcribe audio a `voice.sample_rate`. `language=None` lo deja elegir."""
        ...


@runtime_checkable
class TextToSpeech(Protocol):
    """Motor de sintesis."""

    @property
    def sample_rate(self) -> int:
        """Frecuencia del audio que produce."""
        ...

    def synthesize(self, text: str) -> Iterator[AudioArray]:
        """Audio por trozos, idealmente frase por frase, para que suene antes de terminar."""
        ...


@runtime_checkable
class AudioSink(Protocol):
    """A donde va el habla de Jarvis: los altavoces, o una lista en una prueba."""

    async def write(self, block: AudioArray, sample_rate: int) -> None:
        """Encola un bloque y vuelve cuando el dispositivo tiene sitio para el siguiente.

        Ese ritmo es el que mantiene la amplitud publicada al HUD sincronizada
        con lo que de verdad suena.
        """
        ...

    async def drain(self) -> None:
        """Espera a que termine de sonar todo lo encolado."""
        ...

    async def abort(self) -> None:
        """Calla al instante y descarta lo pendiente. Es el barge-in."""
        ...

    async def close(self) -> None:
        """Libera el dispositivo."""
        ...


@dataclass(frozen=True, slots=True)
class VoiceEngines:
    """Los seis motores juntos, ya construidos. Ver `voice.factory.build_engines`."""

    source: AudioSource
    wake: WakeWordDetector
    vad: VoiceActivityDetector
    stt: SpeechToText
    tts: TextToSpeech
    sink: AudioSink
