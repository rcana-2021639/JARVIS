"""Primitivas de audio: el buffer circular, medidores y conversiones.

Todo el pipeline trabaja con un unico formato en memoria: `float32` mono en el
rango [-1, 1] a la frecuencia de `voice.sample_rate`. Los adaptadores que
hablan con el hardware o con un modelo que pide otra cosa (openWakeWord y Silero
quieren PCM de 16 bits) convierten en su frontera, nunca en el medio.
"""

from __future__ import annotations

import wave
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

__all__ = [
    "AudioArray",
    "RingBuffer",
    "chunk",
    "float_to_pcm16",
    "frame_level",
    "pcm16_to_float",
    "read_wav",
    "resample_linear",
    "write_wav",
]

AudioArray = npt.NDArray[np.float32]
"""Audio mono en float32 dentro de [-1, 1]. El formato interno del pipeline."""


class RingBuffer:
    """Los ultimos N segundos de audio, en RAM, sobrescribiendose solos.

    Es lo unico que existe mientras Jarvis esta DORMIDO: el microfono escribe
    aqui y nada sale de aqui hasta que la palabra clave se dispara. Entonces se
    extraen los `wake_preroll_ms` previos para no perder la orden cuando viene
    pegada a "Jarvis".

    DECISION: un arreglo de tamano fijo con un indice que da la vuelta, y no un
    `deque` de bloques. Con un deque, extraer "los ultimos 500 ms" obliga a
    concatenar bloques y recortar el primero; aqui es una o dos rebanadas.
    Y la memoria queda reservada desde el arranque: el buffer jamas crece.
    """

    __slots__ = ("_buffer", "_filled", "_write", "sample_rate")

    def __init__(self, seconds: float, sample_rate: int) -> None:
        """Reserva el buffer completo."""
        capacity = max(1, round(seconds * sample_rate))
        self._buffer: AudioArray = np.zeros(capacity, dtype=np.float32)
        self._write = 0
        self._filled = 0
        self.sample_rate = sample_rate

    @property
    def capacity(self) -> int:
        """Muestras que caben."""
        return int(self._buffer.shape[0])

    def __len__(self) -> int:
        """Muestras validas guardadas, hasta la capacidad."""
        return self._filled

    def append(self, samples: AudioArray) -> None:
        """Escribe muestras nuevas, pisando las mas viejas si hace falta."""
        count = int(samples.shape[0])
        if count == 0:
            return
        capacity = self.capacity
        if count >= capacity:
            self._buffer[:] = samples[-capacity:]
            self._write = 0
            self._filled = capacity
            return
        end = self._write + count
        if end <= capacity:
            self._buffer[self._write : end] = samples
        else:
            split = capacity - self._write
            self._buffer[self._write :] = samples[:split]
            self._buffer[: count - split] = samples[split:]
        self._write = end % capacity
        self._filled = min(capacity, self._filled + count)

    def last(self, milliseconds: float) -> AudioArray:
        """Copia de los ultimos `milliseconds` de audio, en orden cronologico."""
        wanted = min(self._filled, round(milliseconds * self.sample_rate / 1000.0))
        if wanted <= 0:
            return np.zeros(0, dtype=np.float32)
        start = self._write - wanted
        if start >= 0:
            return self._buffer[start : self._write].copy()
        return np.concatenate((self._buffer[start:], self._buffer[: self._write]))

    def clear(self) -> None:
        """Olvida todo. Lo usa el silencio del microfono y el killswitch."""
        self._buffer.fill(0.0)
        self._write = 0
        self._filled = 0


def frame_level(samples: AudioArray) -> tuple[float, float]:
    """RMS y pico de un bloque, ambos en [0, 1]."""
    if samples.shape[0] == 0:
        return 0.0, 0.0
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    peak = float(np.max(np.abs(samples)))
    return min(rms, 1.0), min(peak, 1.0)


def pcm16_to_float(samples: npt.NDArray[np.int16]) -> AudioArray:
    """PCM de 16 bits a float32 en [-1, 1]."""
    return (samples.astype(np.float32) / 32768.0).astype(np.float32)


def float_to_pcm16(samples: AudioArray) -> npt.NDArray[np.int16]:
    """float32 en [-1, 1] a PCM de 16 bits, recortando lo que se salga."""
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16)


def resample_linear(samples: AudioArray, source_rate: int, target_rate: int) -> AudioArray:
    """Cambia la frecuencia de muestreo por interpolacion lineal.

    Suficiente para voz en los scripts y pruebas (la voz de Piper sale a 22050 Hz
    y Whisper la quiere a 16000). Los adaptadores de hardware, donde la calidad
    importa en tiempo real, usan un remuestreo polifasico de verdad.
    """
    if source_rate == target_rate or samples.shape[0] == 0:
        return samples.astype(np.float32, copy=False)
    duration = samples.shape[0] / source_rate
    target_count = max(1, round(duration * target_rate))
    source_t = np.arange(samples.shape[0], dtype=np.float64) / source_rate
    target_t = np.arange(target_count, dtype=np.float64) / target_rate
    return np.interp(target_t, source_t, samples).astype(np.float32)


def chunk(samples: AudioArray, size: int) -> Iterator[AudioArray]:
    """Parte un audio en bloques de `size` muestras. El ultimo se rellena con ceros."""
    if size <= 0:
        msg = "El tamano de bloque debe ser positivo"
        raise ValueError(msg)
    for start in range(0, samples.shape[0], size):
        block = samples[start : start + size]
        if block.shape[0] < size:
            block = np.concatenate((block, np.zeros(size - block.shape[0], dtype=np.float32)))
        yield block


def read_wav(path: Path) -> tuple[AudioArray, int]:
    """Lee un WAV PCM de 16 bits, mezclando a mono si trae varios canales."""
    with wave.open(str(path), "rb") as handle:
        if handle.getsampwidth() != 2:
            msg = f"{path}: solo se admite PCM de 16 bits"
            raise ValueError(msg)
        rate = handle.getframerate()
        channels = handle.getnchannels()
        raw = np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)
    if channels > 1:
        raw = raw.reshape(-1, channels).mean(axis=1).astype(np.int16)
    return pcm16_to_float(raw), rate


def write_wav(path: Path, samples: AudioArray, sample_rate: int) -> None:
    """Escribe un WAV mono de 16 bits.

    Solo lo usan los scripts y el banco de pruebas. El pipeline en si nunca
    escribe audio a disco: esa es una de las garantias de la palabra clave.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(float_to_pcm16(samples).tobytes())
