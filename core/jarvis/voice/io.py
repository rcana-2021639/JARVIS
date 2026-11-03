"""Fuentes y salidas de audio que no tocan hardware.

`ArraySource` reproduce un audio ya cargado como si fuera un microfono, y
`MemorySink` guarda lo que Jarvis diria en vez de sonarlo. Son lo que usan las
pruebas y el banco `scripts/voice_bench.py`: el pipeline no distingue entre
ellos y el hardware real.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import numpy as np

from jarvis.voice.audio import chunk

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from jarvis.voice.audio import AudioArray

__all__ = ["ArraySource", "MemorySink"]


class ArraySource:
    """Entrega un audio en bloques, opcionalmente al ritmo del tiempo real."""

    def __init__(
        self,
        audio: AudioArray,
        *,
        sample_rate: int,
        frame_ms: int,
        realtime: bool = False,
        tail_silence_ms: int = 0,
    ) -> None:
        """Prepara la fuente.

        Args:
            audio: el audio a entregar, a `sample_rate`.
            sample_rate: frecuencia del audio.
            frame_ms: duracion de cada bloque.
            realtime: si espera la duracion de cada bloque antes de entregar el
                siguiente. Sin esto el pipeline consume el audio tan rapido como
                puede, que es lo que quieren las pruebas.
            tail_silence_ms: silencio que se agrega al final, para que el
                detector de fin de frase tenga con que cerrar.
        """
        tail = np.zeros(round(tail_silence_ms * sample_rate / 1000), dtype=np.float32)
        self._audio = np.concatenate((audio.astype(np.float32), tail))
        self._sample_rate = sample_rate
        self._frame_samples = round(frame_ms * sample_rate / 1000)
        self._frame_s = frame_ms / 1000.0
        self._realtime = realtime
        self._closed = False
        self.frames_delivered = 0

    @property
    def sample_rate(self) -> int:
        """Frecuencia de los bloques."""
        return self._sample_rate

    async def frames(self) -> AsyncIterator[AudioArray]:
        """Los bloques del audio, y luego termina."""
        loop = asyncio.get_running_loop()
        started = loop.time()
        for index, block in enumerate(chunk(self._audio, self._frame_samples)):
            if self._closed:
                return
            if self._realtime:
                await asyncio.sleep(max(0.0, started + index * self._frame_s - loop.time()))
            else:
                # Cede el bucle en cada bloque, igual que haria un microfono: sin
                # esto las tareas de transcripcion no avanzan hasta el final.
                await asyncio.sleep(0)
            self.frames_delivered += 1
            yield block

    async def close(self) -> None:
        """Corta la entrega."""
        self._closed = True


class MemorySink:
    """Guarda en memoria lo que se reproduciria."""

    def __init__(self, *, realtime: bool = False) -> None:
        """Con `realtime`, cada escritura tarda lo que dura el bloque, como un altavoz."""
        self._realtime = realtime
        self.blocks: list[AudioArray] = []
        self.sample_rate: int | None = None
        self.aborted = 0

    async def write(self, block: AudioArray, sample_rate: int) -> None:
        """Guarda el bloque."""
        self.sample_rate = sample_rate
        self.blocks.append(block)
        if self._realtime:
            await asyncio.sleep(block.shape[0] / sample_rate)
        else:
            await asyncio.sleep(0)

    async def drain(self) -> None:
        """No hay nada pendiente: todo se "reprodujo" al escribirse."""

    async def abort(self) -> None:
        """Cuenta la interrupcion."""
        self.aborted += 1

    async def close(self) -> None:
        """Nada que liberar."""

    @property
    def audio(self) -> AudioArray:
        """Todo lo reproducido, concatenado."""
        if not self.blocks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self.blocks)
