"""Microfono y altavoces reales, via PortAudio (sounddevice).

Dos realidades de Windows que este modulo absorbe para que el pipeline no se
entere:

1. **Puede no haber microfono por defecto.** Si Windows no tiene entrada
   predeterminada (pasa con el conector analogico de Realtek cuando el
   microfono esta deshabilitado en Configuracion de sonido), PortAudio solo lo
   expone por WDM-KS. Se busca la primera entrada disponible en cualquier API
   antes de rendirse.
2. **El dispositivo casi nunca acepta 16 kHz.** WDM-KS y WASAPI exclusivo solo
   abren a la frecuencia nativa (44.1 o 48 kHz). Se abre a la nativa y se
   remuestrea en flujo, con filtro antialias y fase continua entre bloques.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
from typing import TYPE_CHECKING, Any

import numpy as np
import sounddevice as sd
from scipy.signal import firwin, lfilter, lfilter_zi

from jarvis.logging import get_logger
from jarvis.voice.audio import resample_linear

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from jarvis.voice.audio import AudioArray

__all__ = [
    "CaptureStalledError",
    "DeviceNotFoundError",
    "MicrophoneSource",
    "SpeakerSink",
    "StreamResampler",
    "describe_devices",
    "resolve_device",
]

_log = get_logger(__name__)

#: Bloques de captura que pueden esperar en cola. A 32 ms son ~2 s: si el
#: pipeline se atrasa mas que eso, lo viejo se descarta en vez de acumular
#: retraso para siempre.
_CAPTURE_QUEUE_FRAMES = 64

#: Coeficientes del filtro antialias del remuestreo. 63 es de sobra para voz.
_RESAMPLER_TAPS = 63

#: Si el microfono pasa este tiempo sin entregar un solo bloque, esta muerto.
#: Pasa de verdad: en Windows, un conector Realtek deshabilitado se abre por
#: WDM-KS sin error y despues nunca llama al callback.
_CAPTURE_STALL_S = 3.0


class DeviceNotFoundError(RuntimeError):
    """No hay dispositivo de audio que sirva. El mensaje dice que hacer."""


class CaptureStalledError(RuntimeError):
    """El microfono se abrio pero dejo de entregar audio (o nunca empezo)."""


def describe_devices() -> str:
    """Lista legible de los dispositivos, para el mensaje de error y el diagnostico."""
    hostapis = sd.query_hostapis()
    lines = []
    for index, device in enumerate(sd.query_devices()):
        api = hostapis[device["hostapi"]]["name"]
        lines.append(
            f"  {index:>2}  [{api}] {device['name']}  "
            f"(entradas {device['max_input_channels']}, salidas {device['max_output_channels']}, "
            f"{int(device['default_samplerate'])} Hz)",
        )
    return "\n".join(lines)


def resolve_device(name: str | None, *, kind: str) -> int:
    """Encuentra el indice de un dispositivo de entrada (`kind="input"`) o salida.

    Con nombre, el primero cuyo nombre lo contenga (sin distinguir mayusculas).
    Sin nombre, el predeterminado de Windows y, si no hay, el primero que exista
    en cualquier API.
    """
    channels_key = "max_input_channels" if kind == "input" else "max_output_channels"
    devices = list(enumerate(sd.query_devices()))
    candidates = [(i, d) for i, d in devices if d[channels_key] > 0]

    if name:
        wanted = name.lower()
        matches = [(i, d) for i, d in candidates if wanted in d["name"].lower()]
        if matches:
            # El mismo aparato aparece una vez por API de audio. DECISION: se
            # prefiere WASAPI porque entrega la frecuencia nativa (16 kHz en los
            # auriculares Bluetooth) sin el remuestreo ni el bufer extra de MME.
            hostapis = sd.query_hostapis()
            for index, device in matches:
                if "WASAPI" in hostapis[device["hostapi"]]["name"]:
                    return index
            return matches[0][0]
        msg = (
            f"No hay dispositivo de {kind} que contenga '{name}'. "
            f"Disponibles:\n{describe_devices()}"
        )
        raise DeviceNotFoundError(msg)

    default_in, default_out = sd.default.device
    default = default_in if kind == "input" else default_out
    if default is not None and default >= 0:
        return int(default)
    if candidates:
        index, device = candidates[0]
        _log.warning(
            "audio.no_default_device",
            kind=kind,
            using=device["name"],
            hint="Windows no tiene dispositivo predeterminado; se usa el primero disponible.",
        )
        return index
    hint = (
        "Conecta un microfono o habilitalo en Configuracion > Sistema > Sonido."
        if kind == "input"
        else "Conecta unos altavoces o audifonos."
    )
    msg = f"No hay ningun dispositivo de {kind}. {hint}\n{describe_devices()}"
    raise DeviceNotFoundError(msg)


def _supported_rate(device: int, rate: int, *, kind: str) -> bool:
    check = sd.check_input_settings if kind == "input" else sd.check_output_settings
    try:
        check(device=device, samplerate=rate, channels=1, dtype="float32")
    except sd.PortAudioError:
        return False
    return True


class StreamResampler:
    """Remuestreo en flujo: bloque a bloque, sin chasquidos en las uniones.

    Filtra con un FIR antialias que conserva su estado entre bloques, e
    interpola linealmente llevando la fase fraccionaria de un bloque al
    siguiente. Remuestrear cada bloque por separado meteria una discontinuidad
    cada 32 ms, que Whisper oye como ruido.
    """

    def __init__(self, source_rate: int, target_rate: int) -> None:
        """Prepara el filtro para pasar de `source_rate` a `target_rate`."""
        self._step = source_rate / target_rate
        self._taps: AudioArray | None = None
        self._zi: Any = None
        if source_rate > target_rate:
            cutoff = 0.45 * target_rate
            self._taps = firwin(_RESAMPLER_TAPS, cutoff, fs=source_rate).astype(np.float64)
            self._zi = lfilter_zi(self._taps, 1.0) * 0.0
        self._position = 0.0
        self._previous = np.zeros(1, dtype=np.float64)

    def process(self, block: AudioArray) -> AudioArray:
        """Remuestrea un bloque. La longitud de salida varia en una muestra arriba o abajo."""
        samples = block.astype(np.float64)
        if self._taps is not None:
            samples, self._zi = lfilter(self._taps, 1.0, samples, zi=self._zi)
        buffer = np.concatenate((self._previous, samples))
        last = buffer.shape[0] - 1
        if last < self._position:
            self._position -= samples.shape[0]
            self._previous = buffer[-1:]
            return np.zeros(0, dtype=np.float32)
        count = math.floor((last - self._position) / self._step) + 1
        positions = self._position + np.arange(count) * self._step
        out = np.interp(positions, np.arange(buffer.shape[0]), buffer)
        self._position = positions[-1] + self._step - last
        self._previous = buffer[-1:]
        return out.astype(np.float32)


class MicrophoneSource:
    """El microfono como `AudioSource`: bloques exactos de `frame_ms` a `sample_rate`."""

    def __init__(self, *, device: str | None, sample_rate: int, frame_ms: int) -> None:
        """Resuelve el dispositivo. No lo abre hasta que alguien pida `frames`."""
        self._device = resolve_device(device, kind="input")
        self._sample_rate = sample_rate
        self._frame_samples = round(frame_ms * sample_rate / 1000)
        info = sd.query_devices(self._device)
        self.device_name: str = info["name"]
        if _supported_rate(self._device, sample_rate, kind="input"):
            self._device_rate = sample_rate
        else:
            self._device_rate = int(info["default_samplerate"])
        self._queue: asyncio.Queue[AudioArray | None] | None = None
        self._closed = False
        self.dropped_frames = 0

    @property
    def sample_rate(self) -> int:
        """Frecuencia de los bloques entregados (no la del dispositivo)."""
        return self._sample_rate

    async def frames(self) -> AsyncIterator[AudioArray]:
        """Abre el microfono y entrega bloques hasta `close`."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[AudioArray | None] = asyncio.Queue(_CAPTURE_QUEUE_FRAMES)
        self._queue = queue
        resampler = (
            StreamResampler(self._device_rate, self._sample_rate)
            if self._device_rate != self._sample_rate
            else None
        )
        device_block = round(self._frame_samples * self._device_rate / self._sample_rate)

        def push(block: AudioArray) -> None:
            if queue.full():
                with contextlib.suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
                self.dropped_frames += 1
                if self.dropped_frames % 50 == 1:
                    _log.warning("audio.capture_lagging", dropped=self.dropped_frames)
            queue.put_nowait(block)

        def callback(indata: Any, _frames: int, _time: Any, status: Any) -> None:
            if status:
                _log.debug("audio.capture_status", status=str(status))
            loop.call_soon_threadsafe(push, indata[:, 0].astype(np.float32))

        stream = sd.InputStream(
            device=self._device,
            channels=1,
            samplerate=self._device_rate,
            blocksize=device_block,
            dtype="float32",
            callback=callback,
        )
        _log.info(
            "audio.mic_open",
            device=self.device_name,
            device_rate=self._device_rate,
            rate=self._sample_rate,
        )
        pending = np.zeros(0, dtype=np.float32)
        with stream:
            while not self._closed:
                try:
                    raw = await asyncio.wait_for(queue.get(), _CAPTURE_STALL_S)
                except TimeoutError:
                    msg = (
                        f"El microfono '{self.device_name}' no entrega audio desde hace "
                        f"{_CAPTURE_STALL_S:.0f} s. Si es un conector analogico, revisa que "
                        "haya algo enchufado y que este habilitado en Configuracion > "
                        "Sistema > Sonido; o elige otro con voice.input_device."
                    )
                    raise CaptureStalledError(msg) from None
                if raw is None:
                    break
                block = resampler.process(raw) if resampler is not None else raw
                pending = np.concatenate((pending, block))
                while pending.shape[0] >= self._frame_samples:
                    yield pending[: self._frame_samples]
                    pending = pending[self._frame_samples :]

    async def close(self) -> None:
        """Cierra el microfono."""
        self._closed = True
        if self._queue is not None:
            with contextlib.suppress(asyncio.QueueFull):
                self._queue.put_nowait(None)


class SpeakerSink:
    """Los altavoces como `AudioSink`, con escritura bloqueante que marca el ritmo."""

    def __init__(self, *, device: str | None) -> None:
        """Resuelve el dispositivo. Lo abre en la primera escritura, a la frecuencia del TTS."""
        self._device = resolve_device(device, kind="output")
        self.device_name: str = sd.query_devices(self._device)["name"]
        self._stream: Any = None
        self._stream_rate = 0
        self._source_rate = 0
        self._aborting = False

    def _open(self, sample_rate: int) -> None:
        if self._stream is not None:
            self._stream.close()
        rate = sample_rate
        if not _supported_rate(self._device, sample_rate, kind="output"):
            rate = int(sd.query_devices(self._device)["default_samplerate"])
        self._stream = sd.OutputStream(
            device=self._device,
            channels=1,
            samplerate=rate,
            dtype="float32",
            latency="low",
        )
        self._stream.start()
        self._stream_rate = rate
        self._source_rate = sample_rate
        _log.info("audio.speaker_open", device=self.device_name, rate=rate)

    async def write(self, block: AudioArray, sample_rate: int) -> None:
        """Escribe un bloque; vuelve cuando el dispositivo lo acepto."""
        self._aborting = False
        if self._stream is None or self._source_rate != sample_rate:
            await asyncio.to_thread(self._open, sample_rate)
        if not self._stream.active:
            self._stream.start()
        data = block
        if self._stream_rate != sample_rate:
            data = resample_linear(block, sample_rate, self._stream_rate)
        try:
            await asyncio.to_thread(self._stream.write, data.reshape(-1, 1))
        except sd.PortAudioError:
            if not self._aborting:
                raise

    async def drain(self) -> None:
        """Espera a que suene lo pendiente (`stop` de PortAudio drena antes de parar)."""
        if self._stream is not None and self._stream.active:
            await asyncio.to_thread(self._stream.stop)

    async def abort(self) -> None:
        """Calla al instante, descartando lo que quedaba en el buffer."""
        self._aborting = True
        if self._stream is not None and self._stream.active:
            await asyncio.to_thread(self._stream.abort)

    async def close(self) -> None:
        """Cierra el dispositivo."""
        if self._stream is not None:
            stream, self._stream = self._stream, None
            await asyncio.to_thread(stream.close)
