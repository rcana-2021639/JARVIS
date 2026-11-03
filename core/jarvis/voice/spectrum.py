"""Las 32 bandas de frecuencia que mueven la esfera del HUD.

Cada bloque de audio (el del microfono al escuchar, el de Piper al hablar) se
resume en 32 numeros de 0 a 1: la energia en bandas espaciadas en escala
logaritmica de 80 Hz a 8 kHz, como las oye el oido. El HUD los recibe ~60
veces por segundo y deforma la malla con ellos.

Cuesta ~0.15 ms por bloque (una FFT de 512 puntos): nada frente a los
32 ms que dura el bloque.
"""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["BANDS", "band_levels"]

BANDS = 32
_LOW_HZ = 80.0
_HIGH_HZ = 8000.0
_FLOOR_DB = -72.0
"""Energia por debajo de esto es silencio (0)."""
_CEIL_DB = -12.0
"""Energia por encima de esto satura (1). Una voz normal ronda -30 a -20 dB."""


@cache
def _layout(size: int, sample_rate: int) -> tuple[NDArray[np.float32], NDArray[np.intp]]:
    """La ventana de Hann y a que banda pertenece cada coeficiente de la FFT."""
    window = np.hanning(size).astype(np.float32)
    freqs = np.fft.rfftfreq(size, d=1.0 / sample_rate)
    high = min(_HIGH_HZ, sample_rate / 2.0)
    edges = np.geomspace(_LOW_HZ, high, BANDS + 1)
    owner = np.searchsorted(edges, freqs, side="right") - 1
    owner[(freqs < _LOW_HZ) | (freqs >= high)] = -1
    return window, owner.astype(np.intp)


def band_levels(frame: NDArray[np.float32], sample_rate: int) -> list[float]:
    """32 niveles de 0 a 1 para un bloque mono en float32 (-1..1).

    Una banda sin coeficientes propios (las mas graves con bloques cortos)
    toma el valor de su vecina de arriba: mejor una banda repetida que un
    hueco que se vea como un diente en la esfera.
    """
    samples = np.asarray(frame, dtype=np.float32).ravel()
    if samples.size < 16:
        return [0.0] * BANDS
    window, owner = _layout(samples.size, sample_rate)
    spectrum = np.abs(np.fft.rfft(samples * window)) / (samples.size / 4.0)
    valid = owner >= 0
    sums = np.bincount(owner[valid], weights=spectrum[valid] ** 2, minlength=BANDS)[:BANDS]
    counts = np.bincount(owner[valid], minlength=BANDS)[:BANDS]
    power = np.zeros(BANDS)
    filled = counts > 0
    power[filled] = sums[filled] / counts[filled]
    for index in range(BANDS - 2, -1, -1):
        if not filled[index]:
            power[index] = power[index + 1]
    decibels = 10.0 * np.log10(np.maximum(power, 1e-12))
    levels = np.clip((decibels - _FLOOR_DB) / (_CEIL_DB - _FLOOR_DB), 0.0, 1.0)
    return [round(float(value), 3) for value in levels]
