"""Diagnostico del microfono y de la palabra clave, en vivo.

Muestra cada medio segundo el nivel del microfono y la puntuacion mas alta del
detector de palabra clave. Sirve para saber si el problema es que no llega
audio, que llega muy bajo, o que el detector no reconoce la pronunciacion.

Uso:

    python scripts/mic_check.py               # 30 segundos
    python scripts/mic_check.py --seconds 60
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))

from jarvis.config import load_config
from jarvis.voice.adapters.sounddevice_io import MicrophoneSource
from jarvis.voice.audio import frame_level
from jarvis.voice.factory import build_wake


async def run(seconds: float) -> int:
    voice = load_config().voice
    wake = build_wake(voice)
    source = MicrophoneSource(
        device=voice.input_device, sample_rate=voice.sample_rate, frame_ms=voice.frame_ms
    )
    print(f"Microfono: {getattr(source, 'device_name', '?')}")
    print(f'Umbral de la palabra clave: {voice.wake_threshold}. Di "Hey Jarvis" varias veces.\n')
    print("  seg   nivel (pico)                         palabra clave")
    started = time.monotonic()
    window_peak = 0.0
    window_score = 0.0
    best = 0.0
    last = started
    async for frame in source.frames():
        _, peak = frame_level(frame)
        score = float(wake.score(frame))
        window_peak = max(window_peak, peak)
        window_score = max(window_score, score)
        best = max(best, score)
        now = time.monotonic()
        if now - last >= 0.5:
            bar = "#" * int(min(1.0, window_peak) * 30)
            hit = "  <-- DISPARA" if window_score >= voice.wake_threshold else ""
            print(f"{now - started:5.1f}   {window_peak:.3f} {bar:30} {window_score:.3f}{hit}")
            window_peak = window_score = 0.0
            last = now
        if now - started >= seconds:
            break
    await source.close()
    print(f"\nPuntuacion maxima de la palabra clave: {best:.3f} (umbral {voice.wake_threshold})")
    return 0


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seconds", type=float, default=30.0)
    args = parser.parse_args()
    return asyncio.run(run(args.seconds))


if __name__ == "__main__":
    sys.exit(main())
