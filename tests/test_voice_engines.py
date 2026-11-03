"""Pruebas de los motores reales, con los modelos descargados.

Se saltan solas si falta el extra `[voice]` o algun modelo: las pruebas del
resto del nucleo no deben depender de 600 MB de pesos. Para correrlas:

    python scripts/download_voice_models.py
    python -m pytest -m models
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from jarvis.config import load_config, project_root
from jarvis.contracts import Language
from jarvis.voice.audio import chunk, resample_linear
from jarvis.voice.models import resolve_model_paths
from tests.voice_fakes import RATE, tone

pytestmark = pytest.mark.models

_VOICE = load_config().voice
_PATHS = resolve_model_paths(_VOICE, project_root())


def _skip_without(module: str, *files: object) -> None:
    pytest.importorskip(module)
    for path in files:
        if not path.exists():  # type: ignore[attr-defined]
            pytest.skip(f"falta el modelo {path}; corre scripts/download_voice_models.py")


@pytest.fixture(scope="module")
def tts():
    _skip_without("piper", _PATHS.piper_model)
    from jarvis.voice.factory import build_tts

    return build_tts(_VOICE)


@pytest.fixture(scope="module")
def stt():
    _skip_without("faster_whisper", _PATHS.whisper_dir / "model.bin")
    from jarvis.voice.factory import build_stt

    return build_stt(_VOICE)


def _speech(tts, text: str) -> np.ndarray:
    audio = np.concatenate(list(tts.synthesize(text)))
    return resample_linear(audio, tts.sample_rate, RATE)


def test_piper_habla_a_la_frecuencia_que_dice(tts) -> None:
    pieces = list(tts.synthesize("Hola. Soy Jarvis."))
    assert len(pieces) == 2  # una frase por trozo
    assert all(p.dtype == np.float32 for p in pieces)
    assert tts.sample_rate == 22050


def test_whisper_entiende_lo_que_piper_dice(tts, stt) -> None:
    result = stt.transcribe(_speech(tts, "Abre el navegador y busca el clima."), language=None)
    assert "navegador" in result.text.lower()
    assert result.language is Language.ES
    assert result.confidence > 0.5


def test_silero_separa_la_voz_del_silencio(tts) -> None:
    _skip_without("openwakeword", _PATHS.silero_vad)
    from jarvis.voice.adapters.openwakeword_wake import SileroVAD

    vad = SileroVAD(model_path=_PATHS.silero_vad)
    speech = [vad.probability(f) for f in chunk(_speech(tts, "Sube el volumen al maximo."), 512)]
    vad.reset()
    silence = [vad.probability(f) for f in chunk(np.zeros(RATE, dtype=np.float32), 512)]
    assert np.mean(speech) > 0.6
    assert max(silence) < 0.2


def test_la_palabra_clave_no_salta_con_frases_comunes(tts) -> None:
    """Cero falsos positivos sobre ordenes normales sin "Jarvis"."""
    _skip_without("openwakeword", _PATHS.wake_model, _PATHS.melspec_model)
    from jarvis.voice.factory import build_wake

    wake = build_wake(_VOICE)
    for text in ["Abre el navegador.", "Hola, como estas.", "Pon musica tranquila."]:
        scores = [wake.score(f) for f in chunk(_speech(tts, text), 512)]
        wake.reset()
        assert max(scores) < _VOICE.wake_threshold, text


def test_el_remuestreo_en_flujo_no_mete_chasquidos() -> None:
    """44.1 kHz -> 16 kHz bloque a bloque debe parecerse al seno ideal."""
    pytest.importorskip("scipy")
    pytest.importorskip("sounddevice")
    from jarvis.voice.adapters.sounddevice_io import StreamResampler

    source = tone(1000, 0.5, rate=44100)
    resampler = StreamResampler(44100, RATE)
    blocks = list(chunk(source, 1411))
    out = np.concatenate([resampler.process(b) for b in blocks])
    # Sin deriva: la salida tiene exactamente la duracion de la entrada.
    expected = sum(b.shape[0] for b in blocks) * RATE / 44100
    assert abs(out.shape[0] - expected) <= 1

    # Pasado el retardo del filtro, cada muestra debe seguir al seno original.
    delay = (63 - 1) / 2 / 44100
    t = np.arange(out.shape[0]) / RATE - delay
    ideal = 0.5 * np.sin(2 * math.pi * 220.0 * t)
    middle = slice(200, RATE - 400)
    assert np.max(np.abs(out[middle] - ideal[middle])) < 0.02
