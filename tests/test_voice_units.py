"""Pruebas de las piezas sueltas del pipeline: buffer, medidores, VAD y texto."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from jarvis.config import VoiceConfig, load_config
from jarvis.voice.audio import (
    RingBuffer,
    chunk,
    float_to_pcm16,
    frame_level,
    pcm16_to_float,
    read_wav,
    resample_linear,
    write_wav,
)
from jarvis.voice.models import ModelMissingError, resolve_model_paths
from jarvis.voice.text import strip_wake_word
from jarvis.voice.vad import EndpointDetector, EndpointEvent, EnergyVAD
from tests.voice_fakes import RATE, tone

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Buffer circular
# ---------------------------------------------------------------------------


def test_el_buffer_circular_conserva_solo_lo_ultimo() -> None:
    ring = RingBuffer(seconds=1.0, sample_rate=10)
    ring.append(np.arange(25, dtype=np.float32))
    assert len(ring) == 10
    assert ring.last(1000).tolist() == list(range(15, 25))


def test_el_buffer_circular_da_la_vuelta_sin_perder_el_orden() -> None:
    ring = RingBuffer(seconds=1.0, sample_rate=10)
    for start in range(0, 37, 3):
        ring.append(np.arange(start, start + 3, dtype=np.float32))
    assert ring.last(500).tolist() == [34.0, 35.0, 36.0, 37.0, 38.0]


def test_el_buffer_circular_no_inventa_audio() -> None:
    """Pedir mas de lo que hay devuelve solo lo que hay."""
    ring = RingBuffer(seconds=3.0, sample_rate=RATE)
    ring.append(np.ones(800, dtype=np.float32))
    assert ring.last(500).shape[0] == 800
    ring.clear()
    assert ring.last(500).shape[0] == 0


def test_el_buffer_circular_nunca_crece() -> None:
    ring = RingBuffer(seconds=0.5, sample_rate=RATE)
    for _ in range(100):
        ring.append(np.ones(RATE, dtype=np.float32))
    assert ring.capacity == RATE // 2
    assert len(ring) == RATE // 2


# ---------------------------------------------------------------------------
# Medidores y conversiones
# ---------------------------------------------------------------------------


def test_el_nivel_de_un_seno_es_el_esperado() -> None:
    rms, peak = frame_level(tone(100, 0.5))
    assert rms == pytest.approx(0.5 / np.sqrt(2), rel=0.02)
    assert peak == pytest.approx(0.5, rel=0.02)
    assert frame_level(np.zeros(0, dtype=np.float32)) == (0.0, 0.0)


def test_pcm16_ida_y_vuelta() -> None:
    audio = tone(50, 0.7)
    assert np.allclose(pcm16_to_float(float_to_pcm16(audio)), audio, atol=1e-4)
    assert float_to_pcm16(np.array([2.0, -2.0], dtype=np.float32)).tolist() == [32767, -32767]


def test_remuestrear_conserva_la_duracion() -> None:
    audio = tone(1000, 0.5, rate=22050)
    resampled = resample_linear(audio, 22050, RATE)
    assert resampled.shape[0] == RATE
    assert resampled.dtype == np.float32


def test_partir_en_bloques_rellena_el_ultimo() -> None:
    blocks = list(chunk(np.ones(1100, dtype=np.float32), 512))
    assert [b.shape[0] for b in blocks] == [512, 512, 512]
    assert blocks[-1][76:].sum() == 0


def test_wav_ida_y_vuelta(tmp_path: Path) -> None:
    audio = tone(200, 0.4)
    write_wav(tmp_path / "a.wav", audio, RATE)
    loaded, rate = read_wav(tmp_path / "a.wav")
    assert rate == RATE
    assert np.allclose(loaded, audio, atol=1e-3)


# ---------------------------------------------------------------------------
# VAD por energia y detector de fin de frase
# ---------------------------------------------------------------------------


def test_el_vad_de_energia_distingue_voz_de_silencio() -> None:
    vad = EnergyVAD()
    noise = (np.random.default_rng(0).standard_normal(512) * 0.002).astype(np.float32)
    for _ in range(20):
        quiet = vad.probability(noise)
    loud = vad.probability(tone(32, 0.3))
    assert quiet < 0.2
    assert loud > 0.9


def _detector() -> EndpointDetector:
    return EndpointDetector(threshold=0.5, frame_ms=32, min_speech_ms=96, silence_ms=320)


def test_la_frase_empieza_tras_habla_continua() -> None:
    detector = _detector()
    events = [detector.update(p) for p in [1, 0, 1, 1, 1]]
    assert events == [None, None, None, None, EndpointEvent.STARTED]


def test_la_frase_termina_tras_el_silencio_configurado() -> None:
    detector = _detector()
    events = [detector.update(p) for p in [1, 1, 1] + [0] * 10]
    assert events[2] is EndpointEvent.STARTED
    assert events.index(EndpointEvent.ENDED) == 2 + 10


def test_una_pausa_corta_no_cierra_la_frase() -> None:
    detector = _detector()
    events = [detector.update(p) for p in [1, 1, 1, 0, 0, 0, 1, 1] + [0] * 9]
    assert EndpointEvent.ENDED not in events
    assert detector.speech_ms == pytest.approx(5 * 32)


# ---------------------------------------------------------------------------
# Limpieza de la palabra clave
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("heard", "expected"),
    [
        ("Jarvis, sube el volumen.", "sube el volumen."),
        ("jarvis sube el volumen", "sube el volumen"),
        ("Hey Jarvis, abre Spotify", "abre Spotify"),
        ("Oye, Jarvis: ¿qué hora es?", "¿qué hora es?"),
        ("Yarvis, baja el brillo", "baja el brillo"),
        ("Charvis abre el navegador", "abre el navegador"),
        ("vis, cierra la ventana", "cierra la ventana"),
        ("Jarvis.", ""),
        ("sube el volumen", "sube el volumen"),
        ("dile a Jarvis que me espere", "dile a Jarvis que me espere"),
        ("Hey, abre el correo", "Hey, abre el correo"),
        ("Javier, ven acá", "Javier, ven acá"),
    ],
)
def test_quitar_la_palabra_clave(heard: str, expected: str) -> None:
    assert strip_wake_word(heard, "jarvis") == expected


# ---------------------------------------------------------------------------
# Configuracion y modelos
# ---------------------------------------------------------------------------


def test_la_configuracion_de_voz_trae_lo_que_el_pipeline_necesita() -> None:
    voice = load_config().voice
    assert voice.frame_ms * voice.sample_rate // 1000 == 512
    assert voice.wake_preroll_ms <= voice.ring_buffer_seconds * 1000
    assert voice.vad_engine in {"silero", "energy"}


def test_la_configuracion_rechaza_un_idioma_desconocido() -> None:
    with pytest.raises(ValueError, match="stt_language"):
        VoiceConfig(stt_language="pt")  # type: ignore[arg-type]


def test_las_rutas_de_los_modelos_salen_de_la_configuracion(tmp_path: Path) -> None:
    paths = resolve_model_paths(VoiceConfig(stt_model="tiny", tts_voice="es_ES-x-low"), tmp_path)
    assert paths.whisper_dir == tmp_path / "models" / "whisper" / "tiny"
    assert paths.piper_model.name == "es_ES-x-low.onnx"
    assert paths.wake_model.name == "hey_jarvis_v0.1.onnx"


def test_un_modelo_de_palabra_clave_propio_se_usa_tal_cual(tmp_path: Path) -> None:
    paths = resolve_model_paths(VoiceConfig(wake_model="models/jarvis_es.onnx"), tmp_path)
    assert paths.wake_model == tmp_path / "models" / "jarvis_es.onnx"


def test_el_error_de_modelo_faltante_dice_como_arreglarlo(tmp_path: Path) -> None:
    error = ModelMissingError("faster-whisper", tmp_path / "nada")
    assert "download_voice_models.py" in str(error)
