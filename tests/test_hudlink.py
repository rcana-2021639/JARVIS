"""El canal del nucleo al HUD (Fase 4): bandas de frecuencia, traduccion y WebSocket."""

from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest
from websockets.asyncio.client import connect

from jarvis.contracts import (
    AudioLevel,
    CascadeLevel,
    CoreState,
    Event,
    EventType,
    PrivacyMode,
    RouteDecision,
    StateChange,
    TtsAmplitude,
)
from jarvis.hudlink import HudServer
from jarvis.hudlink.feed import HudFeed
from jarvis.voice.spectrum import BANDS, band_levels

# ---------------------------------------------------------------------------
# Bandas


@pytest.mark.parametrize(
    ("frequency", "low", "high"),
    [(100, 0, 1), (440, 9, 13), (1000, 15, 19), (3000, 23, 27), (7000, 29, 31)],
)
def test_un_tono_cae_en_su_banda(frequency, low, high):
    sample_rate = 16_000
    t = np.arange(512) / sample_rate
    tone = (0.3 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
    levels = band_levels(tone, sample_rate)
    assert len(levels) == BANDS
    assert low <= int(np.argmax(levels)) <= high


def test_el_silencio_es_cero_y_un_bloque_minimo_no_rompe():
    assert band_levels(np.zeros(512, dtype=np.float32), 16_000) == [0.0] * BANDS
    assert band_levels(np.zeros(4, dtype=np.float32), 16_000) == [0.0] * BANDS


def test_las_bandas_sirven_para_el_tts_a_otra_frecuencia():
    sample_rate = 22_050
    t = np.arange(368) / sample_rate  # ~16 ms: un bloque de amplitud a 60 Hz
    tone = (0.3 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    levels = band_levels(tone, sample_rate)
    assert 15 <= int(np.argmax(levels)) <= 19
    assert all(0.0 <= value <= 1.0 for value in levels)


# ---------------------------------------------------------------------------
# Traduccion del bus


async def _publish(bus, event_type, payload) -> None:
    await bus.publish(Event.of(event_type, payload, source="prueba"))
    await bus.wait_idle(timeout=2.0)


async def test_el_feed_traduce_estado_audio_y_carga(bus):
    sent: list[dict[str, object]] = []
    feed = HudFeed(bus, sent.append)
    feed.start()
    await _publish(
        bus,
        EventType.CORE_STATE_CHANGED,
        StateChange(
            previous=CoreState.DORMIDO, current=CoreState.PENSANDO, reason="orden", source="prueba"
        ),
    )
    bands = [0.5] * BANDS
    await _publish(bus, EventType.VOICE_LEVEL, AudioLevel(rms=0.1, peak=0.3, bands=bands))
    await _publish(bus, EventType.VOICE_TTS_AMPLITUDE, TtsAmplitude(amplitude=0.7, bands=bands))
    await _publish(
        bus,
        EventType.COGNITION_ROUTED,
        RouteDecision(
            level=CascadeLevel.CLOUD_LLM,
            reason="codigo",
            intent="code.write",
            attempt=1,
            privacy_mode=PrivacyMode.HYBRID,
            correlation_id="c",
        ),
    )
    await feed.stop()

    assert sent[0] == {"type": "state", "state": "PENSANDO", "previous": "DORMIDO",
                       "reason": "orden"}  # fmt: skip
    assert sent[1] == {"type": "cognition", "load": 0.5, "level": 0}
    assert sent[2] == {"type": "audio", "source": "mic", "level": 0.4, "bands": bands}
    assert sent[3] == {"type": "audio", "source": "tts", "level": 0.7, "bands": bands}
    assert sent[4] == {"type": "cognition", "load": 1.0, "level": 3}
    assert feed.hello()["state"] == "PENSANDO"


# ---------------------------------------------------------------------------
# Servidor


async def test_el_hud_recibe_el_saludo_y_los_cambios_por_websocket(bus):
    server = HudServer(bus, port=0)
    await server.start()
    try:
        async with connect(f"ws://127.0.0.1:{server.port}") as client:
            hello = json.loads(await asyncio.wait_for(client.recv(), 2))
            assert hello == {"type": "hello", "version": 1, "bands": BANDS, "state": "DORMIDO"}
            await _publish(
                bus,
                EventType.CORE_STATE_CHANGED,
                StateChange(
                    previous=CoreState.DORMIDO,
                    current=CoreState.ESCUCHANDO,
                    reason="wake",
                    source="prueba",
                ),
            )
            message = json.loads(await asyncio.wait_for(client.recv(), 2))
            assert message["type"] == "state" and message["state"] == "ESCUCHANDO"
            assert server.clients == 1
    finally:
        await server.stop()


async def test_un_hud_atrasado_pierde_lo_viejo_y_no_frena_nada(bus):
    server = HudServer(bus, port=0)
    await server.start()
    try:
        async with connect(f"ws://127.0.0.1:{server.port}") as client:
            await asyncio.wait_for(client.recv(), 2)
            # 500 mensajes de golpe, sin ceder el bucle: la cola del cliente (64)
            # se llena y descarta lo mas viejo. Publicar nunca espera al HUD.
            for index in range(500):
                server.feed._send({"type": "cognition", "load": index / 500, "level": 0})
            loads: list[float] = []
            with pytest.raises(TimeoutError):
                while True:
                    loads.append(json.loads(await asyncio.wait_for(client.recv(), 0.3))["load"])
            assert 0 < len(loads) <= 64
            assert loads[-1] == pytest.approx(499 / 500)  # lo ultimo siempre llega
    finally:
        await server.stop()


def test_el_hud_nunca_se_sirve_fuera_de_la_interfaz_local(bus):
    with pytest.raises(ValueError, match="interfaz local"):
        HudServer(bus, host="0.0.0.0")
