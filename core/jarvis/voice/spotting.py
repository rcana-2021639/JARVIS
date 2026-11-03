"""Palabra clave por transcripcion: Whisper escucha si la frase empieza con "Jarvis".

El detector de openWakeWord ("hey_jarvis") se entreno con voces inglesas y no
reconoce "Jarvis" dicho en espanol: medido con la voz del usuario por un
microfono Bluetooth, su mejor puntuacion fue 0.22 con umbral 0.5. Este modo
(`wake_engine = "whisper"`) lo reemplaza:

    1. el VAD corta el audio en frases mientras Jarvis duerme
    2. al cerrarse una frase, Whisper la transcribe en local
    3. si empieza con la palabra clave, la interaccion arranca; si no, se descarta

DECISION: la frase "Jarvis, que hora es" dicha de corrido no se transcribe dos
veces. La misma transcripcion que encontro la palabra clave trae la orden, y se
publica directo como `voice.stt.final`.

DECISION (privacidad): en este modo Whisper transcribe en local cualquier frase
que oiga el microfono para saber si empieza con "Jarvis". Nada sale de la
maquina, nada se guarda, y una frase sin la palabra clave no se publica ni se
registra (solo su longitud). Es el mismo trato que openWakeWord da al audio,
aplicado a un texto que muere en el acto.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from jarvis.voice.vad import EndpointDetector, EndpointEvent

if TYPE_CHECKING:
    from jarvis.config import VoiceConfig
    from jarvis.voice.audio import AudioArray, RingBuffer

__all__ = ["PhraseSegmenter"]


class PhraseSegmenter:
    """Corta el audio en frases con el VAD, para transcribirlas una por una."""

    def __init__(self, config: VoiceConfig) -> None:
        """Args: config, la seccion de voz (umbral del VAD, silencios, topes)."""
        self._config = config
        self._endpoint = self._new_endpoint(config)
        self._chunks: list[AudioArray] = []
        self._samples = 0

    @staticmethod
    def _new_endpoint(config: VoiceConfig) -> EndpointDetector:
        return EndpointDetector(
            threshold=config.vad_threshold,
            frame_ms=float(config.frame_ms),
            min_speech_ms=float(config.vad_min_speech_ms),
            silence_ms=float(config.vad_silence_ms),
        )

    def reset(self, config: VoiceConfig) -> None:
        """Descarta la frase a medias y relee la configuracion."""
        self._config = config
        self._endpoint = self._new_endpoint(config)
        self._chunks = []
        self._samples = 0

    def feed(self, frame: AudioArray, probability: float, ring: RingBuffer) -> AudioArray | None:
        """Agrega un bloque; devuelve la frase completa cuando se cierra.

        Args:
            frame: el bloque de audio, ya agregado al buffer circular.
            probability: lo que dijo el VAD de este bloque.
            ring: el buffer circular, para rescatar el comienzo de la frase.
        """
        cfg = self._config
        event = self._endpoint.update(probability)
        if not self._endpoint.started:
            return None
        if event is EndpointEvent.STARTED:
            # El VAD tarda `min_speech_ms` en confirmar; lo previo esta en el
            # buffer circular, junto con un margen para no cortar la "J".
            self._chunks = [ring.last(cfg.vad_min_speech_ms + cfg.wake_preroll_ms)]
        else:
            self._chunks.append(frame)
        self._samples = sum(chunk.shape[0] for chunk in self._chunks)
        too_long = self._samples >= cfg.wake_phrase_max_s * cfg.sample_rate
        if event is EndpointEvent.ENDED or too_long:
            phrase = np.concatenate(self._chunks)
            self.reset(cfg)
            return phrase
        return None
