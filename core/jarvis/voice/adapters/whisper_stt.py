"""Transcripcion local con faster-whisper."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from jarvis.contracts.enums import Language
from jarvis.voice.engines import Transcription
from jarvis.voice.models import ModelMissingError

if TYPE_CHECKING:
    from pathlib import Path

    from jarvis.voice.audio import AudioArray

__all__ = ["WhisperSTT"]

_SUPPORTED = {language.value: language for language in Language}


class WhisperSTT:
    """faster-whisper restringido a espanol e ingles.

    Whisper detecta entre 99 idiomas, y con frases cortas a veces elige
    portugues o gallego para un espanol perfectamente claro. Si el idioma
    detectado no es uno de los que Jarvis habla, se queda con el mas probable de
    los dos y vuelve a transcribir. Pasa poco, y es mejor que un texto en el
    idioma equivocado.
    """

    def __init__(
        self,
        *,
        model_dir: Path,
        device: str,
        compute_type: str,
        beam_size: int,
        prompt: str = "",
        cpu_threads: int = 0,
    ) -> None:
        """Carga el modelo. Tarda unos segundos: se hace una vez al arrancar."""
        from faster_whisper import WhisperModel

        # faster-whisper anuncia en INFO cada audio que procesa. Con parciales
        # son varias lineas por segundo que no dicen nada que no diga ya el bus.
        logging.getLogger("faster_whisper").setLevel(logging.WARNING)

        if not (model_dir / "model.bin").exists():
            raise ModelMissingError("faster-whisper", model_dir)
        self._model = WhisperModel(
            str(model_dir),
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
        )
        self._beam_size = beam_size
        self._prompt = prompt or None

    def transcribe(self, audio: AudioArray, *, language: Language | None) -> Transcription:
        """Transcribe audio mono a 16 kHz."""
        text, confidence, detected, probabilities = self._run(audio, language)
        if language is None and detected not in _SUPPORTED:
            ranked = sorted((prob, code) for code, prob in probabilities if code in _SUPPORTED)
            fallback = _SUPPORTED[ranked[-1][1]] if ranked else Language.ES
            text, confidence, _, _ = self._run(audio, fallback)
            detected = fallback.value
        return Transcription(
            text=text,
            confidence=confidence,
            language=_SUPPORTED.get(detected, Language.ES),
        )

    def _run(
        self,
        audio: AudioArray,
        language: Language | None,
    ) -> tuple[str, float, str, list[tuple[str, float]]]:
        segments, info = self._model.transcribe(
            audio,
            language=language.value if language is not None else None,
            beam_size=self._beam_size,
            initial_prompt=self._prompt,
            condition_on_previous_text=False,
            without_timestamps=True,
            vad_filter=False,
        )
        texts: list[str] = []
        logprobs: list[float] = []
        no_speech: list[float] = []
        for segment in segments:
            texts.append(segment.text)
            logprobs.append(segment.avg_logprob)
            no_speech.append(segment.no_speech_prob)
        text = " ".join(part.strip() for part in texts).strip()
        if logprobs:
            # avg_logprob es el logaritmo de la probabilidad media por token: su
            # exponencial es una confianza razonable en [0, 1]. Se descuenta la
            # probabilidad de que no hubiera habla en absoluto.
            confidence = math.exp(sum(logprobs) / len(logprobs))
            confidence *= 1.0 - max(no_speech)
        else:
            confidence = 0.0
        probabilities = list(info.all_language_probs or [])
        return text, min(max(confidence, 0.0), 1.0), str(info.language), probabilities
