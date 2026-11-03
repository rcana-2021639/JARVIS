"""Contratos del pipeline de voz."""

from __future__ import annotations

from datetime import datetime
from typing import Self

from pydantic import Field, model_validator

from jarvis.contracts.base import BaseContract, CorrelationId, utc_now
from jarvis.contracts.enums import Emotion, Language, SpeechEndReason

__all__ = [
    "AudioLevel",
    "SpeechActivity",
    "SpeechOutput",
    "TtsAmplitude",
    "Utterance",
    "WakeDetection",
]


class Utterance(BaseContract):
    """Un fragmento de audio ya transcrito a texto.

    El motor de transcripcion emite muchas Utterance parciales mientras el
    usuario habla (`is_final=False`, que el HUD muestra en gris apareciendo
    palabra por palabra) y exactamente una final cuando el detector de fin de
    habla decide que la frase termino.

    Solo la final alimenta al clasificador de intencion. Las parciales existen
    para que la interfaz se sienta viva y, desde la Fase 7, para la ejecucion
    optimista: empezar a abrir YouTube mientras el usuario aun termina la frase.
    """

    text: str = Field(
        description="Texto transcrito. Puede venir vacio si el usuario no dijo nada.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confianza del motor de transcripcion, de 0 a 1.",
    )
    language: Language = Field(
        default=Language.ES,
        description="Idioma detectado. El pipeline distingue espanol de ingles tecnico.",
    )
    duration_ms: float = Field(
        ge=0.0,
        description="Duracion del audio transcrito en milisegundos.",
    )
    started_at: datetime = Field(
        default_factory=utc_now,
        description="Instante en que empezo el audio de esta transcripcion.",
    )
    ended_at: datetime = Field(
        default_factory=utc_now,
        description="Instante en que termino el audio de esta transcripcion.",
    )
    is_final: bool = Field(
        default=False,
        description="Si es la transcripcion definitiva o una parcial en vivo.",
    )
    correlation_id: CorrelationId = Field(
        description="Interaccion a la que pertenece esta transcripcion.",
    )

    @model_validator(mode="after")
    def _validate_time_range(self) -> Self:
        """El audio no puede terminar antes de empezar."""
        if self.ended_at < self.started_at:
            msg = (
                f"ended_at ({self.ended_at.isoformat()}) es anterior a "
                f"started_at ({self.started_at.isoformat()})"
            )
            raise ValueError(msg)
        return self

    @property
    def is_empty(self) -> bool:
        """Si la transcripcion no trajo texto util.

        Pasa mas seguido de lo que parece: el usuario dice "Jarvis" y se queda
        callado, o el disparo fue un falso positivo. El pipeline debe tratarlo
        como caso normal, no como error.
        """
        return not self.text.strip()


class AudioLevel(BaseContract):
    """Nivel del microfono en un bloque de captura, ~30 veces por segundo.

    Alimenta la esfera del HUD mientras Jarvis escucha. Es un flujo de alta
    frecuencia donde lo reciente vale mas que lo completo: sus consumidores se
    suscriben con `OverflowPolicy.DROP_OLDEST` y una cola pequena.
    """

    rms: float = Field(ge=0.0, le=1.0, description="Energia RMS del bloque, de 0 a 1.")
    peak: float = Field(ge=0.0, le=1.0, description="Pico absoluto del bloque, de 0 a 1.")
    speech_probability: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Probabilidad de voz segun el VAD. None si el VAD no corrio en este bloque.",
    )
    bands: list[float] = Field(
        default_factory=list,
        max_length=64,
        description=(
            "Energia en 32 bandas logaritmicas de 80 Hz a 8 kHz, de 0 a 1 (Fase 4). "
            "Deforma la esfera del HUD. Vacia si no se calculo."
        ),
    )
    correlation_id: CorrelationId | None = Field(
        default=None,
        description="Interaccion en curso. None si se publica en reposo.",
    )


class WakeDetection(BaseContract):
    """La palabra clave se disparo. Es donde nace el id de correlacion."""

    keyword: str = Field(min_length=1, description="Palabra o modelo que disparo.")
    score: float = Field(ge=0.0, le=1.0, description="Confianza del detector en el disparo.")
    preroll_ms: int = Field(
        ge=0,
        description="Audio previo al disparo que se conservo para no perder silabas.",
    )
    barge_in: bool = Field(
        default=False,
        description="Si el disparo interrumpio a Jarvis mientras hablaba.",
    )
    correlation_id: CorrelationId = Field(description="Interaccion que empieza aqui.")


class SpeechActivity(BaseContract):
    """Inicio o fin de una frase segun el detector de actividad de voz."""

    duration_ms: float = Field(
        ge=0.0,
        description="Habla acumulada hasta este punto. Cero en el evento de inicio.",
    )
    reason: SpeechEndReason | None = Field(
        default=None,
        description="Por que se cerro la escucha. Solo en el evento de fin.",
    )
    correlation_id: CorrelationId = Field(description="Interaccion a la que pertenece.")


class SpeechOutput(BaseContract):
    """Ciclo de vida de una respuesta hablada: inicio y fin del TTS."""

    text: str = Field(description="Lo que Jarvis dice o dijo.")
    emotion: Emotion = Field(default=Emotion.NEUTRAL, description="Intencion expresiva.")
    duration_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Audio efectivamente reproducido. Cero en el evento de inicio.",
    )
    first_audio_ms: float | None = Field(
        default=None,
        ge=0.0,
        description="Cuanto tardo en sonar el primer audio desde que se pidio hablar.",
    )
    interrupted: bool = Field(
        default=False,
        description="Si el usuario lo interrumpio antes de terminar.",
    )
    correlation_id: CorrelationId | None = Field(
        default=None,
        description="Interaccion a la que responde. None en avisos del sistema.",
    )


class TtsAmplitude(BaseContract):
    """Amplitud del habla de Jarvis, ~60 veces por segundo, sincronizada con lo que suena."""

    amplitude: float = Field(ge=0.0, le=1.0, description="Envolvente del bloque, de 0 a 1.")
    bands: list[float] = Field(
        default_factory=list,
        max_length=64,
        description=(
            "Energia en 32 bandas logaritmicas de 80 Hz a 8 kHz, de 0 a 1 (Fase 4). "
            "Deforma la esfera del HUD. Vacia si no se calculo."
        ),
    )
    correlation_id: CorrelationId | None = Field(default=None)
