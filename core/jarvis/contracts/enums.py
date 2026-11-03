"""Enumeraciones compartidas por todos los contratos del sistema.

Regla del proyecto: ningun estado viaja jamas como string suelto. Si un campo
solo admite un conjunto cerrado de valores, ese conjunto vive aqui.

DECISION: los miembros conservan el vocabulario en espanol que define el
documento maestro (DORMIDO, VERDE, EPISODICA...) aunque el resto del codigo
use nombres en ingles. Son terminos de dominio: el usuario los escribe a mano
en policy.toml, el HUD los muestra en pantalla y la bitacora de auditoria los
guarda literalmente. Traducirlos obligaria a mantener un diccionario de ida y
vuelta en tres lenguajes distintos sin ganar nada.

DECISION: los valores serializados van sin acentos (AMBAR, no AMBAR con tilde,
EPISODICA, no EPISODICA con tilde). Estos valores terminan en claves de TOML,
en nombres de archivo de la bitacora y en JSON que cruza un WebSocket hacia
Electron y React Native. Un acento ahi es una fuente de errores de codificacion
a cambio de cero beneficio.
"""

from __future__ import annotations

from enum import IntEnum, StrEnum

__all__ = [
    "CascadeLevel",
    "CommandOrigin",
    "CoreState",
    "Emotion",
    "EventType",
    "Language",
    "MemoryKind",
    "PrivacyMode",
    "RiskLevel",
    "SpeechEndReason",
]


class CoreState(StrEnum):
    """Los ocho estados del nucleo. Gobiernan la esfera 3D del HUD (Fase 4)."""

    DORMIDO = "DORMIDO"
    """En reposo. El microfono captura al buffer circular pero nada sale de ahi."""

    DESPIERTO = "DESPIERTO"
    """La palabra clave acaba de dispararse. Transicion breve hacia ESCUCHANDO."""

    ESCUCHANDO = "ESCUCHANDO"
    """Grabando y transcribiendo la orden del usuario."""

    PENSANDO = "PENSANDO"
    """La cascada cognitiva esta resolviendo. Puede durar de 50ms a varios segundos."""

    HABLANDO = "HABLANDO"
    """Sintetizando y reproduciendo la respuesta. Interrumpible."""

    EJECUTANDO = "EJECUTANDO"
    """Corriendo una o varias herramientas."""

    ERROR = "ERROR"
    """Algo fallo. Nunca se llega aqui en silencio: siempre se anuncia."""

    CONFIRMANDO = "CONFIRMANDO"
    """Hay una accion de nivel ROJO detenida esperando confirmacion hablada."""


class RiskLevel(StrEnum):
    """Los cuatro niveles de accion que evalua el Guardian (Fase 8)."""

    VERDE = "VERDE"
    """Ejecuta directo. Solo se registra."""

    AMBAR = "AMBAR"
    """Ejecuta, registra y anuncia en voz. Deshacible durante 10 segundos."""

    ROJO = "ROJO"
    """Suspende y exige confirmacion hablada explicita antes de tocar nada."""

    NEGRO = "NEGRO"
    """Prohibido siempre. No existe frase que lo autorice."""


class SpeechEndReason(StrEnum):
    """Por que se cerro una escucha. Viaja en `voice.speech.ended`."""

    SILENCIO = "SILENCIO"
    """El caso normal: el usuario dejo de hablar y paso `vad_silence_ms`."""

    TOPE = "TOPE"
    """Se alcanzo `max_listen_s`. Corte elegante, se transcribe lo que hubo."""

    SIN_HABLA = "SIN_HABLA"
    """Tras la palabra clave nadie hablo. Se emite una transcripcion vacia."""

    CANCELADA = "CANCELADA"
    """El pipeline se detuvo a media escucha. No se transcribe nada."""


class CascadeLevel(IntEnum):
    """Los tres niveles de la cascada cognitiva.

    Es un IntEnum a proposito: el enrutador compara niveles con `<` y `>` para
    decidir si escala, y el orden numerico es justo la semantica que queremos.
    """

    FAST_PATH = 1
    """Determinista, sin LLM. ~50ms, costo cero. Meta: absorber el 50% de ordenes."""

    LOCAL_LLM = 2
    """Modelo local via Ollama. ~350ms, costo cero."""

    CLOUD_LLM = 3
    """API de Anthropic. ~1.5s, con costo. Solo cuando el nivel inferior no basta."""


class CommandOrigin(StrEnum):
    """De donde llego una orden."""

    VOZ = "VOZ"
    TEXTO = "TEXTO"
    MOVIL = "MOVIL"
    RUTINA = "RUTINA"
    """Disparada por el motor de rutinas, sin intervencion humana directa."""


class MemoryKind(StrEnum):
    """Las cinco capas del sistema de memoria (Fase 3)."""

    TRABAJO = "TRABAJO"
    """Ultimos turnos y estado actual. Vive en RAM, se vacia al reiniciar."""

    EPISODICA = "EPISODICA"
    """Todo lo ocurrido, con marca de tiempo. SQLite + FTS5."""

    SEMANTICA = "SEMANTICA"
    """Hechos destilados sobre el usuario y su mundo. Vectorial en LanceDB."""

    ENTIDAD = "ENTIDAD"
    """Nodos del grafo: personas, proyectos, archivos, apps, sitios, lugares."""

    PROCEDIMIENTO = "PROCEDIMIENTO"
    """Secuencias de acciones que funcionaron y se promovieron a rutina."""


class Language(StrEnum):
    """Idiomas que el pipeline de voz reconoce."""

    ES = "es"
    EN = "en"


class Emotion(StrEnum):
    """Intencion expresiva de una respuesta hablada.

    Deliberadamente corta. La personalidad de Jarvis es seca y precisa: no hay
    entusiasmo ni servilismo que expresar, asi que no hay enum que lo soporte.
    """

    NEUTRAL = "NEUTRAL"
    AFIRMATIVA = "AFIRMATIVA"
    """Confirma que algo se hizo. Breve."""

    ALERTA = "ALERTA"
    """Avisa de algo que el usuario deberia atender."""

    ERROR = "ERROR"
    """Reporta un fallo. Sin drama, sin disculpas largas."""

    INTERROGATIVA = "INTERROGATIVA"
    """Pide una confirmacion o una aclaracion."""


class PrivacyMode(StrEnum):
    """Modos de privacidad. Determinan si la cascada puede llegar al nivel 3."""

    HYBRID = "HYBRID"
    """Local por defecto, nube por excepcion. Es el modo normal."""

    LOCAL_ONLY = "LOCAL_ONLY"
    """Corta la nube por completo. Ni un byte sale de la maquina."""

    CLOUD_PREFERRED = "CLOUD_PREFERRED"
    """Escala agresivamente. Util en maquinas sin GPU dedicada."""


class EventType(StrEnum):
    """Catalogo de eventos del bus.

    Los valores usan espacios de nombres separados por punto para que las
    suscripciones por patron glob sean utiles: `voice.*` recibe todo lo del
    pipeline de voz, `*.error` recibe todos los fallos.

    DECISION: este catalogo cubre lo que la Fase 0 emite mas los eventos que el
    documento maestro nombra explicitamente. Las fases siguientes agregan
    miembros aqui; nadie publica un tipo que no este en esta lista.
    """

    # --- Nucleo y ciclo de vida -------------------------------------------
    CORE_STATE_CHANGED = "core.state.changed"
    CORE_STARTED = "core.started"
    CORE_STOPPING = "core.stopping"
    CORE_ERROR = "core.error"

    # --- Observabilidad ----------------------------------------------------
    TRACE_SPAN = "trace.span"
    """Emitido por el decorador @traced con la duracion de una operacion."""

    BUS_METRICS = "bus.metrics"
    BUS_CONSUMER_LAGGING = "bus.consumer.lagging"
    """Un consumidor se atraso y su cola empezo a descartar eventos."""

    # --- Configuracion -----------------------------------------------------
    CONFIG_RELOADED = "config.reloaded"

    # --- Voz (Fase 1) ------------------------------------------------------
    VOICE_WAKE_DETECTED = "voice.wake.detected"
    VOICE_LEVEL = "voice.level"
    VOICE_SPEECH_STARTED = "voice.speech.started"
    VOICE_SPEECH_ENDED = "voice.speech.ended"
    VOICE_STT_PARTIAL = "voice.stt.partial"
    VOICE_STT_FINAL = "voice.stt.final"
    VOICE_TTS_STARTED = "voice.tts.started"
    VOICE_TTS_AMPLITUDE = "voice.tts.amplitude"
    VOICE_TTS_FINISHED = "voice.tts.finished"
    VOICE_BARGE_IN = "voice.barge_in"

    # --- Cognicion (Fase 2) ------------------------------------------------
    COGNITION_INTENT_CLASSIFIED = "cognition.intent.classified"
    COGNITION_ROUTED = "cognition.routed"
    COGNITION_RESPONSE = "cognition.response"
    COGNITION_RESPONSE_CHUNK = "cognition.response.chunk"
    """Una frase de la respuesta en cuanto existe. La voz la dice sin esperar al resto."""
    COGNITION_COST_UPDATED = "cognition.cost.updated"
    COGNITION_COMPLETED = "cognition.completed"
    """Resumen de una interaccion: niveles probados, tiempos y costo."""

    # --- Memoria (Fase 3) --------------------------------------------------
    MEMORY_RECORD_STORED = "memory.record.stored"
    MEMORY_CONSOLIDATION_FINISHED = "memory.consolidation.finished"

    # --- Herramientas (Fase 6) ---------------------------------------------
    TOOL_CALL_STARTED = "tool.call.started"
    TOOL_CALL_FINISHED = "tool.call.finished"

    # --- Guardian (Fase 8) -------------------------------------------------
    GUARDIAN_AUDIT_ENTRY = "guardian.audit.entry"
    GUARDIAN_CONFIRMATION_REQUIRED = "guardian.confirmation.required"
    GUARDIAN_KILLSWITCH = "guardian.killswitch"
