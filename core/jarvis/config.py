"""Configuracion del nucleo, con jerarquia y recarga en caliente.

La jerarquia, de menor a mayor prioridad:

    1. Los valores por defecto que declara cada seccion en este archivo
    2. `config/jarvis.toml`
    3. Variables de entorno con prefijo `JARVIS_`

El orden importa: los defaults hacen que el sistema arranque sin configurar
nada, el TOML es lo que el usuario edita a diario, y las variables de entorno
existen para ajustes puntuales sin tocar archivos (util en pruebas y al
empaquetar en la Fase 11).

Para secciones anidadas, la variable de entorno usa doble guion bajo:
`JARVIS_VOICE__WAKE_THRESHOLD=0.7`.

DECISION: cero valores magicos incrustados en el codigo. Si un numero gobierna
un comportamiento -- un umbral, un tiempo limite, un tamano de buffer -- vive
aqui y no en el modulo que lo usa. Es la unica forma de afinar el sistema sin
recompilar nada.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from jarvis.contracts.enums import PrivacyMode

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "BridgeConfig",
    "CognitionConfig",
    "GuardianConfig",
    "JarvisConfig",
    "MemoryConfig",
    "ModelPrice",
    "ToolsConfig",
    "UIConfig",
    "VoiceConfig",
    "default_config_path",
    "load_config",
    "project_root",
]


def project_root() -> Path:
    """Raiz del monorepo, deducida desde la ubicacion de este archivo."""
    return Path(__file__).resolve().parents[2]


def default_config_path() -> Path:
    """Ruta por defecto de `config/jarvis.toml`."""
    return project_root() / "config" / "jarvis.toml"


#: Ruta del TOML que leera la proxima construccion de `JarvisConfig`.
#: DECISION: es un global de modulo porque `settings_customise_sources` es un
#: classmethod al que pydantic-settings no deja pasarle argumentos propios.
#: Siempre se fija desde `load_config`, nunca a mano.
_toml_path: Path = default_config_path()


class _Section(BaseModel):
    """Base de las secciones de configuracion."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class VoiceConfig(_Section):
    """Pipeline de voz: captura, palabra clave, transcripcion y sintesis."""

    input_device: str | None = Field(
        default=None,
        description="Nombre del microfono. None usa el dispositivo por defecto del sistema.",
    )
    output_device: str | None = Field(
        default=None,
        description="Nombre de la salida de audio. None usa la salida por defecto del sistema.",
    )
    sample_rate: int = Field(default=16000, gt=0, description="Frecuencia de muestreo en Hz.")
    frame_ms: int = Field(
        default=32,
        gt=0,
        le=200,
        description=(
            "Tamano del bloque de captura. 32ms son 512 muestras a 16kHz: lo que "
            "Silero VAD procesa de un golpe y lo que da el medidor de nivel a ~30Hz."
        ),
    )
    models_dir: Path = Field(
        default=Path("models"),
        description="Donde viven los modelos de voz descargados. Relativa a la raiz.",
    )
    ring_buffer_seconds: float = Field(
        default=3.0,
        gt=0,
        description=(
            "Segundos del buffer circular previo a la palabra clave. Vive en RAM "
            "y se sobrescribe solo: jamas toca disco."
        ),
    )
    wake_word: str = Field(default="jarvis", min_length=1, description="Palabra de activacion.")
    wake_engine: Literal["openwakeword", "whisper", "none"] = Field(
        default="openwakeword",
        description=(
            "Detector de palabra clave. 'openwakeword' es liviano pero solo entiende "
            "'Hey Jarvis' a la inglesa; 'whisper' transcribe cada frase en local y "
            "entiende 'Jarvis' en espanol (~0.7 s mas). 'none' solo para pruebas."
        ),
    )
    wake_phrase_max_s: float = Field(
        default=8.0,
        gt=0.0,
        description=(
            "Con wake_engine='whisper': frase mas larga que se transcribe para "
            "buscar la palabra clave."
        ),
    )
    wake_model: str = Field(
        default="hey_jarvis",
        min_length=1,
        description=(
            "Modelo de openWakeWord: un nombre del catalogo oficial o la ruta a un "
            ".onnx propio. El catalogo trae 'hey_jarvis'; un modelo entrenado solo "
            "con 'Jarvis' se deja en models/ y se apunta aqui."
        ),
    )
    wake_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Umbral de confianza del detector. Subirlo reduce falsos positivos.",
    )
    wake_cooldown_s: float = Field(
        default=1.5,
        ge=0.0,
        description="Enfriamiento tras un disparo para evitar dobles activaciones.",
    )
    wake_preroll_ms: int = Field(
        default=500,
        ge=0,
        description=(
            "Milisegundos ANTERIORES al disparo que se conservan. A veces la orden "
            "empieza pegada a la palabra clave y esas silabas no se pueden perder."
        ),
    )
    vad_engine: Literal["silero", "energy"] = Field(
        default="silero",
        description=(
            "Detector de actividad de voz. 'energy' no necesita modelo pero se "
            "confunde con ruido de fondo; es el respaldo si Silero no carga."
        ),
    )
    vad_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Probabilidad de voz a partir de la cual un bloque cuenta como habla.",
    )
    vad_min_speech_ms: int = Field(
        default=150,
        ge=0,
        description="Habla continua minima para dar por empezada una frase. Filtra golpes.",
    )
    vad_silence_ms: int = Field(
        default=700,
        gt=0,
        description="Silencio que marca el fin de una frase.",
    )
    speech_start_timeout_s: float = Field(
        default=5.0,
        gt=0,
        description=(
            "Si tras la palabra clave nadie habla en este plazo, la escucha se "
            "cierra con una transcripcion vacia. Es un caso normal, no un error."
        ),
    )
    max_listen_s: float = Field(
        default=30.0,
        gt=0,
        description="Tope de escucha continua, con corte elegante.",
    )
    publish_idle_level: bool = Field(
        default=False,
        description=(
            "Publicar el nivel del microfono tambien en DORMIDO. Apagado por "
            "defecto: sin palabra clave no sale nada del pipeline, ni siquiera "
            "cuanto ruido hay en la habitacion."
        ),
    )
    stt_engine: str = Field(default="local", description="Motor de transcripcion: local o cloud.")
    stt_model: str = Field(
        default="base",
        description=(
            "Modelo de faster-whisper. DECISION: 'base' y no 'small' ni "
            "'large-v3-turbo' porque esta maquina no tiene GPU dedicada. Medido en "
            "el i5-10500: small tarda ~2.5 s por frase y base ~0.85 s; con "
            "`stt_prompt` base iguala a small en ordenes cortas. Ver ESTADO.md."
        ),
    )
    stt_device: str = Field(default="cpu", description="cpu o cuda para faster-whisper.")
    stt_compute_type: str = Field(
        default="int8",
        description="Cuantizacion de faster-whisper. int8 es la mas rapida en CPU.",
    )
    stt_beam_size: int = Field(
        default=1,
        gt=0,
        description="Busqueda en haz. 1 es voraz: menos precision, mucha menos latencia.",
    )
    stt_language: Literal["auto", "es", "en"] = Field(
        default="es",
        description=(
            "Idioma fijo, o 'auto' para elegir entre espanol e ingles en cada "
            "frase. DECISION: 'es' por defecto porque detectar el idioma triplica "
            "el costo de la transcripcion en CPU (0.8 s -> 2.5 s con base)."
        ),
    )
    stt_prompt: str = Field(
        default=(
            "Jarvis, sube el volumen. Baja el brillo. Abre el navegador. Pon musica en Spotify."
        ),
        description=(
            "Texto que orienta a Whisper hacia el vocabulario de ordenes. Sin el, "
            "base escribe 'Maja el brillo' y 'Harbys, su el volumen'; con el, acierta. "
            "Vacio lo desactiva."
        ),
    )
    stt_partials: bool = Field(
        default=True,
        description="Transcribir en vivo mientras el usuario habla, para que el HUD muestre texto.",
    )
    stt_partial_interval_ms: int = Field(
        default=700,
        gt=0,
        description=(
            "Cada cuanto se lanza una transcripcion parcial. Si la anterior no "
            "termino, se salta: las parciales nunca se encolan ni frenan la final."
        ),
    )
    tts_engine: str = Field(default="local", description="Motor de sintesis: local o cloud.")
    tts_voice: str = Field(default="es_MX-claude-high", description="Voz de Piper.")
    tts_amplitude_hz: int = Field(
        default=60,
        gt=0,
        le=240,
        description="Frecuencia a la que se publica la amplitud del habla para el HUD.",
    )
    barge_in_enabled: bool = Field(
        default=True,
        description="Si hablar encima de Jarvis lo calla al instante.",
    )


class ModelPrice(_Section):
    """Precio de un modelo de la nube, en dolares por millon de tokens."""

    input_per_mtok: float = Field(ge=0.0)
    output_per_mtok: float = Field(ge=0.0)


def _default_prices() -> dict[str, ModelPrice]:
    # Tarifas publicas de la API de Anthropic (octubre de 2026). Si cambian, se
    # corrigen en jarvis.toml sin tocar codigo.
    return {
        "claude-haiku-5-5": ModelPrice(input_per_mtok=0.10, output_per_mtok=0.50),
        "claude-sonnet-5-5": ModelPrice(input_per_mtok=2.00, output_per_mtok=10.00),
        "claude-opus-5-5": ModelPrice(input_per_mtok=4.00, output_per_mtok=20.00),
    }


class CognitionConfig(_Section):
    """La cascada de tres niveles y su disciplina de costo."""

    privacy_mode: PrivacyMode = Field(
        default=PrivacyMode.HYBRID,
        description="HYBRID normal, LOCAL_ONLY corta la nube por completo.",
    )

    # --- Clasificador -------------------------------------------------------
    classifier_engine: Literal["embeddings", "patterns"] = Field(
        default="embeddings",
        description="patterns se queda solo con regex y difusa; no carga ningun modelo.",
    )
    intent_model_dir: Path = Field(
        default=Path("models/intent"),
        description=(
            "MiniLM multilingue en ONNX. Se baja con scripts/download_cognition_models.py."
        ),
    )
    fuzzy_threshold: float = Field(
        default=86.0,
        ge=0.0,
        le=100.0,
        description="Puntaje minimo de rapidfuzz para aceptar una frase de forma fija.",
    )
    embedding_min_similarity: float = Field(
        default=0.55,
        ge=0.0,
        le=1.0,
        description="Debajo de esto el ejemplo mas parecido no cuenta: se trata como charla.",
    )
    embedding_fast_similarity: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Similitud minima para mandar al nivel 1 algo que solo los embeddings vieron.",
    )
    fast_path_min_confidence: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Confianza del clasificador por debajo de la cual el nivel 1 no se arriesga.",
    )
    max_input_chars: int = Field(
        default=2000,
        gt=0,
        description=(
            "Orden mas larga que se acepta. Lo que pase se rechaza antes de "
            "clasificar: un texto enorme pegado iria entero a la nube y costaria."
        ),
    )
    classifier_timeout_ms: int = Field(
        default=10,
        gt=0,
        description="Meta de tiempo del clasificador. Pasarse se registra como aviso.",
    )

    # --- Nivel 2: modelo local ---------------------------------------------
    ollama_url: str = Field(default="http://127.0.0.1:11434")
    local_model: str = Field(
        default="qwen3:4b",
        description=(
            "Modelo de Ollama del nivel 2. DECISION: 4B en CPU porque no hay GPU "
            "dedicada; la tabla de VRAM del documento maestro manda eso."
        ),
    )
    local_temperature: float = Field(default=0.4, ge=0.0, le=2.0)
    local_max_tokens: int = Field(default=160, gt=0, description="Tope de tokens generados.")
    local_num_ctx: int = Field(default=4096, gt=0, description="Ventana de contexto de Ollama.")
    local_keep_alive: str = Field(
        default="-1",
        description=(
            "Cuanto conserva Ollama el modelo en RAM. -1 es para siempre: la carga "
            "en frio cuesta segundos y arruina la latencia."
        ),
    )
    local_timeout_s: float = Field(
        default=60.0,
        gt=0.0,
        description="Tope de una respuesta completa del nivel 2.",
    )
    local_tools: bool = Field(
        default=False,
        description=(
            "Si el nivel 2 puede llamar herramientas. DECISION: apagado en esta CPU: "
            "qwen3:4b ignora /no_think en ese modo, escribe su razonamiento como texto "
            "y tarda 40+ s. Sin herramientas contesta siempre con esquema."
        ),
    )
    local_confidence_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Debajo de esto, el nivel 2 escala al nivel 3.",
    )
    confidence_logprob_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Peso de la probabilidad de los tokens frente a la autoevaluacion.",
    )
    confidence_stream_margin: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description=(
            "Con confianza temprana >= umbral + margen se habla en streaming. Entre el "
            "umbral y eso, se genera entera y se mide antes de decir nada."
        ),
    )
    fast_step_percent: int = Field(
        default=10,
        gt=0,
        le=100,
        description="Cuanto suben o bajan volumen y brillo sin numero explicito.",
    )
    spoken_max_sentences: int = Field(
        default=2,
        gt=0,
        description="Frases que se dicen en voz alta; el resto va al chat.",
    )

    # --- Nivel 3: API de Claude ---------------------------------------------
    cloud_model_light: str = Field(default="claude-haiku-5-5")
    cloud_model_standard: str = Field(default="claude-sonnet-5-5")
    cloud_model_deep: str = Field(default="claude-opus-5-5")
    cloud_effort: Literal["low", "medium", "high"] = Field(
        default="low",
        description="Esfuerzo de razonamiento para lo hablado. Mas esfuerzo, mas latencia.",
    )
    cloud_max_tokens: int = Field(default=2048, gt=0)
    cloud_timeout_s: float = Field(default=30.0, gt=0.0)
    cloud_retries: int = Field(default=3, ge=0, description="Reintentos ante 429, 5xx o red.")
    cloud_retry_base_s: float = Field(
        default=0.5,
        gt=0.0,
        description="Primera espera; se duplica en cada reintento.",
    )
    cloud_prices: dict[str, ModelPrice] = Field(default_factory=_default_prices)
    daily_budget_usd: float = Field(
        default=0.50,
        ge=0.0,
        description=(
            "Tope de gasto diario. Al alcanzarlo se avisa por voz y se degrada a "
            "solo-local en vez de seguir cobrando. Cero significa sin limite."
        ),
    )
    spend_ledger: Path = Field(
        default=Path("config/data/spend.json"),
        description="Donde se guarda el gasto del dia, para que reiniciar no lo borre.",
    )

    # --- Enrutador y contexto -----------------------------------------------
    max_context_tokens: int = Field(
        default=8000,
        gt=0,
        description="Pasado este contexto, la peticion escala al nivel 3.",
    )
    max_chained_steps: int = Field(
        default=3,
        gt=0,
        description="Mas pasos encadenados que esto, escala al nivel 3.",
    )
    cloud_context_tokens: int = Field(
        default=32000,
        gt=0,
        description="Presupuesto del contexto que ve el nivel 3. La nube aguanta mucho mas.",
    )
    local_context_tokens: int = Field(
        default=1500,
        gt=0,
        description="Presupuesto del contexto que ve el nivel 2. En CPU, cada token cuesta.",
    )
    history_turns: int = Field(
        default=12,
        gt=0,
        description="Turnos que se guardan en RAM. Lo mas viejo se resume en una linea.",
    )
    max_tool_iterations: int = Field(
        default=8,
        gt=0,
        description="Tope del bucle de herramientas para que no se cicle.",
    )
    tool_timeout_s: float = Field(default=10.0, gt=0.0, description="Tope de una herramienta.")


class MemoryConfig(_Section):
    """Las cinco capas de memoria y su consolidacion."""

    data_dir: Path = Field(
        default=Path("config/data"),
        description="Donde viven SQLite y LanceDB. Relativa a la raiz del proyecto.",
    )
    embedding_model: str = Field(
        default="paraphrase-multilingual-MiniLM-L12-v2",
        description=(
            "Modelo de embeddings para la capa semantica. Es el mismo MiniLM del "
            "clasificador: sin GPU, bge-m3 tarda 100-200 ms por consulta."
        ),
    )
    embedding_model_dir: Path = Field(
        default=Path("models/intent"),
        description="Donde estan model.onnx y tokenizer.json del modelo de embeddings.",
    )
    embedding_max_tokens: int = Field(
        default=128,
        gt=0,
        description="Tokens que se codifican de cada hecho. Un hecho cabe en 128.",
    )
    working_turns: int = Field(
        default=20,
        gt=0,
        description="Turnos que conserva la memoria de trabajo en RAM.",
    )
    retrieval_budget_tokens: int = Field(
        default=1500,
        gt=0,
        description="Tope de tokens que la recuperacion puede meter al contexto del nivel 3.",
    )
    local_retrieval_tokens: int = Field(
        default=300,
        gt=0,
        description=(
            "Tope para el nivel 2. Compite con `cognition.local_context_tokens` y en "
            "CPU cada token de contexto se paga en latencia."
        ),
    )
    retrieval_timeout_ms: int = Field(
        default=100,
        gt=0,
        description="Objetivo duro de la recuperacion hibrida de punta a punta.",
    )
    retrieval_candidates: int = Field(
        default=20,
        gt=0,
        description="Candidatos que aporta cada una de las cuatro busquedas antes de fusionar.",
    )
    rrf_k: int = Field(
        default=60,
        gt=0,
        description="Constante de Reciprocal Rank Fusion. 60 es el valor del articulo original.",
    )
    vector_min_similarity: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description="Coseno minimo para que un hecho cuente como parecido a la consulta.",
    )
    duplicate_similarity: float = Field(
        default=0.93,
        ge=0.0,
        le=1.0,
        description="Coseno a partir del cual un hecho nuevo es el mismo que uno guardado.",
    )
    forget_min_similarity: float = Field(
        default=0.55,
        ge=0.0,
        le=1.0,
        description="Coseno minimo para que 'olvida lo de X' incluya un hecho.",
    )
    recency_half_life_hours: float = Field(
        default=72.0,
        gt=0,
        description="Vida media del peso por recencia en la recuperacion.",
    )
    graph_depth: int = Field(
        default=2,
        ge=1,
        le=4,
        description="Saltos que se recorren en el grafo desde una entidad mencionada.",
    )
    remember_importance: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Importancia de lo que el usuario pide recordar explicitamente.",
    )
    consolidation_enabled: bool = Field(
        default=True,
        description="Si la consolidacion nocturna se programa sola.",
    )
    consolidation_hour: int = Field(
        default=4,
        ge=0,
        le=23,
        description="Hora local a la que corre la consolidacion nocturna.",
    )
    consolidation_engine: Literal["local_llm", "rules"] = Field(
        default="local_llm",
        description=(
            "Quien extrae hechos y entidades: el modelo local (Ollama) o solo reglas. "
            "Si Ollama no responde, se cae a reglas y se dice en el reporte."
        ),
    )
    consolidation_batch: int = Field(
        default=8,
        gt=0,
        description="Turnos por llamada al modelo local al consolidar.",
    )
    consolidation_catch_up: bool = Field(
        default=True,
        description="Si al arrancar se consolida lo que quedo pendiente de noches sin correr.",
    )
    consolidation_idle_minutes: float = Field(
        default=2.0,
        ge=0.0,
        description=(
            "La consolidacion cede el modelo local si hubo una orden hace menos de "
            "esto: Ollama atiende una peticion a la vez."
        ),
    )
    session_gap_minutes: float = Field(
        default=30.0,
        gt=0,
        description="Silencio que separa una conversacion de la siguiente.",
    )
    summary_min_turns: int = Field(
        default=6,
        gt=1,
        description="Conversaciones con al menos tantos turnos se resumen al consolidar.",
    )
    decay_half_life_days: float = Field(
        default=30.0,
        gt=0,
        description="Vida media del decaimiento por desuso.",
    )
    routine_promotion_threshold: int = Field(
        default=3,
        gt=0,
        description="Repeticiones exitosas tras las que un patron se vuelve rutina.",
    )
    episodic_compression_days: int = Field(
        default=90,
        gt=0,
        description="Episodios mas viejos que esto se comprimen.",
    )


class ToolsConfig(_Section):
    """Servidores MCP y su ejecucion."""

    enabled_servers: list[str] = Field(
        default_factory=lambda: ["system", "files", "apps", "windows", "media"],
        description="Servidores MCP que se levantan al arrancar.",
    )
    default_timeout_s: float = Field(
        default=30.0,
        gt=0,
        description="Tiempo limite por herramienta.",
    )
    parallel_calls: bool = Field(
        default=True,
        description="Ejecutar en paralelo las herramientas que sean independientes.",
    )
    trash_retention_hours: int = Field(
        default=48,
        gt=0,
        description=(
            "Horas que sobrevive lo que Jarvis borra en su propia papelera. Con "
            "acceso total, los errores son caros y esta es la red de seguridad."
        ),
    )
    screenshot_retention_minutes: int = Field(
        default=10,
        gt=0,
        description="Minutos tras los que se borran las capturas temporales.",
    )
    restart_backoff_s: float = Field(
        default=1.0,
        gt=0,
        description="Retroceso inicial al reiniciar un servidor MCP caido.",
    )


class GuardianConfig(_Section):
    """Permisos, confirmaciones, boveda y auditoria."""

    policy_file: Path = Field(
        default=Path("config/policy.toml"),
        description="Donde vive la clasificacion de niveles por herramienta.",
    )
    confirmation_timeout_s: float = Field(
        default=15.0,
        gt=0,
        description="Sin respuesta en este plazo, una accion ROJA se cancela.",
    )
    undo_stack_size: int = Field(
        default=20,
        gt=0,
        description="Acciones reversibles que se recuerdan.",
    )
    undo_window_s: float = Field(
        default=10.0,
        gt=0,
        description="Ventana en la que una accion AMBAR se puede deshacer.",
    )
    vault_autolock_minutes: int = Field(
        default=30,
        gt=0,
        description="Inactividad tras la que la boveda se bloquea sola.",
    )
    audit_retention_days: int = Field(
        default=365,
        gt=0,
        description="Retencion de la bitacora de auditoria.",
    )


class BridgeConfig(_Section):
    """Puente remoto hacia el telefono (Fase 9)."""

    enabled: bool = Field(default=False, description="Se activa en la Fase 9.")
    host: str = Field(
        default="127.0.0.1",
        description=(
            "REGLA DURA: en produccion escucha SOLO en la interfaz de Tailscale, "
            "nunca en 0.0.0.0. El valor por defecto es la interfaz local."
        ),
    )
    port: int = Field(default=8765, gt=0, le=65535)
    heartbeat_s: float = Field(default=15.0, gt=0, description="Latido de presencia.")
    max_clients: int = Field(default=5, gt=0, description="Dispositivos simultaneos.")


class UIConfig(_Section):
    """HUD de escritorio."""

    layout: str = Field(
        default="full",
        description="Modo de disposicion: full, compact o hidden.",
    )
    always_on_top: bool = Field(default=True)
    show_fps: bool = Field(default=False, description="Contador de fps en desarrollo.")
    particle_count: int = Field(
        default=30000,
        gt=0,
        description=(
            "Puntos de la malla de la esfera. El nivel de detalle adaptativo lo "
            "baja solo si los fotogramas caen."
        ),
    )
    ui_sounds: bool = Field(default=True, description="Sonidos sinteticos de interfaz.")
    hud_enabled: bool = Field(
        default=True,
        description="Si el nucleo abre el WebSocket local que alimenta al HUD (Fase 4).",
    )
    hud_host: str = Field(
        default="127.0.0.1",
        description="REGLA DURA: solo la interfaz local; lo remoto es el puente de la Fase 9.",
    )
    hud_port: int = Field(default=8766, gt=0, le=65535, description="Puerto del WebSocket del HUD.")
    target_fps: int = Field(default=60, gt=0)


class JarvisConfig(BaseSettings):
    """Configuracion completa del nucleo."""

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_prefix="JARVIS_",
        env_nested_delimiter="__",
        extra="forbid",
        validate_assignment=True,
        frozen=True,
    )

    user_name: str = Field(default="Usuario", description="Como se dirige Jarvis al usuario.")
    log_level: str = Field(default="INFO", description="DEBUG, INFO, WARNING o ERROR.")
    development: bool = Field(
        default=True,
        description="Salida de registro legible. En produccion, JSON puro.",
    )
    log_file: Path | None = Field(
        default=Path("logs/jarvis.log"),
        description="Archivo de registro con rotacion. None desactiva el archivo.",
    )
    log_user_content: bool = Field(
        default=False,
        description=(
            "Si el registro guarda lo que dice el usuario. Apagado escribe solo la "
            "longitud: el archivo de registro no esta cifrado."
        ),
    )

    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    cognition: CognitionConfig = Field(default_factory=CognitionConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    guardian: GuardianConfig = Field(default_factory=GuardianConfig)
    bridge: BridgeConfig = Field(default_factory=BridgeConfig)
    ui: UIConfig = Field(default_factory=UIConfig)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Fija la jerarquia. El primero de la tupla es el que manda."""
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            TomlConfigSettingsSource(settings_cls, toml_file=_toml_path),
            file_secret_settings,
        )

    def resolved(self, relative: Path) -> Path:
        """Convierte una ruta relativa de la configuracion en absoluta."""
        return relative if relative.is_absolute() else project_root() / relative

    def to_wire(self) -> dict[str, Any]:
        """Serializa la configuracion para mandarla al HUD."""
        return self.model_dump(mode="json")


def load_config(path: Path | None = None, **overrides: Any) -> JarvisConfig:
    """Carga la configuracion aplicando la jerarquia completa.

    Args:
        path: TOML a leer. Por defecto `config/jarvis.toml`. Si el archivo no
            existe se usan los valores por defecto, sin error: el sistema tiene
            que poder arrancar en una maquina recien instalada.
        **overrides: valores que ganan a todo. Solo para pruebas.

    Returns:
        La configuracion validada e inmutable.
    """
    global _toml_path
    _toml_path = path if path is not None else default_config_path()
    return JarvisConfig(**overrides)


def config_sections() -> Sequence[str]:
    """Nombres de las secciones anidadas. Lo usa el inspector y las pruebas."""
    return ("voice", "cognition", "memory", "tools", "guardian", "bridge", "ui")
