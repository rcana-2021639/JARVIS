# ESTADO DEL PROYECTO JARVIS

Bitácora de fases. Cada sesión lee este archivo antes de escribir una sola línea.

---

## FASE 0 — FUNDACIONES Y CONTRATOS — **COMPLETA**

Fecha: 2026-08-01

### Entregado

**Contratos de datos** — `core/jarvis/contracts/`

| Archivo | Qué contiene | Estado |
|---|---|---|
| `enums.py` | Los 9 enums: `CoreState` (8 estados), `RiskLevel`, `CascadeLevel`, `CommandOrigin`, `MemoryKind`, `Language`, `Emotion`, `PrivacyMode`, `EventType` | Completo |
| `base.py` | `BaseContract` (inmutable, `extra="forbid"`), `utc_now`, `new_correlation_id` | Completo |
| `voice.py` | `Utterance` | Completo |
| `cognition.py` | `Intent`, `Command`, `Response` | Completo |
| `tools.py` | `ToolCall`, `ToolResult` | Completo |
| `memory.py` | `MemoryRecord` con decaimiento exponencial | Completo |
| `system.py` | `Telemetry`, `SystemState` | Completo |
| `audit.py` | `AuditEntry` con encadenamiento de hashes preparado | Completo |
| `events.py` | `Event`, el sobre del bus | Completo |

Los diez contratos que pedía el documento, más `Telemetry` que `SystemState` necesitaba.

**Bus de eventos** — `core/jarvis/bus/`

- `base.py` — `EventBus` abstracto, `OverflowPolicy`, `RetryPolicy`, `Subscription`
- `memory_bus.py` — `InMemoryEventBus`: cola propia por suscriptor, contrapresión, reintentos con retroceso exponencial
- `metrics.py` — contadores globales y por tipo, latencias con p50/p95, rendimiento en ventana móvil

**Configuración** — `core/jarvis/config.py`, `core/jarvis/config_watcher.py`, `config/jarvis.toml`

Siete secciones (`voice`, `cognition`, `memory`, `tools`, `guardian`, `bridge`, `ui`), jerarquía defaults → TOML → entorno, y recarga en caliente con `ConfigManager`.

**Registro** — `core/jarvis/logging.py`

structlog con id de correlación en `ContextVar`, rotación de archivos, consola legible en desarrollo y JSON en producción, decorador `@traced` que mide y publica al bus.

**Scripts y pruebas**

- `scripts/export_schemas.py` — genera 13 archivos en `shared/protocol/`, con modo `--check`
- `scripts/bus_demo.py` — 1000 eventos, 3 consumidores, tabla de latencias
- `tests/` — 118 pruebas en 4 archivos

### Criterios de aceptación

- [x] `mypy core/ --strict` pasa sin errores — 35 archivos, sin incidencias
- [x] `ruff check core/` limpio — verificado también sobre `tests/` y `scripts/`
- [x] `pytest` verde, mínimo 20 pruebas — **118 pruebas**, 1.3 s
- [x] Un script publica 1000 eventos, tres consumidores los reciben todos y se imprimen métricas — `scripts/bus_demo.py`, ~120.000 eventos/s, código de salida 0
- [x] Los esquemas JSON existen en `shared/protocol/` — 13 archivos, verificados al día por `test_schemas.py`
- [x] Cambiar un valor en `jarvis.toml` se refleja sin reiniciar — verificado en vivo (0.5 → 0.77) y en `test_el_vigilante_detecta_un_cambio_real`

### Decisiones tomadas

- **DECISIÓN**: los enums conservan el vocabulario en español del documento (`DORMIDO`, `VERDE`, `EPISODICA`) aunque el resto del código sea inglés, **porque** son términos de dominio que el usuario escribe a mano en `policy.toml`, el HUD muestra en pantalla y la bitácora guarda literalmente. Traducirlos obligaría a mantener un diccionario de ida y vuelta en tres lenguajes.

- **DECISIÓN**: los valores serializados van sin acentos (`AMBAR`, no `ÁMBAR`) **porque** acaban en claves de TOML, nombres de archivo y JSON que cruza un WebSocket hacia Electron y React Native. Hay una prueba que lo verifica (`test_los_valores_de_los_enums_son_ascii`).

- **DECISIÓN**: `Event.payload` es un diccionario serializable y no un modelo genérico `Event[T]`, **porque** un genérico de Pydantic produce un esquema JSON con referencias anidadas que los generadores de tipos de TypeScript manejan mal, y porque el bus reparte tipos heterogéneos por el mismo canal. Se recupera el tipo con `payload_as(Contrato)`, que valida de verdad.

- **DECISIÓN**: `logging.py` no importa el bus; recibe la referencia con `bind_bus()`, **porque** el bus necesita registrar y el registro necesita publicar, y sin inyección eso es un ciclo de imports.

- **DECISIÓN**: los contratos son inmutables (`frozen=True`), **porque** son mensajes, no estado mutable: una vez publicado un evento nadie puede modificarlo por debajo a otro consumidor.

- **DECISIÓN**: `wait_idle(timeout=...)` lleva el plazo en la firma en vez de dejarlo a `asyncio.timeout()` del llamador (lo que ruff sugiere con ASYNC109), **porque** aquí vencer el plazo no es una excepción sino una respuesta válida: "quedaron cosas pendientes", que es justo lo que el cierre del núcleo necesita para decidir si corta igual.

- **DECISIÓN**: el bus no publica un evento de error cuando un manejador falla, **porque** el timeline del HUD va a estar suscrito a `*` y cada error generaría un evento que provocaría otro error, en bucle. El fallo se registra y ahí muere.

- **DECISIÓN**: `stt_model = "small"` y `local_model = "qwen3:4b"` **porque** esta máquina no tiene GPU dedicada (ver más abajo).

### Deuda técnica

- **`_toml_path` es un global de módulo** en `config.py`. `settings_customise_sources` es un classmethod al que pydantic-settings no deja pasarle argumentos propios, así que la ruta del TOML viaja por un global que `load_config` fija justo antes de construir. Funciona y está encapsulado (nadie lo toca fuera de `load_config`), pero no es bonito y no sería seguro si dos hilos cargaran configuraciones distintas a la vez. Hoy nadie lo hace.

- **`_STARTUP_SETTLE_S = 0.25` en `config_watcher.py` es una heurística.** `awatch` monta el observador del sistema de archivos dentro de su primera iteración y no expone ninguna señal de "ya estoy listo". El margen de 250 ms cubre eso en la práctica, y la relectura de puesta al día que hace `start()` cierra el hueco que quedara. Aun así, es un número elegido a ojo.

- **El catálogo de `EventType` incluye eventos de fases futuras** (`voice.*`, `cognition.*`, `tool.*`, `guardian.*`). No se implementó nada de esas fases — solo se reservó el nombre, porque definir el vocabulario compartido es justamente el trabajo de esta fase. Si una fase posterior necesita un evento que no está, se agrega ahí.

- **`config/policy.toml` no existe todavía.** La configuración lo referencia porque el Guardián lo va a leer, pero el archivo es de la Fase 8.

### Nota de hardware — afecta a las fases 1, 2 y 3

La máquina de desarrollo **no tiene GPU dedicada**: Intel UHD 630 integrada y 15.2 GB de RAM. Según la tabla de VRAM del documento maestro, eso cae en la fila *"sin GPU dedicada → Qwen3 4B en CPU + nube agresiva"*. Consecuencias concretas para lo que viene:

- **Fase 1**: `faster-whisper` con `large-v3-turbo` en CPU no llega ni de lejos a la meta de menos de 400 ms de transcripción final. Los valores por defecto ya usan `small`. Contar con apoyarse en Deepgram para acercarse a la meta de latencia, o aceptar y documentar una meta más alta en local.
- **Fase 2**: la distribución 50/35/15 entre niveles va a inclinarse hacia el nivel 3 más de lo previsto. El presupuesto por defecto (`daily_budget_usd = 0.50`) está puesto pensando en eso.
- **Fase 3**: el modelo de embeddings (`bge-m3`) correrá en CPU. El objetivo duro de menos de 100 ms en la recuperación híbrida hay que medirlo antes de darlo por bueno.

Nada de esto invalida el diseño — el documento contempla explícitamente este caso. Pero conviene medir antes de prometer.

### Para la Fase 1 — Pipeline de voz

Lo que ya está listo para usarse y no hay que reconstruir:

- **Publica al bus, no llames a nadie.** `bus.publish(Event.of(EventType.VOICE_WAKE_DETECTED, ...))`. Los tipos de evento del pipeline de voz ya están en el enum `EventType`: `VOICE_WAKE_DETECTED`, `VOICE_LEVEL`, `VOICE_SPEECH_STARTED`, `VOICE_SPEECH_ENDED`, `VOICE_STT_PARTIAL`, `VOICE_STT_FINAL`, `VOICE_TTS_STARTED`, `VOICE_TTS_AMPLITUDE`, `VOICE_TTS_FINISHED`, `VOICE_BARGE_IN`.

- **El medidor de nivel a 30 Hz y la amplitud del TTS a 60 Hz deben usar `OverflowPolicy.DROP_OLDEST`** con cola pequeña. Es exactamente el caso para el que se diseñó: que el HUD se pierda un fotograma de amplitud no le importa a nadie, pero que frene el pipeline sí.

- **`Utterance` ya existe** con `is_final` para distinguir parciales de definitivas, y `is_empty` para el caso de "dijo Jarvis y se quedó callado", que hay que tratar como normal y no como error.

- **Todos los parámetros del pipeline ya están en `config/jarvis.toml`**, sección `[voice]`: `ring_buffer_seconds`, `wake_threshold`, `wake_cooldown_s`, `wake_preroll_ms` (los 500 ms previos al disparo), `vad_silence_ms`, `max_listen_s`. No incrustes números en el código; si falta un parámetro, agrégalo a `VoiceConfig`.

- **Usa `@traced` en cada etapa** (`captura`, `wake`, `vad`, `stt`, `tts`). El banco de pruebas `scripts/voice_bench.py` que pide la Fase 1 puede construirse consumiendo los eventos `TRACE_SPAN` del bus en vez de instrumentar a mano.

- **El id de correlación nace en el wake.** Llama a `set_correlation_id(new_correlation_id())` cuando dispare la palabra clave y todas las líneas de registro y todos los eventos de esa interacción quedarán enlazados solos.

- **Si agregas o cambias un contrato**, corre `python scripts/export_schemas.py`. Hay una prueba que falla si los esquemas quedan desactualizados.

Comandos de verificación de esta fase, por si hay que reconfirmar que la base sigue sana:

```bash
.venv/Scripts/python -m pytest
```

```bash
.venv/Scripts/python -m mypy core/ --strict
```
