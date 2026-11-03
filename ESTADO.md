# ESTADO DEL PROYECTO JARVIS

Bitácora de fases. Cada sesión lee este archivo antes de escribir una sola línea.

---

## FASE 1 — PIPELINE DE VOZ — **COMPLETA** (con dos metas que este hardware no alcanza)

Fecha: 2026-10-26

### Entregado

**Pipeline** — `core/jarvis/voice/`

| Archivo | Qué contiene |
|---|---|
| `pipeline.py` | `VoicePipeline`: la máquina de estados DORMIDO → DESPIERTO → ESCUCHANDO → (final) → DORMIDO, y HABLANDO con barge-in. Publica todo al bus |
| `engines.py` | Los seis protocolos: `AudioSource`, `WakeWordDetector`, `VoiceActivityDetector`, `SpeechToText`, `TextToSpeech`, `AudioSink` |
| `audio.py` | `RingBuffer` (tamaño fijo, solo RAM), nivel RMS/pico, conversiones PCM, WAV para scripts |
| `vad.py` | `EnergyVAD` (respaldo sin modelo) y `EndpointDetector` (inicio tras `vad_min_speech_ms`, fin tras `vad_silence_ms`) |
| `text.py` | `strip_wake_word`: quita "Jarvis", "Hey Jarvis", "Yarvis", "...vis" del principio del texto |
| `io.py` | `ArraySource` y `MemorySink`: el pipeline completo sin hardware |
| `models.py` | Rutas de los modelos y su descarga |
| `factory.py` | Construye los motores reales desde `VoiceConfig` |
| `adapters/` | PortAudio (micrófono y altavoces, con remuestreo en flujo), openWakeWord, Silero VAD, faster-whisper, Piper |

**Contratos nuevos** — `AudioLevel`, `WakeDetection`, `SpeechActivity`, `SpeechOutput`, `TtsAmplitude` (en `voice.py`), `StateChange` (en `system.py`) y el enum `SpeechEndReason`. Exportados a `shared/protocol/` (19 archivos).

**Configuración** — `VoiceConfig` pasó de 13 a 33 campos, todos documentados en `config/jarvis.toml`. Nada de números incrustados.

**Scripts**

- `scripts/download_voice_models.py` — baja Whisper, la voz de Piper y openWakeWord a `models/` (no versionado)
- `scripts/voice_bench.py` — el banco de la fase: Piper dice órdenes, el pipeline completo las transcribe y se miden latencias desde los eventos del bus
- `scripts/voice_live.py` — la demostración con micrófono: "Hey Jarvis" + orden, y Jarvis repite lo que oyó

**Pruebas** — 181 en total (63 nuevas). `test_voice_pipeline.py` prueba la orquestación con motores falsos deterministas (`tests/voice_fakes.py`); `test_voice_engines.py` prueba los motores reales y se salta sola si faltan los modelos.

### Mediciones — i5-10500 sin GPU, `scripts/voice_bench.py` en tiempo real

| Medida | Resultado | Meta | |
|---|---|---|---|
| Transcripción final (fin de frase → texto), p50 | **886 ms** (p95 1156) | < 400 ms | ❌ |
| Precisión (WER medio, 6 órdenes) | 0.17 | — | errores: "70%" por "setenta por ciento", "Recuerda me" |
| Palabra clave, costo por bloque de 32 ms, p95 | 6.6 ms (20 %) | < 32 ms | ✅ |
| Silero VAD, costo por bloque, p95 | 1.8 ms (6 %) | < 32 ms | ✅ |
| TTS, tiempo hasta el primer audio | 40 / 107 / 166 ms (1, 5 y 9 palabras) | — | ✅ |
| Detector real "Hey Jarvis" (voz inglesa) | 6 de 6, cero falsos positivos con frases en español | — | ✅ |

Cómo se llegó a `base`: medido, `small` tarda 2.5 s por frase, `base` 0.85 s y `tiny` 0.4 s con errores graves. Detectar el idioma automáticamente **triplica** el costo (otra pasada del codificador), por eso `stt_language = "es"`. Y un `stt_prompt` con vocabulario de órdenes hace que `base` acierte lo mismo que `small`: sin él escribe "Maja el brillo" y "Harbys, su el volumen".

### Criterios de aceptación

- [x] `mypy core/ --strict` limpio — 48 archivos
- [x] `ruff check core/ tests/ scripts/` limpio
- [x] `pytest` verde — 181 pruebas, ~19 s
- [x] La palabra clave es sagrada: sin ella no sale ni un evento de voz, ni se llama al VAD ni a Whisper (`test_sin_palabra_clave_no_sale_nada`)
- [x] Preroll de 500 ms rescatado del buffer circular (`test_el_preroll_rescata_el_audio_previo_al_disparo`)
- [x] "Jarvis" y silencio → transcripción vacía, no error (`test_jarvis_y_silencio_da_una_transcripcion_vacia`)
- [x] Nivel a ~30 Hz y amplitud del TTS a 60 Hz, sincronizada con lo que suena
- [x] Barge-in: la palabra clave calla a Jarvis y pasa directo a escuchar
- [x] Nada falla en silencio: Whisper roto, micrófono desconectado o micrófono mudo llegan al bus como `core.error`
- [x] Umbrales recargables en caliente (`test_la_configuracion_se_lee_en_caliente`)
- [x] Banco `scripts/voice_bench.py` construido sobre los `trace.span` del bus
- [ ] **Transcripción final < 400 ms** — 886 ms en esta CPU. Ver deuda.
- [ ] **Probado con voz humana por micrófono** — este equipo no tiene micrófono funcionando (ver deuda). Probado de punta a punta con voz sintética y con los altavoces reales.

### Decisiones tomadas

- **DECISIÓN**: el pipeline no entra en PENSANDO tras la transcripción final, **porque** ese estado es de la cascada (Fase 2). El pipeline publica `voice.stt.final` y vuelve a DORMIDO; si nadie contesta no queda colgado en un estado ajeno.
- **DECISIÓN**: el barge-in se dispara con la palabra clave y no con cualquier voz, **porque** sin cancelación de eco el VAD oye a Jarvis por los altavoces y se interrumpiría a sí mismo.
- **DECISIÓN**: el tiempo del pipeline se cuenta en milisegundos de audio procesado, no de reloj, **porque** así los plazos son deterministas en las pruebas sin esperar de verdad, y con el micrófono es lo mismo.
- **DECISIÓN**: los motores de inferencia son síncronos y el pipeline decide qué mandar a un hilo, **porque** todos envuelven bibliotecas de CPU síncronas; una fachada `async` solo escondería dónde se bloquea.
- **DECISIÓN**: al cerrarse una frase, el micrófono vuelve a vigilar la palabra clave mientras Whisper transcribe aparte, **porque** con 0.9 s de transcripción un segundo "Jarvis" inmediato se perdía. El estado sigue en ESCUCHANDO hasta publicar la final.
- **DECISIÓN**: los modelos se descargan con un comando explícito, nunca solos al arrancar, **porque** bajar 500 MB sin que nadie lo pida viola "local por defecto". Si falta uno, `ModelMissingError` dice qué correr.
- **DECISIÓN**: los motores reales viven en el extra `[voice]` de `pyproject.toml` y solo los importa `voice/factory.py`, **porque** las pruebas del núcleo no deben depender de 600 MB de pesos.
- **DECISIÓN**: `publish_idle_level = false` por defecto, **porque** sin palabra clave no debe salir nada del pipeline, ni siquiera cuánto ruido hay en la habitación.
- **DECISIÓN**: `stt_model = "base"`, `stt_language = "es"` y `stt_partials = false` en esta máquina, **porque** son los valores medidos que dejan la latencia en ~0.9 s. Las parciales compiten por la CPU con la final y la retrasarían.

### Deuda técnica y pendientes

- **La meta de 400 ms no se alcanza en CPU.** Opciones, de menor a mayor esfuerzo: (1) motor `cloud` con Deepgram en streaming (el campo `stt_engine` ya existe, falta el adaptador); (2) una GPU dedicada y `large-v3-turbo`; (3) recortar el contexto del codificador de Whisper, que hoy procesa siempre 30 s aunque la frase dure 2.
- **`hey_jarvis` no reconoce "Jarvis" pronunciado en español.** El modelo del catálogo se entrenó con voces inglesas: responde a "DYAR-vis" (0.96–0.998) pero no a "Jarvis" con jota (0.001). Con pronunciación inglesa funciona perfecto y sin falsos positivos. La solución es entrenar un modelo propio con openWakeWord (entrena con voz sintética: las voces es_MX de Piper sirven) y apuntar `wake_model` a su `.onnx`.
- **Este equipo no tiene micrófono funcionando.** Windows no tiene entrada predeterminada; el conector Realtek se abre por WDM-KS pero nunca entrega audio (deshabilitado o sin nada enchufado), y los micrófonos Bluetooth no estaban conectados. El pipeline lo detecta a los 3 s y lo anuncia (`CaptureStalledError`). Para probar en vivo: conectar un micrófono o habilitarlo en Configuración > Sistema > Sonido, y correr `scripts/voice_live.py`.
- **El WER del banco penaliza números**: Whisper escribe "70%" por "setenta por ciento". Para el clasificador es correcto, pero el banco lo cuenta como error. Normalizar números si llega a importar.
- **Al interrumpir, Piper termina la frase que estaba generando**: el altavoz calla al instante, pero el hilo de síntesis no se suelta hasta acabar esa frase (decenas de ms; no se oye).
- **Quedaron en `models/` Whisper `small` y `tiny` y la voz inglesa** de las mediciones. No se versionan, y `voice_bench.py --real-wake` usa la voz inglesa.

### Para la Fase 2 — Orquestador cognitivo

- **Suscríbete a `voice.stt.final`.** La carga es una `Utterance` con `is_final=True`, el texto ya sin la palabra clave y el `correlation_id` de la interacción. Puede venir vacía (`is_empty`): "Jarvis" y silencio. Contesta algo breve o nada, pero no lo trates como error.
- **Para hablar, publica `cognition.response`** con un `Response`. El pipeline lo dice solo si `should_speak` y el texto no está vacío, con el mismo `correlation_id`. No llames a `speak()` directamente: así el HUD y la bitácora ven la respuesta.
- **El estado PENSANDO es tuyo.** Publícalo en `core.state.changed` con un `StateChange` al empezar a resolver. El pipeline pasa a HABLANDO cuando empieza a decir la respuesta y vuelve a DORMIDO al terminar.
- **El fast path recibe texto de `base`, no de `large`.** Las órdenes cortas salen bien gracias a `stt_prompt`, pero conviene que las reglas del nivel 1 toleren variaciones ("Sierra todas las ventanas", "al 70%"). Si una orden frecuente falla, agregar su vocabulario a `stt_prompt` cuesta cero.
- **Para probar sin micrófono**, `ArraySource` + `MemorySink` y los falsos de `tests/voice_fakes.py` corren el pipeline entero en milisegundos.

```bash
.venv/Scripts/python -m pip install -e ".[dev,voice]"
```

```bash
.venv/Scripts/python scripts/download_voice_models.py
```

```bash
.venv/Scripts/python scripts/voice_bench.py
```

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
