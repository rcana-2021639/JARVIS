# ESTADO DEL PROYECTO JARVIS

Bitácora de fases. Cada sesión lee este archivo antes de escribir una sola línea.

---

## FASE 2 — ORQUESTADOR COGNITIVO — **COMPLETA** (con dos criterios pendientes: la latencia del nivel 2 en esta CPU y el costo real sin clave de la API)

Fecha: 2026-11-02

### Entregado

**La cascada** — `core/jarvis/cognition/`

| Archivo | Qué contiene |
|---|---|
| `classifier.py` | Clasificador en dos etapas: regex del catálogo → difusa con rapidfuzz → MiniLM. Nunca lanza: lo desconocido es `chat.free` |
| `catalog.py`, `intents_fast.py`, `intents_open.py` | **55 intenciones** (34 de forma fija, 21 abiertas) en 9 dominios, con patrones, frases canónicas y ejemplos |
| `text.py` | Normalización: acentos, "setenta y cinco" → 75, "por ciento" → %, duraciones |
| `embeddings.py` | paraphrase-multilingual-MiniLM-L12-v2 en ONNX (AVX2, 118 MB), sin PyTorch, con caché en disco |
| `fastpath.py`, `phrases.py`, `timers.py` | Nivel 1: 34 manejadores, frases que rotan sin repetirse, temporizadores reales en proceso |
| `local_llm.py`, `streaming.py` | Nivel 2: Ollama con streaming, esquema JSON con autoevaluación, logprobs, herramientas y modelo siempre en RAM |
| `cloud_llm.py`, `budget.py` | Nivel 3: API de Claude con streaming, herramientas, modelo por complejidad, costo en vivo, tope diario persistido y reintentos |
| `router.py` | Las reglas de la cascada, la compuerta de confianza y el interruptor de privacidad |
| `levels.py` | Los caminos de los niveles 2 y 3: escaladas, caídas y degradaciones, todas con aviso hablado |
| `context.py` | Historial en RAM con resúmenes de lo viejo y presupuesto estricto de tokens por prioridad |
| `tool_loop.py`, `demo_tools.py` | Bucle de herramientas con tope, detección de bucles, plazo y estrategia alterna; calculadora y reloj de demostración |
| `speaker.py` | El canal hablado: frases a `cognition.response.chunk` conforme se generan |
| `orchestrator.py`, `factory.py` | Todo conectado al bus; la fábrica arma la versión real desde la configuración |
| `prompts/jarvis.md`, `prompts/local_json.md` | La personalidad y el formato con autoevaluación, editables sin tocar Python |

**Secretos** — `core/jarvis/secrets.py`: la clave de la API vive en el llavero de Windows (servicio `jarvis`); respaldo por variable de entorno. Nunca en un archivo.

**Contratos nuevos** — `ResponseChunk`, `RouteDecision`, `CostUpdate`, `CascadeRecord` (en `contracts/cascade.py`), el campo `Response.streamed` y los eventos `cognition.response.chunk`, `cognition.cost.updated`, `cognition.completed`. Exportados a `shared/protocol/` (23 archivos).

**Voz** — el pipeline dice cada `cognition.response.chunk` en cuanto llega, no repite una respuesta ya dicha en streaming, no vuelve a DORMIDO entre frases en cola y, al oír la palabra clave, descarta lo pendiente de las órdenes anteriores.

**Configuración** — `CognitionConfig` pasó de 11 a 39 campos, todos en `config/jarvis.toml`. Los modelos de la nube se corrigieron a la generación vigente (`claude-haiku-5-5`, `claude-sonnet-5-5`, `claude-opus-5-5`) con su tabla de precios.

**Scripts**

- `scripts/cognition_bench.py` — el banco de la fase: 50 órdenes y su distribución; `--run` las ejecuta, `--privacy` vigila sockets en modo solo-local, `--kill-ollama` mata Ollama a media respuesta
- `scripts/jarvis_chat.py` — hablar con Jarvis escribiendo (y oírlo con `--voz`): la demostración sin micrófono
- `scripts/download_cognition_models.py` — MiniLM y `ollama pull`
- `scripts/set_secret.py` — guarda la clave de la API en el llavero
- `scripts/voice_live.py` — ahora contesta con la cascada; `--echo` conserva el eco de la Fase 1

**Pruebas** — 298 en total (117 nuevas). `test_cognition_units.py` (49) cubre texto, clasificador, streaming, gasto, enrutador y contexto; `test_cognition_clients.py` (23) los clientes de Ollama y Claude contra un transporte y un SDK simulados, y el bucle de herramientas; `test_cognition_cascade.py` (31) los criterios de aceptación sobre el bus con falsos deterministas (`tests/cognition_fakes.py`); `test_cognition_models.py` (6) MiniLM y Ollama reales, y se salta sola si faltan.

### Mediciones — i5-10500 sin GPU, `qwen3:4b` en Ollama 0.40, sin clave de la API

| Medida | Resultado | Meta | |
|---|---|---|---|
| Distribución del enrutador, 50 órdenes (`cognition_bench.py`) | **52 / 34 / 14 %** | ~50 / 35 / 15 | ✅ |
| Nivel que atendió, con la cascada real (`--run`) | 52 / 28 / 20 % | — | 3 escaladas, todas legítimas: clima, archivos y web |
| Clasificador | p50 0.06 ms, p95 5.2 ms, máx 5.8 ms | < 10 ms | ✅ |
| Nivel 1 de punta a punta | p50 1 ms, p95 2 ms | < 100 ms | ✅ |
| Nivel 2, primera frase hablable | p50 **7.4 s** | < 1 s | ❌ |
| Nivel 2, respuesta completa | p50 10.1 s, p95 19.7 s | — | |
| Nivel 2, prefill del sistema | 4.3 s la primera vez, 0.6 s con la caché de Ollama | — | |
| Nivel 2, generación | 5–7 tokens/s | — | es lo que pone el piso |
| LOCAL_ONLY con la red vigilada (`--privacy`) | solo 127.0.0.1 (Ollama), en asyncio y en `netstat` | nada sale | ✅ |
| Ollama muerto a media respuesta (`--kill-ollama`) | dice la frase ya generada, avisa "Perdí la conexión", la siguiente orden funciona, cero `core.error` | no crashea | ✅ |

Cómo se llegó de 29 s a 7.4 s en la primera frase del nivel 2, todo medido con el banco: (1) el historial expulsaba un turno por orden y metía su resumen en el prompt de sistema, así que Ollama reprocesaba ~1500 tokens en cada petición; ahora se expulsa por bloques y el prefijo se reutiliza. (2) Las órdenes del nivel 1 dejaron de entrar al historial: el modelo copiaba su "ese control todavía no existe". (3) La compuerta temprana usaba la probabilidad de los dígitos de la confianza y escalaba de más; ahora usa solo la autoevaluación. (4) Se pide que la primera oración sea corta, porque es lo primero que suena.

### Criterios de aceptación

- [x] "Sube el volumen" se resuelve por nivel 1 en menos de 100 ms — 1–2 ms (`test_sube_el_volumen_se_resuelve_en_nivel_1_en_menos_de_100_ms` y el banco). Hoy contesta que todavía no controla el volumen: la herramienta es de la Fase 6.
- [ ] **"¿Qué opinas de X?" va a nivel 2 y responde en menos de 1 s** — va al nivel 2 (`test_que_opinas_va_al_nivel_2_y_habla_en_streaming`), pero la primera frase tarda 7.4 s en esta CPU. Ver deuda.
- [x] "Analiza este archivo de 300 líneas y refactorízalo" escala a nivel 3 — por la regla de código, con Sonnet (`test_refactorizar_un_archivo_escala_al_nivel_3`). Ejecutado con la nube simulada: no hay clave.
- [x] Activar LOCAL_ONLY hace que nada salga a la red — verificado observando el tráfico: `cognition_bench.py --privacy` corre las 50 órdenes con una clave falsa a mano, vigila asyncio y `netstat` por PID, y solo aparece 127.0.0.1. Por voz: "modo local" (`test_modo_local_por_voz_corta_la_nube`).
- [ ] **El contador de costo refleja el gasto real de la API** — la contabilidad está probada contra la forma real de `usage` del SDK, con caché incluida (`test_claude_cobra_y_publica_el_costo`), pero sin clave no se pudo comparar contra el gasto real. Pendiente de una clave.
- [x] Al llegar al límite diario, degrada y avisa en vez de seguir gastando (`test_al_llegar_al_limite_avisa_y_degrada`). El acumulado se persiste: reiniciar no regala presupuesto.
- [x] Un banco corre 50 órdenes variadas y reporta la distribución por nivel — `scripts/cognition_bench.py`: 52 / 34 / 14.
- [x] Apagar Ollama a mitad de una respuesta no crashea nada — verificado en vivo con `--kill-ollama` (mata el servidor tras la primera frase) y en pruebas (`test_apagar_ollama_a_mitad_de_respuesta_no_crashea`).
- [x] `mypy core/ --strict` limpio (76 archivos), `ruff check core/ tests/ scripts/` limpio, `pytest` verde (298), esquemas al día (23).
- [x] Ningún archivo de `cognition/` pasa de 400 líneas: `orchestrator.py` se partió en `control.py`, `reporting.py` y `spoken.py`.

### Decisiones tomadas

- **DECISIÓN**: la confianza va **primero** en el esquema JSON del nivel 2, **porque** así se conoce antes de generar la respuesta. Con ella, una compuerta decide: si es holgada (≥ umbral + 0.15), se habla en streaming; si está al límite, se genera entera y se mide la confianza combinada antes de decir nada; si es baja, se corta y se escala sin haber dicho una sola palabra local.
- **DECISIÓN**: confianza combinada = 0.5 · exp(media de logprobs) + 0.5 · autoevaluación, **porque** cada señal falla distinto: los logprobs miden fluidez (un modelo puede inventar con soltura) y la autoevaluación mide si sabe (pero los modelos chicos se sobreestiman).
- **DECISIÓN**: sin nube a donde escalar (sin clave, sin presupuesto o en LOCAL_ONLY), la compuerta nunca aborta. Se genera entera y, si la confianza no alcanza, se contesta precedida de "No estoy del todo seguro", **porque** abortar para no tener a quién pasar la pregunta solo dejaría al usuario sin nada.
- **DECISIÓN**: el modo estructurado (con esquema) es el normal del nivel 2, **porque** qwen3 escribe su razonamiento como texto aunque se le pida `think: false`. El esquema lo encierra. En el modo herramientas se agrega `/no_think` y se limpian las etiquetas `<think>`.
- **DECISIÓN**: MiniLM **multilingüe** y no el all-MiniLM inglés, **porque** las órdenes llegan en español. Corre en ONNX con onnxruntime y tokenizers, que ya estaban instalados por la voz: no hace falta PyTorch. La entrada se trunca a 32 tokens, **porque** con 64 una frase larga tardaba 15 ms y la intención se decide en las primeras palabras.
- **DECISIÓN**: si el clasificador se pasa de 10 ms, el resultado se usa igual y se registra el aviso, **porque** descartar un resultado ya calculado para devolver "no sé" no ahorra nada.
- **DECISIÓN**: los manejadores del nivel 1 que necesitan el sistema operativo (volumen, apps, brillo, capturas) delegan en el `ToolRegistry`. Como esas herramientas son de la Fase 6, hoy contestan "todavía no tengo control de eso", **porque** fingir que se subió el volumen viola "nada falla en silencio". Al registrar las herramientas en la Fase 6 funcionan sin tocar `fastpath.py`.
- **DECISIÓN**: los temporizadores sí son reales, **porque** no necesitan al sistema operativo: son tareas de asyncio que anuncian por el bus.
- **DECISIÓN**: una orden a la vez. Si llega otra mientras la anterior piensa, la anterior se cancela (y queda registrada como cancelada), **porque** dos respuestas mezclándose en los altavoces es peor que perder una.
- **DECISIÓN**: el gasto del día se persiste en `config/data/spend.json`, **porque** si viviera solo en RAM, cada reinicio regalaría un presupuesto nuevo. No es memoria de la conversación (eso sigue en RAM, como pide la fase).
- **DECISIÓN**: el cliente de la nube consulta el modo de privacidad por su cuenta antes de cada llamada, además del enrutador, **porque** LOCAL_ONLY tiene que cortar la red aunque el enrutador tuviera un error. Defensa en profundidad.
- **DECISIÓN**: el prompt de sistema no lleva hora ni estado; el estado viaja pegado a la orden del usuario, **porque** un prefijo idéntico entre peticiones es lo que permite a Ollama y a la API reutilizar su caché. En CPU, reprocesar el prefijo cuesta cientos de milisegundos.
- **DECISIÓN**: los tokens del contexto se estiman (caracteres / 3.5) y no se cuentan con el tokenizador de cada modelo, **porque** contar exacto costaría una llamada por petición y el presupuesto es un límite de seguridad, no una factura.
- **DECISIÓN**: los resúmenes del historial viejo son una línea recortada por turno y no un resumen con LLM, **porque** resumir con el modelo cuesta segundos en esta CPU. La consolidación de verdad es la Fase 3.
- **DECISIÓN**: Haiku 5.5 se llama con el razonamiento apagado y Sonnet 5.5 con `between_tools`, ambos con esfuerzo `low`; Opus 5.5 siempre razona y se llama con esfuerzo `high`, **porque** lo hablado necesita latencia y lo profundo, calidad. Ningún modelo actual acepta `temperature`.
- **DECISIÓN**: lo que no cabe en dos frases no se dice: se habla lo primero y "El resto está en el chat". Los bloques de código nunca se dicen.
- **DECISIÓN**: `phrases.py` tutea, igual que la personalidad. Nada de exclamaciones ni de "¡Claro que sí!"; hay una prueba que lo verifica.

### Revisión de seguridad (2026-11-02, después del cierre)

Se probó lo delicado de las fases 0-2. Lo que se corrigió:

- **Lo que dice el usuario ya no va al registro.** `voice.final` y las etiquetas de los temporizadores escribían el texto tal cual en `logs/jarvis.log`, que es texto plano y rota durante semanas. Ahora `jarvis.logging.user_text()` escribe `<N caracteres>`. Para depurar: `log_user_content = true` en `jarvis.toml`. Prueba: `test_la_transcripcion_no_queda_en_el_registro`.
- **Tope de largo de una orden** (`max_input_chars = 2000`). Un texto enorme iba entero a la nube (contexto > 8k → nivel 3) y costaba. Ahora se rechaza antes de clasificar, sin llamar a ningún modelo, y al registro va recortado. Prueba: `test_una_orden_enorme_se_rechaza_sin_llamar_a_ningun_modelo`.

Lo que se probó y está bien: ninguna regex del catálogo se dispara con entradas largas o patológicas (20 000 palabras: 0.3 s, ahora rechazadas antes); la calculadora rechaza `9**9**9`, desbordes, división por cero y cualquier cosa que no sea aritmética; `format_exc_info` no vuelca variables locales en las excepciones; no hay secretos en el repositorio.

Lo que queda a la vista, sin corregir:

- **El repositorio es público** y muestra el nombre del usuario (`user_name` en `jarvis.toml`) y su correo en los commits.
- **Desde la Fase 6 los modelos podrán actuar sobre la PC.** Contenido de archivos o páginas web puede traer instrucciones escondidas (inyección de prompt). El Guardián (Fase 8) debería existir antes de dar herramientas que escriban, borren o envíen algo.
- **Ollama escucha en 127.0.0.1 sin autenticación**: cualquier programa local puede usarlo. Es su comportamiento estándar.

### Deuda técnica y pendientes

- **La meta de < 1 s del nivel 2 no se alcanza en CPU.** qwen3:4b genera a 5–7 tokens/s: solo el prefijo del JSON (`{"confianza": 0.9, "respuesta": "`) son ~2 s, y una primera frase corta otros 2–4. Opciones, de menor a mayor esfuerzo: (1) **poner la clave de la API** y `privacy_mode = "CLOUD_PREFERRED"`: Haiku 5.5 sin razonamiento debería contestar en torno a 1 s (no medido: no hay clave), y es lo que la tabla de VRAM del documento manda para máquinas sin GPU ("nube agresiva"); (2) `ollama pull qwen3:1.7b` y `local_model = "qwen3:1.7b"`: más rápido, peor criterio (no medido: no se bajó sin permiso); (3) una GPU dedicada.
- **Sin clave de la API no se probó el nivel 3 real.** La contabilidad, los reintentos, el streaming y la selección de modelo están probados contra un SDK simulado con la forma del real. La primera llamada real conviene hacerla con `jarvis_chat.py`, que imprime el costo. Para guardar la clave: `python scripts/set_secret.py anthropic_api_key`.
- **El nivel 2 no usa herramientas en esta CPU** (`local_tools = false`). Con ellas, qwen3:4b ignora `/no_think`, escribe su razonamiento en inglés como texto y tarda más de 40 s. El bucle de herramientas está completo y probado: lo usa el nivel 3 y, con GPU, el 2.
- **Las órdenes de forma fija que tocan el sistema no hacen nada todavía.** Volumen, apps, brillo, capturas, ventanas y bloqueo contestan con honestidad que ese control no existe aún. Al registrar las herramientas de la Fase 6 en el `ToolRegistry` (`media.volume_change`, `apps.open`, `system.lock`…; los nombres están en `fastpath.py`) funcionan sin tocar nada más.
- **El modelo local a veces inventa.** En el banco ofreció cosas que no puede hacer ("puedo programar"). La personalidad lo prohíbe, pero un 4B la sigue a medias. La compuerta de confianza filtra las dudas grandes, no las alucinaciones seguras.
- **Las etiquetas de los temporizadores pierden las tildes** ("para el te") porque se extraen del texto normalizado. Al decirlas suena igual; en el chat se nota.
- **El sondeo de privacidad con `netstat` cada 250 ms podría perder una conexión de vida muy corta.** Por eso hay un segundo observador en asyncio, que ve todas las que pasan por el bucle.
- **Ollama se instaló con winget** (0.40.13) y arranca con la sesión. `--kill-ollama` lo relanza con `ollama serve` desacoplado.
- **Una orden nueva cancela la anterior.** "¿Qué hora es?" mientras el nivel 2 piensa en otra cosa la cancela. Es lo que se quiere casi siempre; si molesta, la Fase 12 puede distinguir órdenes rápidas que no interrumpen.
- **`voice/pipeline.py` (Fase 1) tiene ~850 líneas**, el doble del tope de 400 del documento. Viene así desde su entrega. Partirlo (captura, escucha y habla) es trabajo de una sesión; conviene antes de que la Fase 4 lo toque para el núcleo 3D.
- **Dos pruebas de la Fase 0 eran frágiles**: una de redondeo en `test_bus.py` (0.97799… contra 0.978) y la siguiente.
- **Una prueba de tiempo de la Fase 0 era frágil en Windows** (`test_traced_mide_una_corrutina_y_publica_al_bus`): un `sleep(0.02)` midió 19.7 ms con la CPU ocupada. Se le dio margen (≥ 18 ms).

### Para la Fase 3 — Sistema de memoria

- **El historial vive en `ConversationHistory`** (`cognition/context.py`): turnos completos en RAM y resúmenes de una línea de lo expulsado. La Fase 3 lo reemplaza por la capa TRABAJO y alimenta la episódica desde `cognition.completed`, que trae texto, intención, nivel, tiempos y costo de cada interacción.
- **La memoria recuperada ya tiene su lugar en el contexto.** `ContextBuilder.build(memory=[...])` la recibe, la pone con prioridad 3 (debajo del sistema y el estado, encima de los turnos) y la poda antes que el estado si no cabe. Hoy llega vacía; basta con pasarle lo que devuelva la recuperación híbrida.
- **`memory.remember` y `memory.recall` ya se clasifican.** Hoy van al nivel 2, que honestamente dice que no tiene memoria a largo plazo (`prompts/jarvis.md`, última regla: quitarla cuando exista). La Fase 3 puede darles manejador propio en el nivel 1 o como herramientas en el `ToolRegistry`.
- **El presupuesto de recuperación** (`memory.retrieval_budget_tokens = 1500`) compite con `cognition.local_context_tokens = 1500` en el nivel 2. En CPU cada token de contexto se paga: medir antes de subirlo.
- **Para probar sin modelos**, `tests/cognition_fakes.py` arma el orquestador entero con falsos deterministas en milisegundos.

```bash
.venv/Scripts/python -m pip install -e ".[dev,voice,cognition]"
```

```bash
.venv/Scripts/python scripts/download_cognition_models.py
```

```bash
.venv/Scripts/python scripts/cognition_bench.py --run
```

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
