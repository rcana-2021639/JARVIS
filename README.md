# J.A.R.V.I.S.

Asistente personal residente controlado por voz. Vive como servicio permanente
en la PC, escucha por una palabra de activación, entiende lenguaje natural,
opera el sistema operativo y las aplicaciones, recuerda todo lo que ha pasado, y
se presenta a través de una interfaz tridimensional reactiva al sonido.

Se construye en 13 fases. **Estado actual: Fase 4 completa.**

---

## Estado por fases

| # | Fase | Estado |
|---|------|--------|
| 0 | Fundaciones y contratos | ✅ Completa |
| 1 | Pipeline de voz | ✅ Completa |
| 2 | Orquestador cognitivo | ✅ Completa |
| 3 | Sistema de memoria | ✅ Completa |
| 4 | Núcleo 3D | ✅ Completa |
| 5 | HUD completo | Pendiente |
| 6 | Control del sistema operativo | Pendiente |
| 7 | Automatización web | Pendiente |
| 8 | Guardián y bóveda | Pendiente |
| 9 | Puente remoto | Pendiente |
| 10 | Aplicación móvil | Pendiente |
| 11 | Residencia permanente | Pendiente |
| 12 | Inteligencia avanzada | Pendiente |

El detalle de lo entregado en cada fase está en [ESTADO.md](ESTADO.md).

---

## Qué hay hoy

**Fase 4 — el núcleo 3D** (`hud/`). Jarvis ya tiene rostro: una esfera de
energía en una ventana sin marco, transparente y siempre encima (Electron +
React Three Fiber). Seis capas —semilla, 30.000 partículas con ruido simplex,
anillos orbitales con graduación, filamentos, campo de fresnel y postproceso
con bloom— que reaccionan en tiempo real a la voz, la del usuario entrando y
la de Jarvis saliendo, analizada en 32 bandas de frecuencia. Los ocho estados
(dormido, despierto, escuchando, pensando, hablando, ejecutando, error,
confirmación) se ven distintos y se funden en 800 ms. 60 fps en la GPU
integrada de este equipo. El núcleo Python la alimenta por un WebSocket local
(`core/jarvis/hudlink/`); sin núcleo, el HUD corre una simulación.

**Fase 3 — memoria** (`core/jarvis/memory/`). Jarvis ya recuerda. Cinco capas:
la de trabajo (últimos turnos, en RAM), la episódica (todo lo ocurrido, SQLite
con búsqueda de texto completo), la semántica (hechos con importancia y
decaimiento, vectores en LanceDB), un grafo de entidades (personas, proyectos,
archivos y cómo se relacionan) y la procedimental (lo que se pide igual tres
veces se vuelve rutina de ruta rápida). Antes de cada respuesta de un modelo,
cuatro búsquedas en paralelo (significado, palabras, grafo y recencia) traen
lo que viene al caso en ~30 ms. Cada noche, el modelo local consolida el día:
extrae hechos, enlaza entidades, resume, deja desvanecer lo que no se usa y
propone rutinas.

Por voz: "recuerda que…", "olvida lo de…" (con confirmación), "qué sabes
de…", "qué hicimos ayer" y "qué aprendiste anoche".

**Fase 2 — orquestador cognitivo** (`core/jarvis/cognition/`). Jarvis ya
entiende y contesta. Cada orden pasa por un clasificador de intención (regex,
difusa y MiniLM, en menos de 10 ms) y una cascada de tres niveles:

| Nivel | Qué | Cuándo |
|---|---|---|
| 1 · Ruta rápida | Determinista, sin modelo, ~1 ms | Volumen, apps, hora, temporizadores, modo de privacidad… (34 órdenes) |
| 2 · Modelo local | Ollama (`qwen3:4b`), con confianza propia | Conversación, consultas, redacción corta |
| 3 · Claude | Haiku / Sonnet / Opus según la tarea, con costo | Código, tareas de muchos pasos, mucho contexto, o si el nivel 2 duda |

Nunca se llama a la nube si el nivel inferior resuelve. Cada respuesta registra
qué nivel la atendió, por qué, cuánto tardó y cuánto costó. Hay tope de gasto
diario, modo solo-local y avisos honestos cuando algo no está disponible.

**Fase 1 — pipeline de voz** (`core/jarvis/voice/`). Jarvis ya oye y habla:
escucha la palabra clave, rescata el medio segundo previo del buffer circular,
decide con Silero VAD cuándo terminaste de hablar, transcribe con
faster-whisper y responde con Piper. Decirle la palabra clave mientras habla lo
calla al instante. Todo lo que pasa se publica al bus: estado, nivel del
micrófono a ~30 Hz, amplitud del habla a 60 Hz y transcripciones.

**Fase 0 — fundaciones**: los contratos de datos (`core/jarvis/contracts/`), el
bus de eventos con contrapresión (`core/jarvis/bus/`), la configuración con
recarga en caliente (`core/jarvis/config.py`) y el registro con id de
correlación (`core/jarvis/logging.py`).

---

## Puesta en marcha

Requiere **Python 3.11 o superior**.

```bash
python -m venv .venv
```

```bash
.venv/Scripts/python -m pip install -e ".[dev,voice]"
```

En Linux o macOS, `.venv/bin/python` en lugar de `.venv/Scripts/python`. El
extra `voice` trae los motores locales (faster-whisper, openWakeWord, Piper y
PortAudio); sin él, el núcleo y sus pruebas siguen funcionando.

### Modelos de voz

```bash
.venv/Scripts/python scripts/download_voice_models.py
```

Baja a `models/` (no versionado) el modelo de Whisper, la voz de Piper y la
palabra clave. Jarvis nunca descarga nada por su cuenta.

### Modelos de la cascada

Instala [Ollama](https://ollama.com/download) y después:

```bash
.venv/Scripts/python -m pip install -e ".[dev,voice,cognition]"
```

```bash
.venv/Scripts/python scripts/download_cognition_models.py
```

Baja el clasificador (MiniLM, 118 MB) a `models/intent/` y el modelo local con
`ollama pull`.

### Clave de la API de Claude (opcional)

Sin clave, el nivel 3 queda fuera y Jarvis lo dice. Para activarlo:

```bash
.venv/Scripts/python scripts/set_secret.py anthropic_api_key
```

La clave se guarda en el Administrador de credenciales de Windows, nunca en un
archivo.

### Hablar con Jarvis

```bash
.venv/Scripts/python scripts/jarvis_chat.py --voz
```

Escribes y Jarvis contesta en pantalla y por los altavoces; muestra qué nivel
atendió cada orden. Sin `--voz`, solo texto. Lo que le cuentes queda en la
memoria (`config/data/`, no versionado) entre sesiones.

Con micrófono:

```bash
.venv/Scripts/python scripts/voice_live.py
```

Di **"Jarvis"** y una orden. Con `wake_engine = "whisper"` (el valor de este
equipo) Whisper reconoce "Jarvis" en español; con `"openwakeword"` hay que
decir "Hey Jarvis" a la inglesa.
`--echo` vuelve al eco de la Fase 1 y `--list-devices` muestra qué micrófonos
ve.

### Medir la cascada

```bash
.venv/Scripts/python scripts/cognition_bench.py
```

Clasifica y enruta 50 órdenes variadas y reporta la distribución por nivel.
Con `--run` las ejecuta de verdad; `--privacy` verifica que en modo solo-local
no sale tráfico y `--kill-ollama` mata Ollama a media respuesta.

### El HUD

Requiere [Node.js](https://nodejs.org) 20 o superior.

```bash
cd hud && npm install
```

```bash
cd hud && npm run build && npm run electron
```

La ventana aparece arriba a la derecha y se arrastra por su barra. Con
`jarvis_chat.py` o `voice_live.py` corriendo se conecta sola al núcleo
("NÚCLEO EN LÍNEA"); sin núcleo, muestra una simulación de los ocho estados.
Para verla reaccionar a la voz de Jarvis sin micrófono:

```bash
.venv/Scripts/python scripts/hud_demo.py
```

Para trabajar el HUD en el navegador con recarga en caliente: `npm run dev`
dentro de `hud/` y abrir http://127.0.0.1:5173 (`?fps=1` muestra los fps).

### Inspeccionar la memoria

```bash
.venv/Scripts/python scripts/memory_inspect.py --query "mi hermana"
```

Muestra cuántos registros hay por capa, los hechos por importancia, el grafo
y las rutinas, y prueba una recuperación con sus latencias. `--bench 10000`
mide la recuperación con 10.000 registros sintéticos, `--simulate` consolida
100 episodios simulados con el modelo local y `--consolidate` corre la
consolidación ya, sin esperar a la noche.

### Medir el pipeline de voz

```bash
.venv/Scripts/python scripts/voice_bench.py
```

Sintetiza órdenes, las pasa por el pipeline completo y reporta latencia,
precisión y costo por etapa. No necesita micrófono.

### Verificar que todo está bien

```bash
.venv/Scripts/python -m pytest
```

```bash
.venv/Scripts/python -m mypy core/ --strict
```

```bash
.venv/Scripts/ruff check core/ tests/ scripts/
```

### Ver el bus funcionando

```bash
.venv/Scripts/python scripts/bus_demo.py
```

Publica 1000 eventos a tres consumidores y reporta latencias. Con
`--slow-consumer` demuestra la contrapresión: un consumidor atascado pierde
eventos pero no bloquea al publicador ni afecta a los demás.

### Regenerar los esquemas compartidos

```bash
.venv/Scripts/python scripts/export_schemas.py
```

Escribe en `shared/protocol/` un JSON Schema por contrato, para que el HUD
(TypeScript) y la app móvil no redeclaren los tipos a mano.

---

## Estructura

```
core/jarvis/     el cerebro y el cuerpo (Python)
  contracts/     modelos Pydantic -- el idioma común
  bus/           publicación/suscripción de eventos
  voice/         wake · vad · stt · tts · interrupción      (Fase 1) ✅
  cognition/     router · cascada · prompts · confianza      (Fase 2) ✅
  memory/        cinco capas · recuperación · consolidación  (Fase 3) ✅
  hudlink/       WebSocket local núcleo → HUD                 (Fase 4) ✅
  tools/         servidores MCP                              (Fases 6-7)
  guardian/      permisos · confirmación · auditoría         (Fase 8)
  bridge/        servidor WS · autenticación · presencia     (Fase 9)
  service/       daemon · supervisor · bandeja               (Fase 11)
hud/             Electron + React + Three.js                 (Fases 4-5)
  electron/      ventana sin marco, transparente, siempre encima
  src/core3d/    las seis capas de la esfera y sus estados   (Fase 4) ✅
  src/design/    tokens de diseño: la única fuente de verdad
mobile/          Expo                                        (Fase 10)
shared/protocol/ esquemas JSON compartidos por los tres
config/          jarvis.toml y policy.toml
scripts/         utilidades y bancos de prueba
```

---

## Configuración

Todo se ajusta en [`config/jarvis.toml`](config/jarvis.toml), que se relee solo:
cambiar un valor se refleja sin reiniciar. Si el archivo queda mal escrito,
Jarvis conserva la configuración anterior y lo dice en el registro.

Para sobrescribir sin tocar el archivo, variables de entorno con prefijo
`JARVIS_` y doble guion bajo para anidar:

```bash
JARVIS_VOICE__WAKE_THRESHOLD=0.7
```

---

## Nota sobre este hardware

La máquina de desarrollo no tiene GPU dedicada (Intel UHD 630, 16 GB de RAM).
Según la tabla de VRAM del documento maestro eso implica modelo local pequeño en
CPU y escalado más frecuente a la nube. Los valores por defecto de
`config/jarvis.toml` ya están calibrados para eso y están marcados con comentarios.
Con una GPU dedicada hay que subir `stt_model` y `local_model`.

Medido en la Fase 1 (i5-10500): la transcripción final tarda ~0.9 s con
Whisper `base`, y la meta es 0.4 s. Las mediciones completas están en
[ESTADO.md](ESTADO.md).

---

## Principios que no se negocian

1. **Latencia ante todo** — primer audio de respuesta en menos de 700 ms.
2. **La palabra clave es sagrada** — sin "Jarvis" no ocurre nada. Ni red, ni
   disco, ni acción.
3. **Local por defecto, nube por excepción.**
4. **Capacidad total con rastro** — todo queda auditado; lo destructivo exige
   confirmación hablada.
5. **Nada falla en silencio.**
6. **Siempre encendido** — nunca abrir una terminal para usarlo.
7. **La estética es funcional** — el HUD comunica estado, no decora.
