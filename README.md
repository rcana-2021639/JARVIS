# J.A.R.V.I.S.

Asistente personal residente controlado por voz. Vive como servicio permanente
en la PC, escucha por una palabra de activación, entiende lenguaje natural,
opera el sistema operativo y las aplicaciones, recuerda todo lo que ha pasado, y
se presenta a través de una interfaz tridimensional reactiva al sonido.

Se construye en 13 fases. **Estado actual: Fase 1 completa.**

---

## Estado por fases

| # | Fase | Estado |
|---|------|--------|
| 0 | Fundaciones y contratos | ✅ Completa |
| 1 | Pipeline de voz | ✅ Completa |
| 2 | Orquestador cognitivo | Pendiente |
| 3 | Sistema de memoria | Pendiente |
| 4 | Núcleo 3D | Pendiente |
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

### Hablar con Jarvis

```bash
.venv/Scripts/python scripts/voice_live.py
```

Di **"Hey Jarvis"** —pronunciado a la inglesa, "DYAR-vis"— y una orden. Por
ahora te la repite: entenderla es la Fase 2. Con `--list-devices` muestra qué
micrófonos ve.

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
  cognition/     router · cascada · prompts · confianza      (Fase 2)
  memory/        cinco capas · consolidación                 (Fase 3)
  tools/         servidores MCP                              (Fases 6-7)
  guardian/      permisos · confirmación · auditoría         (Fase 8)
  bridge/        servidor WS · autenticación · presencia     (Fase 9)
  service/       daemon · supervisor · bandeja               (Fase 11)
hud/             Electron + React + Three.js                 (Fases 4-5)
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
