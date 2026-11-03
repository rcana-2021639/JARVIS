"""Banco de la cascada cognitiva: 50 ordenes variadas y su distribucion por nivel.

Meta del documento maestro: aproximadamente 50 / 35 / 15 entre los niveles
1, 2 y 3.

Modos:

    python scripts/cognition_bench.py                # clasifica y enruta (sin LLM)
    python scripts/cognition_bench.py --run          # ejecuta la cascada completa
    python scripts/cognition_bench.py --kill-ollama  # mata Ollama a media respuesta
    python scripts/cognition_bench.py --privacy      # LOCAL_ONLY vigilando la red

Sin `--run` mide solo clasificador y enrutador, con la nube supuesta
disponible: es la distribucion que el diseno produce. Con `--run` se ejecuta
de verdad: el nivel 2 con Ollama y el nivel 3 con la API si hay clave en el
llavero; si no la hay, con una nube simulada que se marca como tal.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))

from jarvis.bus import InMemoryEventBus
from jarvis.cognition.factory import build_cascade, build_classifier
from jarvis.cognition.llm import LLMReply, LLMRequest
from jarvis.cognition.router import Router, profile_task
from jarvis.config import load_config
from jarvis.contracts import CascadeLevel, CascadeRecord, Event, EventType, PrivacyMode
from jarvis.logging import configure_logging

COMMANDS = [
    # Forma fija
    "Sube el volumen",
    "Baja el volumen al treinta por ciento",
    "Pausa la música",
    "Siguiente canción",
    "Abre Spotify",
    "Cierra el navegador",
    "¿Qué hora es?",
    "¿Qué día es hoy?",
    "Pon un temporizador de diez minutos",
    "Cancela el temporizador",
    "Toma una captura de pantalla",
    "Bloquea la computadora",
    "Sube el brillo",
    "Silencia el audio",
    "Minimiza todo",
    "Repite",
    "Cállate",
    "Gracias",
    "Hola",
    "¿Cuánto llevamos gastado hoy?",
    "Pon música de Queen",
    "Cierra todas las ventanas",
    "Reanuda la música",
    "Cambia de ventana",
    "¿Estás ahí?",
    # Conversacion y consultas
    "¿Qué opinas de la música clásica?",
    "¿Cuál es la capital de Australia?",
    "Explícame cómo funciona un motor eléctrico",
    "Cuéntame algo interesante",
    "Recuerda que mi hermana se llama Ana",
    "¿Qué te dije de mi hermana?",
    "Escribe un mensaje de cumpleaños para mi mamá",
    "¿Cómo se dice gracias en japonés?",
    "¿Quién eres?",
    "¿Qué puedes hacer?",
    "Cuánto es el quince por ciento de doscientos treinta",
    "Estoy cansado, tuve un día largo",
    "¿Qué te parece si salgo a correr hoy?",
    "¿Por qué el cielo es azul?",
    "Dame una idea para cenar",
    "¿Cómo está el clima hoy?",
    "Busca el archivo del presupuesto",
    "Abre la página del banco",
    # Lo que pide la nube
    "Analiza este archivo de 300 líneas y refactorízalo",
    "Escribe un script en Python que renombre fotos por fecha",
    (
        "Organízame la semana: primero el gimnasio, luego el dentista, "
        "después la reunión y al final el cumpleaños"
    ),
    "Pregúntale a Claude qué opina de la teoría de cuerdas",
    "Revisa este código y encuentra el bug: def f(x): return x +",
    "Planea un itinerario de tres días en Roma",
    "Usa Opus y piensa a fondo: ¿es P igual a NP?",
]
TARGET = {1: 50, 2: 35, 3: 15}


def _pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * q))] if ordered else 0.0


def _distribution(levels: list[int]) -> str:
    total = len(levels) or 1
    parts = [f"N{lv} {levels.count(lv) * 100 // total}%" for lv in (1, 2, 3)]
    return " / ".join(parts) + "   (meta " + " / ".join(f"{v}%" for v in TARGET.values()) + ")"


def route_only() -> int:
    """Clasifica y enruta las 50 sin llamar a ningun modelo."""
    config = load_config()
    classifier = build_classifier(lambda: config)
    router = Router(lambda: config.cognition)
    from jarvis.cognition.fastpath import FastPath

    fast = FastPath()
    levels: list[int] = []
    times: list[float] = []
    print(f"{'ms':>5}  {'etapa':9} N  {'intencion':22} motivo  <- orden")
    for text in COMMANDS:
        result = classifier.classify(text)
        route = router.decide(
            result.intent,
            profile_task(text, 400),
            privacy=PrivacyMode.HYBRID,
            cloud_blocker=None,
            fast_path_handles=fast.handles(result.intent.name),
        )
        levels.append(int(route.level))
        times.append(result.elapsed_ms)
        print(
            f"{result.elapsed_ms:5.1f}  {result.stage:9} {int(route.level)}  "
            f"{result.intent.name:22} {route.reason[:28]:28} <- {text[:50]}"
        )
    print()
    print(f"Distribucion del enrutador: {_distribution(levels)}")
    print(
        f"Clasificador: p50 {_pct(times, 0.5):.2f} ms, p95 {_pct(times, 0.95):.2f} ms, "
        f"max {max(times):.2f} ms"
    )
    return 0


class SimulatedCloud:
    """Nivel 3 de mentira para medir la cascada sin clave. Siempre lo dice."""

    level = CascadeLevel.CLOUD_LLM

    def unavailable_reason(self) -> str | None:
        return None

    async def available(self) -> bool:
        return True

    def model_for(self, complexity: Any) -> str:
        return f"nube-simulada-{complexity.value.lower()}"

    async def generate(self, request: LLMRequest) -> LLMReply:
        await asyncio.sleep(1.2)
        text = "Respuesta simulada de la nube."
        if request.on_text:
            await request.on_text(text)
        return LLMReply(
            text=text, level=self.level, model=self.model_for(request.complexity), streamed=True
        )


async def run_cascade(limit: int | None, privacy: PrivacyMode | None) -> int:
    """La cascada real sobre las ordenes del banco."""
    config = load_config()
    if privacy is not None:
        config = config.model_copy(
            update={"cognition": config.cognition.model_copy(update={"privacy_mode": privacy})}
        )
    bus = InMemoryEventBus()
    await bus.start()
    records: list[CascadeRecord] = []

    async def collect(event: Event) -> None:
        records.append(event.payload_as(CascadeRecord))

    bus.subscribe(EventType.COGNITION_COMPLETED, collect)
    cascade = build_cascade(bus, lambda: config)
    cloud_note = ""
    if (
        cascade._cloud is not None
        and cascade._cloud.unavailable_reason() == "no hay clave de la API en el llavero"
    ):
        cascade.set_cloud(SimulatedCloud())
        cloud_note = "  (nivel 3 SIMULADO: no hay clave en el llavero)"
    print(f"Calentando {config.cognition.local_model} ...", flush=True)
    assert cascade._local is not None
    await cascade._local.warm_up("x")
    await cascade.start(listen_voice=False)

    for text in COMMANDS[:limit]:
        started = time.perf_counter()
        response = await cascade.handle_text(text)
        elapsed = (time.perf_counter() - started) * 1000
        await bus.wait_idle(2)
        record = records[-1]
        tried = "->".join(str(int(level)) for level in record.levels_tried)
        first = f"{record.first_chunk_ms:6.0f}" if record.first_chunk_ms is not None else "     -"
        conf = f"{record.answer_confidence:.2f}" if record.answer_confidence is not None else "   -"
        print(
            f"{elapsed:7.0f} ms  1a frase {first} ms  N{tried:5} conf {conf}  "
            f"{text[:40]:40} -> {response.text[:70]}",
            flush=True,
        )

    handled = [int(r.handled_by_level) for r in records]
    first_route = [int(r.levels_tried[0]) for r in records]
    print()
    print(f"Decision inicial: {_distribution(first_route)}")
    print(f"Nivel que atendio: {_distribution(handled)}{cloud_note}")
    for level in (1, 2, 3):
        rows = [r for r in records if int(r.handled_by_level) == level]
        if not rows:
            continue
        totals = [r.total_ms for r in rows]
        firsts = [r.first_chunk_ms for r in rows if r.first_chunk_ms is not None]
        line = (
            f"  N{level}: {len(rows):2} ordenes, total p50 {statistics.median(totals):.0f} ms "
            f"p95 {_pct(totals, 0.95):.0f} ms"
        )
        if firsts:
            line += f", primera frase p50 {statistics.median(firsts):.0f} ms"
        print(line)
    escalated = sum(1 for r in records if len(r.levels_tried) > 1)
    print(
        f"  Escaladas: {escalated}. Degradadas: {sum(r.degraded for r in records)}. "
        f"Costo: {sum(r.cost_usd for r in records):.4f} USD"
    )
    await cascade.stop()
    await bus.stop()
    return 0


def _ollama_exe() -> str:
    return str(Path.home() / "AppData/Local/Programs/Ollama/ollama.exe")


async def kill_ollama_midway() -> int:
    """Criterio: apagar Ollama a mitad de una respuesta no crashea nada."""
    config = load_config()
    bus = InMemoryEventBus()
    await bus.start()
    errors: list[Event] = []

    async def on_error(event: Event) -> None:
        errors.append(event)

    bus.subscribe(EventType.CORE_ERROR, on_error)
    chunks: list[str] = []
    first_chunk = asyncio.Event()

    async def on_chunk(event: Event) -> None:
        chunks.append(str(event.payload.get("text")))
        first_chunk.set()

    bus.subscribe(EventType.COGNITION_RESPONSE_CHUNK, on_chunk)
    cascade = build_cascade(bus, lambda: config)
    assert cascade._local is not None
    await cascade._local.warm_up("x")

    async def killer() -> None:
        # Espera a que salga la primera frase y entonces mata el servidor: el
        # criterio es "a mitad de una respuesta", no antes de empezar.
        await first_chunk.wait()
        print("  -> matando Ollama ...", flush=True)
        for image in ("ollama*", "llama-server*"):
            await asyncio.to_thread(
                subprocess.run, ["taskkill", "/F", "/IM", image], capture_output=True, check=False
            )

    task = asyncio.create_task(killer())
    response = await cascade.handle_text(
        "Explícame en cinco oraciones, paso a paso, cómo funciona la fotosíntesis"
    )
    await task
    await bus.wait_idle(2)
    print(f"  Respuesta: {response.text}")
    print(f"  Frases dichas: {chunks}")
    after = await cascade.handle_text("¿Qué hora es?")
    print(f"  Siguiente orden: {after.text}")
    print(f"  core.error publicados: {len(errors)}")
    await cascade.stop()
    await bus.stop()
    print("  Reiniciando Ollama ...")
    # Desacoplado de este proceso: un hijo de asyncio muere al cerrarse el bucle.
    flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(
        subprocess, "CREATE_NEW_PROCESS_GROUP", 0
    )
    await asyncio.to_thread(
        subprocess.Popen,
        [_ollama_exe(), "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=flags,
    )
    ok = bool(after.text) and not errors
    print("OK: nada se cayo." if ok else "FALLO")
    return 0 if ok else 1


async def privacy_check() -> int:
    """Criterio: con LOCAL_ONLY nada sale a la red.

    Dos observadores independientes, porque uno solo puede estar ciego:

        asyncio   cada `loop.create_connection` (por ahi pasan httpx2 y el SDK
                  de Anthropic), con el host antes de resolverlo
        sistema   `netstat` cada 250 ms filtrado por el PID de este proceso

    Control positivo: las conexiones a Ollama en 127.0.0.1 tienen que aparecer.
    Si no aparecen, los observadores no ven nada y la prueba no vale.
    """
    import os
    import threading

    pid = str(os.getpid())
    attempted: set[str] = set()
    seen: set[str] = set()
    stop = threading.Event()
    real_create = asyncio.BaseEventLoop.create_connection

    async def spy(self: asyncio.BaseEventLoop, *args: Any, **kwargs: Any) -> Any:
        host = kwargs.get("host", args[1] if len(args) > 1 else None)
        attempted.add(str(host))
        return await real_create(self, *args, **kwargs)

    def watch() -> None:
        while not stop.is_set():
            out = subprocess.run(
                ["netstat", "-ano", "-p", "TCP"], capture_output=True, text=True, check=False
            ).stdout
            for line in out.splitlines():
                parts = line.split()
                if len(parts) >= 5 and parts[-1] == pid:
                    seen.add(parts[2].rsplit(":", 1)[0])
            stop.wait(0.25)

    asyncio.BaseEventLoop.create_connection = spy  # type: ignore[method-assign]
    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    # Una clave falsa: si la cascada intentara usar la nube, lo haria con esta.
    os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-falsa-para-probar-que-no-se-usa")
    try:
        await run_cascade(limit=None, privacy=PrivacyMode.LOCAL_ONLY)
    finally:
        asyncio.BaseEventLoop.create_connection = real_create  # type: ignore[method-assign]
        stop.set()
        watcher.join()
    local = {"127.0.0.1", "::1", "[::1]", "localhost", "0.0.0.0", "[::]"}
    print(f"\nIntentos de conexion (asyncio): {sorted(attempted)}")
    print(f"Conexiones vistas por el sistema: {sorted(seen)}")
    if not attempted & local or not seen & local:
        print("NO CONCLUYENTE: los observadores no vieron ni a Ollama.")
        return 2
    outside = (attempted | seen) - local
    print(
        "OK: ni un byte salio de la maquina."
        if not outside
        else f"FALLO: salio trafico a {outside}"
    )
    return 0 if not outside else 1


def main() -> int:
    """Punto de entrada."""
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", action="store_true", help="Ejecutar la cascada completa.")
    parser.add_argument("--limit", type=int, help="Solo las primeras N ordenes.")
    parser.add_argument("--kill-ollama", action="store_true", help="Prueba de caos.")
    parser.add_argument("--privacy", action="store_true", help="LOCAL_ONLY vigilando sockets.")
    args = parser.parse_args()
    configure_logging(level="WARNING", development=True, log_file=None)
    if args.kill_ollama:
        return asyncio.run(kill_ollama_midway())
    if args.privacy:
        return asyncio.run(privacy_check())
    if args.run:
        return asyncio.run(run_cascade(args.limit, None))
    return route_only()


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        sys.exit(main())
