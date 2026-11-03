"""Sistema de memoria (Fase 3): cinco capas, recuperacion hibrida y consolidacion.

    store.py          SQLite (FTS5) y LanceDB en config/data/, con migraciones
    working.py        ultimos turnos + estado del sistema, en RAM
    episodic.py       todo lo ocurrido, con fecha; BM25 y rangos
    semantic.py       hechos con importancia y decaimiento; vectores
    graph.py          entidades y relaciones tipadas; recorrido a N saltos
    procedural.py     secuencias que funcionaron; rutinas de ruta rapida
    retrieval.py      cuatro busquedas en paralelo + RRF, < 100 ms
    extraction.py     de episodios a hechos: reglas y modelo local
    consolidation.py  el ciclo nocturno y su programacion
    recorder.py       del bus a la memoria
    handlers.py       "recuerda que", "olvida lo de", "que sabes de", "que hicimos"
    periods.py        "ayer", "el martes", "esta semana"
    tools.py          las herramientas de memoria para los modelos
    service.py        la fachada: `MemorySystem`
    factory.py        arma la memoria real y la cascada conectada a ella

Uso tipico:

    cascade, memory = build_assistant(bus, lambda: manager.current)
    await memory.start()
    await cascade.start()
"""

from jarvis.memory.factory import build_assistant, build_memory
from jarvis.memory.service import MemorySystem

__all__ = ["MemorySystem", "build_assistant", "build_memory"]
