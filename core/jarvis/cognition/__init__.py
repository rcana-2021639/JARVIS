"""Orquestador cognitivo (Fase 2): la cascada de tres niveles.

    text.py          normalizacion: acentos, numeros en palabras, entidades
    catalog.py       la definicion de una intencion y sus extractores
    intents_fast.py  las 34 intenciones de forma fija (nivel 1)
    intents_open.py  las 21 abiertas (niveles 2 y 3)
    embeddings.py    MiniLM multilingue en ONNX
    classifier.py    patrones -> difusa -> embeddings, en menos de 10 ms
    fastpath.py      el nivel 1: manejadores deterministas
    phrases.py       confirmaciones variadas que rotan
    timers.py        temporizadores en proceso
    llm.py           la interfaz comun de los niveles 2 y 3
    local_llm.py     el nivel 2: Ollama con confianza propia
    cloud_llm.py     el nivel 3: la API de Claude con costo y tope diario
    budget.py        el gasto del dia, persistido
    streaming.py     JSON incremental y corte en frases
    speaker.py       el canal hablado de una respuesta en curso
    context.py       historial en RAM y contexto con presupuesto de tokens
    router.py        las reglas de la cascada y el modo de privacidad
    levels.py        los caminos de los niveles 2 y 3, con escaladas
    tool_loop.py     el bucle de herramientas, listo para MCP
    demo_tools.py    calculadora y reloj para probar el bucle
    control.py       lo que la ruta rapida consulta y cambia de la sesion
    reporting.py     eventos, estado y contabilidad que la cascada publica
    orchestrator.py  todo conectado al bus
    factory.py       arma la cascada real desde la configuracion
    prompts/         la personalidad de Jarvis

Uso tipico:

    cascade = build_cascade(bus, lambda: manager.current)
    await cascade.start()
"""

from jarvis.cognition.classifier import Classification, IntentClassifier
from jarvis.cognition.factory import build_cascade, build_classifier
from jarvis.cognition.orchestrator import CognitiveOrchestrator

__all__ = [
    "Classification",
    "CognitiveOrchestrator",
    "IntentClassifier",
    "build_cascade",
    "build_classifier",
]
