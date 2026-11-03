"""Episodios simulados: una semana de conversaciones de un usuario ficticio.

Sirve para probar la consolidacion sin esperar a que pase una semana de uso
real (criterio de aceptacion: "100 episodios simulados producen hechos y
entidades coherentes") y para el inspector (`memory_inspect.py --simulate`).

Lo que se deberia aprender de aqui, y lo que verifican las pruebas:

    hechos      tu hermana se llama Ana; tu perro se llama Toby; tu jefe se
                llama Carlos; trabajas en el proyecto Aurora; vives en
                Guadalajara; te gusta el jazz; odias el trafico
    entidades   Ana, Toby, Carlos, Maria, Aurora, informe.docx, Guadalajara
    relaciones  Maria trabaja_en Aurora; Maria usa_archivo informe.docx
    rutina      "que clima hace" se pide cada manana -> se propone
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta

from jarvis.contracts.enums import CascadeLevel

__all__ = ["SimulatedTurn", "simulated_week"]


@dataclass(frozen=True, slots=True)
class SimulatedTurn:
    """Un turno simulado: que dijo el usuario, que contesto Jarvis, cuando."""

    at: datetime
    user: str
    answer: str
    intent: str
    level: CascadeLevel


_FACTS = (
    ("Recuerda que mi hermana se llama Ana", "Anotado: tu hermana.", "memory.remember"),
    ("Mi perro se llama Toby y está enfermo", "Que se mejore Toby.", "chat.free"),
    ("Recuerda que mi jefe se llama Carlos", "Anotado: tu jefe.", "memory.remember"),
    ("Trabajo en el proyecto Aurora, desde marzo", "Entendido.", "chat.free"),
    ("María trabaja en el proyecto Aurora, igual que yo", "Anotado.", "chat.free"),
    ("María usa el archivo informe.docx para los reportes", "Entendido.", "chat.free"),
    ("Vivo en Guadalajara", "Bonita ciudad.", "chat.free"),
    ("Me gusta el jazz para concentrarme", "Buena elección para trabajar.", "chat.free"),
    ("Odio el tráfico de los lunes", "Es comprensible.", "chat.free"),
)  # fmt: skip
_ROUTINE = ("¿Qué clima hace?", "No tengo acceso al clima todavía.", "query.weather")
_FILLER = (
    ("¿Qué hora es?", "Son las 10 y cuarto.", "system.time", CascadeLevel.FAST_PATH),
    ("Sube el volumen", "Volumen arriba.", "media.volume_up", CascadeLevel.FAST_PATH),
    ("Pon un temporizador de 5 minutos", "Temporizador de 5 minutos.", "timer.set",
     CascadeLevel.FAST_PATH),
    ("¿Qué opinas de Rust?", "Es un lenguaje sólido para sistemas.", "chat.opinion",
     CascadeLevel.LOCAL_LLM),
    ("Cuéntame un chiste", "Un byte entra a un bar...", "chat.free", CascadeLevel.LOCAL_LLM),
    ("¿Cuál es la capital de Mongolia?", "Ulán Bator.", "query.knowledge",
     CascadeLevel.LOCAL_LLM),
    ("Gracias", "Para eso estoy.", "chat.thanks", CascadeLevel.FAST_PATH),
    ("¿Cuánto es 15 por 23?", "Son 345.", "query.arithmetic", CascadeLevel.FAST_PATH),
    ("Explícame qué es un transformer", "Una red que atiende a todo el texto a la vez.",
     "query.knowledge", CascadeLevel.LOCAL_LLM),
)  # fmt: skip


def simulated_week(end: datetime, *, total: int = 100, seed: int = 7) -> list[SimulatedTurn]:
    """`total` turnos repartidos en los 7 dias anteriores a `end`, en orden.

    Cada manana se pregunta el clima (patron repetido); los hechos se dicen una
    vez cada uno, mezclados con ordenes cotidianas. Determinista con `seed`.
    """
    rng = random.Random(seed)
    start_day = (end - timedelta(days=7)).replace(hour=0, minute=0, second=0, microsecond=0)
    turns: list[SimulatedTurn] = []
    for day in range(7):
        morning = start_day + timedelta(days=day, hours=8, minutes=rng.randint(0, 20))
        turns.append(SimulatedTurn(morning, _ROUTINE[0], _ROUTINE[1], _ROUTINE[2],
                                   CascadeLevel.LOCAL_LLM))  # fmt: skip
    facts = list(_FACTS)
    remaining = total - len(turns) - len(facts)
    slots = sorted(rng.sample(range(remaining + len(facts)), len(facts)))
    fillers: list[tuple[str, str, str, CascadeLevel]] = [
        rng.choice(_FILLER) for _ in range(remaining)
    ]
    sequence: list[tuple[str, str, str, CascadeLevel]] = []
    for index in range(remaining + len(facts)):
        if slots and index == slots[0]:
            slots.pop(0)
            user, answer, intent = facts.pop(0)
            level = (
                CascadeLevel.FAST_PATH if intent == "memory.remember" else CascadeLevel.LOCAL_LLM
            )
            sequence.append((user, answer, intent, level))
        else:
            sequence.append(fillers.pop())
    # Las ordenes se reparten en dos rafagas por dia, a horas al azar entre las
    # 10 y las 21, con un par de minutos entre turnos: conversaciones reales.
    per_day = len(sequence) // 7 + 1
    for day in range(7):
        chunk = sequence[day * per_day : (day + 1) * per_day]
        half = len(chunk) // 2
        for burst, items in enumerate((chunk[:half], chunk[half:])):
            hour = rng.randint(10, 15) if burst == 0 else rng.randint(16, 21)
            begin = start_day + timedelta(days=day, hours=hour)
            for offset, (user, answer, intent, level) in enumerate(items):
                moment = begin + timedelta(minutes=offset * 2 + rng.randint(0, 1))
                turns.append(SimulatedTurn(moment, user, answer, intent, level))
    return sorted(turns, key=lambda turn: turn.at)
