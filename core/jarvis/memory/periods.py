"""Periodos dichos en voz: "ayer", "el martes", "esta semana", "anoche"."""

from __future__ import annotations

import re
from datetime import datetime, timedelta

from jarvis.cognition.text import normalize

__all__ = ["period_bounds"]

_WEEKDAYS = ("lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo")
_DAYS_AGO = re.compile(r"\bhace (\d+) dias?\b")


def _midnight(moment: datetime) -> datetime:
    return moment.replace(hour=0, minute=0, second=0, microsecond=0)


def period_bounds(when: str, now: datetime) -> tuple[datetime, datetime, str]:
    """(inicio, fin, como decirlo) de un periodo, en la hora local de `now`.

    Lo que no se entiende es "hoy": es la lectura mas comun de "¿que hicimos?".
    """
    text = normalize(when)
    local = now.astimezone()
    today = _midnight(local)
    day = timedelta(days=1)
    if "anteayer" in text or "antier" in text or "antes de ayer" in text:
        return today - 2 * day, today - day, "Anteayer"
    if "anoche" in text:
        return today - day + timedelta(hours=18), today + timedelta(hours=6), "Anoche"
    if "ayer" in text:
        return today - day, today, "Ayer"
    if "semana pasada" in text:
        monday = today - timedelta(days=today.weekday())
        return monday - 7 * day, monday, "La semana pasada"
    if "semana" in text:
        return today - timedelta(days=today.weekday()), local, "Esta semana"
    if "mes" in text:
        return today.replace(day=1), local, "Este mes"
    ago = _DAYS_AGO.search(text)
    if ago:
        start = today - int(ago.group(1)) * day
        return start, start + day, f"Hace {ago.group(1)} días"
    for index, name in enumerate(_WEEKDAYS):
        if re.search(rf"\b{name}\b", text):
            back = (today.weekday() - index) % 7 or 7
            start = today - back * day
            return (
                start,
                start + day,
                f"El {name}".replace("miercoles", "miércoles").replace("sabado", "sábado"),
            )
    if "manana" in text:
        return today + timedelta(hours=5), today + timedelta(hours=12), "Esta mañana"
    if "tarde" in text:
        return today + timedelta(hours=12), today + timedelta(hours=19), "Esta tarde"
    if "noche" in text:
        return today + timedelta(hours=19), today + day, "Esta noche"
    return today, local + timedelta(seconds=1), "Hoy"
