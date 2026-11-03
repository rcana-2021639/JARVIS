"""Como se dicen en voz alta las horas, fechas, duraciones y numeros.

Piper lee lo que recibe: "15:25" suena a "quince dos puntos veinticinco".
Estas funciones escriben como habla una persona.
"""

from __future__ import annotations

from datetime import datetime

__all__ = ["local_now", "spoken_date", "spoken_duration", "spoken_number", "spoken_time"]

_WEEKDAYS = ("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo")
_MONTHS = (
    "enero", "febrero", "marzo", "abril", "mayo", "junio",
    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
)  # fmt: skip


def local_now() -> datetime:
    """El instante actual en la zona horaria de la maquina."""
    return datetime.now().astimezone()


def spoken_time(moment: datetime) -> str:
    """La hora como se dice: "Son las 3 y 25 de la tarde"."""
    hour = moment.hour % 12 or 12
    minutes = moment.minute
    head = "Es la 1" if hour == 1 else f"Son las {hour}"
    tail = " en punto" if minutes == 0 else f" y {minutes}"
    period = (
        "de la madrugada" if moment.hour < 6
        else "de la mañana" if moment.hour < 12
        else "de la tarde" if moment.hour < 20
        else "de la noche"
    )  # fmt: skip
    return f"{head}{tail} {period}."


def spoken_date(moment: datetime) -> str:
    """La fecha como se dice: "Hoy es lunes 2 de noviembre"."""
    return f"Hoy es {_WEEKDAYS[moment.weekday()]} {moment.day} de {_MONTHS[moment.month - 1]}."


def spoken_number(value: float) -> str:
    """Un numero como se dice: enteros sin decimales, coma decimal, sin ceros de sobra."""
    if abs(value - round(value)) < 1e-9:
        return str(round(value))
    return f"{value:.4f}".rstrip("0").rstrip(".").replace(".", ",")


def spoken_duration(seconds: float) -> str:
    """Una duracion como se dice: "5 minutos", "1 hora y 10 minutos", "30 segundos"."""
    total = max(0, round(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    parts: list[str] = []
    if hours:
        parts.append(f"{hours} hora" + ("s" if hours != 1 else ""))
    if minutes:
        parts.append(f"{minutes} minuto" + ("s" if minutes != 1 else ""))
    if secs and not hours:
        parts.append(f"{secs} segundo" + ("s" if secs != 1 else ""))
    if not parts:
        return "0 segundos"
    return " y ".join(parts) if len(parts) <= 2 else f"{parts[0]}, {parts[1]} y {parts[2]}"
