"""Las frases del nivel 1: cortas, secas y variadas.

Un asistente que dice exactamente lo mismo cada vez se siente un contestador.
Cada tipo de confirmacion tiene un grupo de 3-4 frases que rotan sin repetir
la ultima. La personalidad manda: nada de "¡Claro que si!", nada de
exclamaciones, nada de servilismo.
"""

from __future__ import annotations

import random
from typing import Final

__all__ = ["PHRASES", "PhrasePicker"]

PHRASES: Final[dict[str, tuple[str, ...]]] = {
    "volume_up": ("Volumen arriba.", "Subido.", "Más fuerte.", "Listo."),
    "volume_down": ("Volumen abajo.", "Bajado.", "Más bajo.", "Hecho."),
    "volume_set": ("Volumen al {value} por ciento.", "En {value}.", "Queda en {value}."),
    "mute": ("Silenciado.", "Sin sonido.", "Audio en silencio."),
    "unmute": ("Sonido de vuelta.", "Audio activo.", "Quitado el silencio."),
    "play": ("Reproduciendo.", "En marcha.", "Va."),
    "pause": ("En pausa.", "Pausado.", "Detenido."),
    "next": ("Siguiente.", "Saltada.", "La que sigue."),
    "previous": ("Anterior.", "De vuelta a la anterior.", "Atrás."),
    "open": ("Abriendo {app}.", "{app}, en camino.", "Abro {app}."),
    "close": ("Cerrando {app}.", "{app} cerrado.", "Cierro {app}."),
    "minimize_all": ("Escritorio despejado.", "Todo minimizado.", "Listo."),
    "close_all": ("Ventanas cerradas.", "Todo cerrado.", "Hecho."),
    "switch": ("Cambiado.", "Ventana anterior.", "Listo."),
    "screenshot": ("Captura tomada.", "Pantalla guardada.", "Hecho."),
    "lock": ("Bloqueando.", "Sesión bloqueada.", "Bloqueado."),
    "brightness_up": ("Más brillo.", "Brillo arriba.", "Subido."),
    "brightness_down": ("Menos brillo.", "Brillo abajo.", "Bajado."),
    "brightness_set": ("Brillo al {value} por ciento.", "Brillo en {value}.", "Queda en {value}."),
    "timer_set": ("Temporizador de {duration}.", "{duration}, contando.", "Listo: {duration}."),
    "timer_set_label": ("{duration} para {label}.", "Te aviso en {duration}: {label}."),
    "timer_done": ("Tiempo.", "Se cumplió el tiempo.", "Ya está el temporizador."),
    "timer_done_label": ("Tiempo: {label}.", "Ya es hora: {label}."),
    "timer_cancel": ("Temporizador cancelado.", "Cancelado.", "Fuera el temporizador."),
    "timer_cancel_many": ("Cancelados los {count} temporizadores.", "Fuera los {count}."),
    "timer_none": ("No hay temporizadores activos.", "No hay ninguno corriendo."),
    "timer_left": ("Faltan {duration}.", "Quedan {duration}."),
    "cancel": ("Cancelado.", "Olvidado.", "Descartado."),
    "cancel_nothing": ("No había nada en curso.", "Nada que cancelar."),
    "repeat_nothing": ("Todavía no he dicho nada.", "No tengo nada que repetir."),
    "privacy_local": (
        "Modo local. Nada sale de esta máquina.",
        "Nube desactivada. Todo se queda aquí.",
    ),
    "privacy_hybrid": ("Modo normal. La nube vuelve a estar disponible.", "Nube reactivada."),
    "privacy_same": ("Ya estaba en ese modo.", "Sin cambios: ya era así."),
    "greeting": ("Aquí estoy.", "Dime, {name}.", "Buenas, {name}.", "A la orden."),
    "thanks": ("Para eso estoy.", "Cuando quieras.", "No hay de qué."),
    "goodbye": ("Hasta luego, {name}.", "Aquí estaré.", "Hasta luego."),
    "ask_value": ("¿A qué porcentaje?", "¿Cuánto exactamente?"),
    "ask_app": ("¿Qué aplicación?", "¿Cuál?"),
    "ask_duration": ("¿Por cuánto tiempo?", "¿Cuánto tiempo le pongo?"),
    "unavailable": (
        "Entendido: {what}. Todavía no tengo control de eso.",
        "{what}: aún no tengo acceso a esa parte del sistema.",
        "Sé que quieres {what}, pero ese control todavía no existe.",
    ),
    "tool_failed": ("No pude {what}: {error}.", "Falló al {what}: {error}."),
}


class PhrasePicker:
    """Elige una frase del grupo sin repetir la anterior de ese mismo grupo."""

    def __init__(self, rng: random.Random | None = None) -> None:
        """Args: rng, generador inyectable para que las pruebas sean deterministas."""
        self._rng = rng or random.Random()
        self._last: dict[str, int] = {}

    def pick(self, key: str, **values: object) -> str:
        """Una frase del grupo `key` con los huecos rellenos."""
        pool = PHRASES[key]
        choices = [i for i in range(len(pool)) if i != self._last.get(key)] or [0]
        index = self._rng.choice(choices)
        self._last[key] = index
        return pool[index].format(**values)
