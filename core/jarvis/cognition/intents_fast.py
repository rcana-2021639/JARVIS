"""Las intenciones de forma fija: el nivel 1, sin modelo, ~50 ms, costo cero.

Meta del documento maestro: que absorban la mitad de las ordenes. Todos los
patrones trabajan sobre texto normalizado (sin acentos, numeros en cifras).

Dentro de cada dominio, las que llevan numero van antes que las genericas:
"sube el volumen al 70" es fijar, no subir.
"""

from __future__ import annotations

from jarvis.cognition.catalog import (
    IntentSpec,
    entity_app,
    entity_expression,
    entity_query,
    entity_timer,
    entity_value,
    spec,
)
from jarvis.contracts.enums import CascadeLevel

__all__ = ["FAST_INTENTS"]

L1 = CascadeLevel.FAST_PATH

_VOL = r"(?:volumen|sonido|audio)"
_UP = r"(?:sube|subele|subir|aumenta|aumentale|incrementa|mas)"
_DOWN = r"(?:baja|bajale|bajar|disminuye|reduce|reducele|menos)"
_BRIGHT = r"(?:brillo|pantalla)"
_TIMER = r"(?:temporizador|timer|alarma|cronometro|cuenta regresiva)"
_APP_VERB_OPEN = r"(?:abre|abreme|abrir|inicia|iniciame|lanza|ejecuta|arranca|pon|prende)"
_APP_VERB_CLOSE = r"(?:cierra|cierrame|cerrar|sierra|termina|mata|quita)"

FAST_INTENTS: tuple[IntentSpec, ...] = (
    # --- Multimedia -------------------------------------------------------
    spec(
        "media.volume_set", L1, "Fija el volumen en un porcentaje.",
        patterns=[rf"\b{_VOL}\b.*\b\d{{1,3}}\b", rf"\b(?:pon|ajusta|deja)\b.*\b{_VOL}\b.*\d"],
        phrases=["pon el volumen al 50", "volumen al 30%"],
        examples=["deja el volumen en 40", "ajusta el sonido a 20 por ciento"],
        extract=entity_value,
    ),
    spec(
        "media.volume_up", L1, "Sube el volumen.",
        patterns=[rf"\b{_UP}\b.*\b{_VOL}\b", r"\bmas (?:fuerte|alto)\b", r"^subele$"],
        phrases=["sube el volumen", "subele al volumen", "mas volumen", "mas fuerte"],
        examples=["no se escucha nada", "aumenta el sonido un poco"],
        extract=entity_value,
    ),
    spec(
        "media.volume_down", L1, "Baja el volumen.",
        patterns=[rf"\b{_DOWN}\b.*\b{_VOL}\b", r"\bmas (?:bajo|despacio|quedito)\b", r"^bajale$"],
        phrases=["baja el volumen", "bajale al volumen", "menos volumen", "mas bajo"],
        examples=["esta muy fuerte", "baja un poco el sonido"],
        extract=entity_value,
    ),
    spec(
        "media.mute", L1, "Silencia el audio del sistema.",
        patterns=[rf"\bsilencia\b.*\b{_VOL}\b", r"\bmutea\b", rf"\bquita el {_VOL}\b", r"^silencia(?: todo)?$"],
        phrases=["silencia el audio", "mutea todo", "quita el sonido"],
        examples=["pon en silencio la computadora"],
    ),
    spec(
        "media.unmute", L1, "Quita el silencio del sistema.",
        patterns=[r"\b(?:quita|desactiva) el silencio\b", rf"\b(?:activa|devuelve|regresa) el {_VOL}\b", r"\bdesmutea\b"],
        phrases=["quita el silencio", "activa el sonido", "desmutea"],
        examples=["vuelve a poner el sonido"],
    ),
    spec(
        "media.pause", L1, "Pausa la reproduccion.",
        patterns=[r"^(?:pausa|pausala|pausalo|stop)\b", r"\b(?:deten|para|pausa)\b.*\b(?:musica|cancion|video|reproduccion)\b"],
        phrases=["pausa", "pausa la musica", "deten la cancion"],
        examples=["para el video", "detente con la musica"],
    ),
    spec(
        "media.next", L1, "Salta a la siguiente pista.",
        patterns=[r"\bsiguiente\b", r"^salta\b", r"\bla que sigue\b", r"\b(?:pasa|cambia) (?:de |la )?cancion\b"],
        phrases=["siguiente cancion", "salta esta", "la que sigue"],
        examples=["no me gusta esta cancion, pasala"],
    ),
    spec(
        "media.previous", L1, "Vuelve a la pista anterior.",
        patterns=[r"\banterior\b", r"\bla de antes\b"],
        phrases=["cancion anterior", "regresa a la anterior", "la de antes"],
        examples=["pon otra vez la que estaba antes"],
    ),
    spec(
        "media.play", L1, "Reanuda o reproduce algo.",
        patterns=[
            r"^(?:reproduce|toca) (?P<query>.+)$",
            r"^(?:pon|ponme) (?:musica|una cancion|algo de musica|la cancion|el disco|el album)(?: de)? (?P<query>.+)$",
            r"^(?:reanuda|continua|play|dale play|reproduce|pon musica|ponme musica|pon algo de musica)$",
        ],
        phrases=["reanuda la musica", "dale play", "pon musica"],
        examples=["sigue con la cancion", "quiero escuchar musica"],
        extract=entity_query,
    ),
    # --- Ventanas ------------------------------------------------------------
    spec(
        "windows.minimize_all", L1, "Minimiza todo y muestra el escritorio.",
        patterns=[r"\bminimiza (?:todo|todas)\b", r"\bmuestra(?:me)? el escritorio\b"],
        phrases=["minimiza todo", "muestra el escritorio"],
        examples=["esconde todas las ventanas"],
    ),
    spec(
        "windows.close_all", L1, "Cierra todas las ventanas abiertas.",
        patterns=[rf"\b{_APP_VERB_CLOSE} todas? (?:las )?ventanas\b"],
        phrases=["cierra todas las ventanas"],
        examples=["cierra todo lo que tengo abierto"],
    ),
    spec(
        "windows.switch", L1, "Cambia a la ventana anterior.",
        patterns=[r"\bcambia de ventana\b", r"\bventana anterior\b", r"\balt tab\b"],
        phrases=["cambia de ventana", "ve a la ventana anterior"],
        examples=["regresa a lo que tenia antes"],
    ),
    # --- Sistema -----------------------------------------------------------
    spec(
        "system.screenshot", L1, "Toma una captura de pantalla.",
        patterns=[r"\b(?:captura|screenshot|pantallazo|foto) (?:de )?(?:la )?pantalla\b", r"\bscreenshot\b", r"\bpantallazo\b"],
        phrases=["toma una captura de pantalla", "haz un screenshot"],
        examples=["guarda lo que hay en la pantalla"],
    ),
    spec(
        "system.lock", L1, "Bloquea la sesion.",
        patterns=[r"\bbloquea\b(?: (?:la|el) (?:pc|computadora|pantalla|equipo|sesion|compu))?$"],
        phrases=["bloquea la computadora", "bloquea la pantalla"],
        examples=["me voy, cierra la sesion con candado"],
    ),
    spec(
        "system.brightness_set", L1, "Fija el brillo en un porcentaje.",
        patterns=[r"\bbrillo\b.*\b\d{1,3}\b"],
        phrases=["pon el brillo al 50", "brillo al 80%"],
        examples=["ajusta la luminosidad a 30"],
        extract=entity_value,
    ),
    spec(
        "system.brightness_up", L1, "Sube el brillo.",
        patterns=[rf"\b{_UP}\b.*\b{_BRIGHT}\b", r"\bmas brillo\b"],
        phrases=["sube el brillo", "mas brillo"],
        examples=["no veo nada, la pantalla esta muy oscura"],
        extract=entity_value,
    ),
    spec(
        "system.brightness_down", L1, "Baja el brillo.",
        patterns=[rf"\b{_DOWN}\b.*\b{_BRIGHT}\b", r"\bmenos brillo\b", r"\bmaja el brillo\b"],
        phrases=["baja el brillo", "menos brillo"],
        examples=["la pantalla me deslumbra"],
        extract=entity_value,
    ),
    spec(
        "system.time", L1, "Dice la hora.",
        patterns=[r"\bque hora\b", r"\bdime la hora\b", r"^(?:la )?hora$", r"\bhora (?:es|actual)\b"],
        phrases=["que hora es", "dime la hora"],
        examples=["tienes la hora", "sabes que hora es"],
    ),
    spec(
        "system.date", L1, "Dice la fecha.",
        patterns=[r"\bque (?:dia|fecha) es\b", r"\ba que (?:dia )?estamos\b", r"\bfecha de hoy\b", r"\bque dia es hoy\b"],
        phrases=["que dia es hoy", "que fecha es"],
        examples=["a cuantos estamos"],
    ),
    # --- Cuentas ----------------------------------------------------------------
    spec(
        "query.arithmetic", L1, "Una cuenta simple, sin modelo.",
        patterns=[r"^(?:cuanto (?:es|son|da)|calcula|cuanto seria) (?:el |la )?(?P<expr>\d[\d\s.,%]*(?:\s*(?:por|x|mas|menos|entre|sobre|dividido (?:entre|por)|elevado a|multiplicado por|% de)\s*\d[\d\s.,%]*)+)$"],
        phrases=["cuanto es 25 por 4"],
        examples=["calcula 15 mas 27"],
        extract=entity_expression,
    ),
    # --- Temporizadores ------------------------------------------------------
    spec(
        "timer.cancel", L1, "Cancela los temporizadores.",
        patterns=[rf"\b(?:cancela|quita|borra|deten|para|apaga|elimina)\b.*\b{_TIMER}"],
        phrases=["cancela el temporizador", "quita la alarma"],
        examples=["ya no necesito el timer"],
    ),
    spec(
        "timer.query", L1, "Dice cuanto falta.",
        patterns=[r"\bcuanto (?:le )?(?:falta|queda)\b", rf"\b(?:como va|estado del?) (?:el )?{_TIMER}"],
        phrases=["cuanto falta", "cuanto le queda al temporizador"],
        examples=["como va el timer"],
    ),
    spec(
        "timer.set", L1, "Programa un temporizador.",
        patterns=[
            rf"\b{_TIMER}\b.*\b\d+\s*(?:segundos?|minutos?|horas?|seg|min)\b",
            r"\b(?:avisame|recuerdame|despiertame)\b.*\ben \d+\s*(?:segundos?|minutos?|horas?)\b",
            rf"\b{_TIMER}\b.*\b(?:media hora|cuarto de hora)\b",
        ],
        phrases=["pon un temporizador de 5 minutos", "avisame en 10 minutos"],
        examples=["cuenta 3 minutos para los huevos", "pon una alarma en media hora"],
        extract=entity_timer,
    ),
    # --- Control del propio Jarvis -------------------------------------------
    spec(
        "jarvis.stop_talking", L1, "Callarse ya.",
        patterns=[r"^(?:callate|silencio|basta|ya|suficiente|shh+|alto|para)$", r"\bpara de hablar\b", r"\bdeja de hablar\b"],
        phrases=["callate", "para de hablar", "silencio"],
        examples=["ya no hables", "no digas nada"],
    ),
    spec(
        "jarvis.cancel", L1, "Cancela lo que este en curso.",
        patterns=[r"^(?:cancela|cancelalo|olvidalo|olvida eso|nada|no importa|dejalo|anula|olvidate)$"],
        phrases=["cancela", "olvidalo", "no importa"],
        examples=["mejor no, dejalo asi"],
    ),
    spec(
        "jarvis.repeat", L1, "Repite la ultima respuesta.",
        patterns=[r"\brepite(?:lo|me)?\b", r"^repetir(?:lo|me)?$", r"\bque dijiste\b", r"\bno te (?:entendi|escuche|oi)\b", r"^otra vez$"],
        phrases=["repite", "que dijiste", "puedes repetir"],
        examples=["no te escuche bien"],
    ),
    spec(
        "jarvis.privacy_local", L1, "Pasa a modo solo-local.",
        patterns=[r"\bmodo (?:local|privado|offline|sin conexion|solo local)\b", r"\b(?:desactiva|apaga|corta|bloquea) la nube\b", r"\bno uses (?:la )?(?:nube|internet)\b"],
        phrases=["modo local", "desactiva la nube", "modo privado"],
        examples=["a partir de ahora nada sale de la computadora"],
    ),
    spec(
        "jarvis.privacy_hybrid", L1, "Vuelve al modo normal hibrido.",
        patterns=[r"\bmodo (?:normal|hibrido|mixto)\b", r"\b(?:activa|enciende|reactiva) la nube\b", r"\bvuelve a usar la nube\b", r"\bpuedes usar la nube\b"],
        phrases=["modo normal", "activa la nube", "modo hibrido"],
        examples=["ya puedes conectarte otra vez"],
    ),
    spec(
        "jarvis.budget", L1, "Reporta el gasto del dia en la nube.",
        patterns=[r"\bcuanto (?:he |has |llevo |llevas |llevamos )?gastado\b", r"\b(?:gasto|presupuesto|costo) (?:de|del) (?:hoy|dia)\b", r"\bcuanto me (?:has )?costado\b"],
        phrases=["cuanto llevamos gastado hoy", "gasto de hoy"],
        examples=["cuanto me ha costado la nube"],
    ),
    spec(
        "jarvis.status", L1, "Reporta el estado de la cascada.",
        patterns=[r"\b(?:estas ahi|sigues ahi|me escuchas)\b", r"\b(?:estado|diagnostico|reporte) del sistema\b", r"^(?:estado|diagnostico|status)$"],
        phrases=["estas ahi", "estado del sistema"],
        examples=["funciona todo bien"],
    ),
    # --- Cortesia --------------------------------------------------------------
    spec(
        "chat.greeting", L1, "Saludo.",
        patterns=[r"^(?:hola|buenos dias|buenas tardes|buenas noches|buenas|hey|que tal|que onda)$"],
        phrases=["hola", "buenos dias", "que tal"],
        examples=["hola que tal"],
    ),
    spec(
        "chat.thanks", L1, "Agradecimiento.",
        patterns=[r"^(?:muchas )?gracias(?: jarvis)?$", r"^(?:perfecto|genial|excelente|listo),? gracias$", r"^te lo agradezco$"],
        phrases=["gracias", "muchas gracias"],
        examples=["te lo agradezco mucho"],
    ),
    spec(
        "chat.goodbye", L1, "Despedida.",
        patterns=[r"^(?:adios|hasta luego|nos vemos|hasta manana|eso es todo|bye)$"],
        phrases=["adios", "hasta luego", "eso es todo"],
        examples=["me voy a dormir, hasta manana"],
    ),
    # --- Aplicaciones: al final, porque "pon X" y "quita X" son muy generales ---
    spec(
        "apps.open", L1, "Abre una aplicacion por su nombre.",
        patterns=[rf"^{_APP_VERB_OPEN} (?P<app>(?!musica\b|una cancion\b|algo\b|un (?:temporizador|timer)\b).+)$"],
        phrases=["abre spotify", "abre el navegador", "abre la calculadora"],
        examples=["inicia visual studio code", "lanza discord"],
        extract=entity_app,
    ),
    spec(
        "apps.close", L1, "Cierra una aplicacion por su nombre.",
        patterns=[rf"^{_APP_VERB_CLOSE} (?P<app>(?!todas? (?:las )?ventanas).+)$"],
        phrases=["cierra spotify", "cierra el navegador"],
        examples=["mata el proceso de chrome", "termina discord"],
        extract=entity_app,
    ),
)  # fmt: skip
