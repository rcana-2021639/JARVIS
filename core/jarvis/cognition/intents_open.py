"""Las intenciones abiertas: lo que necesita un modelo de lenguaje.

Van al nivel 2 (Ollama) salvo las que por naturaleza piden mas: escribir o
refactorizar codigo y planear tareas de muchos pasos sugieren el nivel 3. El
enrutador puede corregir la sugerencia con la complejidad real de la orden.

`PRIORITY_OPEN_INTENTS` lleva patrones muy especificos que deben ganarle a las
fijas generales: "abre el archivo X" no es abrir una aplicacion llamada
"el archivo X".
"""

from __future__ import annotations

from jarvis.cognition.catalog import IntentSpec, entity_query, spec
from jarvis.contracts.enums import CascadeLevel

__all__ = ["OPEN_INTENTS", "PRIORITY_OPEN_INTENTS"]

L2 = CascadeLevel.LOCAL_LLM
L3 = CascadeLevel.CLOUD_LLM

PRIORITY_OPEN_INTENTS: tuple[IntentSpec, ...] = (
    spec(
        "cloud.explicit", L3, "El usuario pide la nube o razonamiento profundo.",
        patterns=[
            r"\b(?:preguntale|pregunta|consulta) (?:a )?claude\b",
            r"\busa (?:la nube|claude|opus|sonnet)\b",
            r"\b(?:piensa|razona|analiza) (?:a fondo|con calma|profundamente)\b",
        ],
        examples=["usa la nube para esto", "quiero la respuesta de Claude"],
    ),
    spec(
        "code.refactor", L3, "Analizar, revisar o refactorizar codigo existente.",
        patterns=[r"\brefactoriza", r"\b(?:analiza|revisa|depura|optimiza)\b.*\b(?:codigo|archivo|script|funcion|clase|programa)\b"],
        examples=["encuentra el bug de este codigo", "revisa este script de python"],
    ),
    spec(
        "code.write", L3, "Escribir codigo nuevo.",
        patterns=[r"\b(?:escribe|hazme|crea|programa|genera)\b.*\b(?:script|codigo|programa|funcion|clase|consulta sql|regex|expresion regular)\b"],
        examples=["haz un script que renombre fotos", "programa una funcion de ordenamiento"],
    ),
    spec(
        "files.open", L2, "Abrir un archivo o carpeta concretos.",
        patterns=[r"^(?:abre|abreme|muestrame) (?:el |la |mi |mis )?(?:archivo|documento|carpeta|pdf|foto|imagen) (?P<query>.+)$"],
        examples=["abre el documento del presupuesto", "muestrame la carpeta de descargas"],
        extract=entity_query,
    ),
    spec(
        "browser.open", L2, "Abrir una pagina web.",
        patterns=[r"^(?:abre|entra a|ve a|abreme) (?:la pagina de |el sitio de |la web de )?(?P<query>\S+\.(?:com|org|net|io|dev|mx|es)\S*)$", r"^(?:abre|entra a|ve a) (?:la pagina|el sitio|la web) del? (?P<query>.+)$"],
        examples=["entra a youtube.com", "abre la pagina del banco"],
        extract=entity_query,
    ),
    spec(
        "browser.search", L2, "Buscar algo en internet.",
        patterns=[r"\bbusca (?:en (?:google|internet|la web)|en linea) (?P<query>.+)$", r"\bgooglea (?P<query>.+)$"],
        examples=["busca en google recetas de pasta", "investiga en internet precios de laptops"],
        extract=entity_query,
    ),
    spec(
        "files.search", L2, "Buscar un archivo en el disco.",
        patterns=[r"\b(?:busca|encuentra|donde (?:esta|quedo|guarde))\b.*\b(?:archivo|documento|carpeta|pdf|foto)\b"],
        examples=["donde guarde la factura de luz", "encuentra mis fotos de diciembre"],
    ),
)  # fmt: skip

OPEN_INTENTS: tuple[IntentSpec, ...] = (
    spec(
        "memory.recall", L2, "Consultar algo que el usuario conto antes.",
        patterns=[r"\bque te (?:dije|conte) (?:de|sobre)\b", r"\bte acuerdas (?:de|que)\b", r"^recuerdas\b"],
        examples=[
            "como te dije que se llamaba mi perro", "que habiamos dicho de la reunion",
            "como se llama mi hermana", "donde vive mi mama", "cual es el nombre de mi jefe",
        ],
    ),
    spec(
        "chat.opinion", L2, "Pedir la opinion de Jarvis.",
        patterns=[r"\bque (?:opinas|piensas) (?:de|sobre|del)\b", r"\btu opinion (?:de|sobre)\b", r"\bque te parece\b"],
        examples=["te gusta el jazz", "crees que vale la pena aprender rust"],
    ),
    spec(
        "query.knowledge", L2, "Pregunta de cultura general o explicacion.",
        patterns=[r"^(?:que|quien|quienes|cual|cuales|cuando|donde) (?:es|son|fue|fueron|era|esta|queda)\b", r"^(?:explicame|como funciona|por que)\b"],
        examples=["cual es la capital de Mongolia", "explicame como funciona un motor electrico"],
    ),
    spec(
        "query.calculation", L2, "Una cuenta o conversion.",
        patterns=[r"^(?:cuanto es|calcula|cuanto da)\b", r"\b\d+ (?:por|mas|menos|entre|x) \d+\b"],
        examples=["cuanto es el 15 por ciento de 230", "convierte 30 grados a fahrenheit"],
    ),
    spec(
        "query.weather", L2, "El clima. Necesita la web (Fase 7).",
        patterns=[r"\b(?:clima|temperatura afuera|va a llover|pronostico)\b"],
        examples=["como esta el clima hoy", "necesito paraguas manana"],
    ),
    spec(
        "query.news", L2, "Noticias de actualidad. Necesita la web (Fase 7).",
        patterns=[r"\b(?:noticias|que paso hoy|ultima hora)\b"],
        examples=["que esta pasando en el mundo", "dame los titulares"],
    ),
    spec(
        "writing.draft", L2, "Redaccion corta: correo, mensaje, nota.",
        patterns=[r"\b(?:redacta|escribe|escribeme)\b.*\b(?:correo|mail|mensaje|carta|nota|tweet|post|respuesta)\b"],
        examples=["ayudame a contestarle a mi jefe que llego tarde", "escribe un mensaje de cumpleanos"],
    ),
    spec(
        "writing.translate", L2, "Traducir un texto.",
        patterns=[r"\btraduce\b", r"\bcomo se dice\b.*\ben (?:ingles|frances|aleman|italiano|portugues|espanol)\b"],
        examples=["como se dice gracias en japones", "pasa esto al ingles"],
    ),
    spec(
        "files.summarize", L2, "Resumir un documento.",
        patterns=[r"\bresume\b", r"\bresumen (?:de|del)\b"],
        examples=["de que trata este pdf", "dame los puntos clave del informe"],
    ),
    spec(
        "task.plan", L3, "Planear algo de muchos pasos.",
        patterns=[r"\b(?:planea|planifica|organiza|arma)(?:me|mela|lo|la)?\b", r"\bhaz(?:me)? un plan\b"],
        examples=["organizame la semana con estas cinco tareas", "arma un itinerario para tres dias en Roma"],
    ),
    spec(
        "jarvis.capabilities", L2, "Que puede hacer Jarvis.",
        patterns=[r"\bque (?:puedes|sabes) hacer\b", r"\bcuales son tus (?:funciones|capacidades)\b"],
        examples=["en que me puedes ayudar", "para que sirves"],
    ),
    spec(
        "jarvis.identity", L2, "Quien es Jarvis.",
        patterns=[r"\bquien eres\b", r"\bcomo te llamas\b", r"\beres una ia\b"],
        examples=["presentate", "que eres exactamente"],
    ),
    spec(
        "chat.free", L2, "Conversacion libre. El comodin cuando nada mas encaja.",
        examples=["me siento cansado hoy", "cuentame algo interesante", "estoy aburrido", "tuve un dia largo"],
    ),
)  # fmt: skip
