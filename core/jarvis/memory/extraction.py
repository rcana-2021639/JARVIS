"""De episodios a hechos, entidades y relaciones.

Dos extractores con la misma interfaz:

    RuleExtractor     reglas fijas sobre lo que dijo el usuario: "recuerda
                      que...", "mi hermana se llama...", "X trabaja en Y".
                      Instantaneo y predecible; solo entiende frases tipicas.
    OllamaExtractor   el modelo local lee el tramo de conversacion y devuelve
                      JSON con esquema. Entiende cualquier cosa; tarda
                      segundos por tramo en esta CPU.

La consolidacion usa el modelo local (es trabajo por lotes sin prisa: no tiene
sentido pagar la nube por el) y cae a reglas si Ollama no responde.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import httpx2 as httpx
from pydantic import BaseModel, Field, ValidationError

from jarvis.cognition.prompts import load_prompt
from jarvis.memory.graph import ENTITY_TYPES
from jarvis.memory.text import clip, fold, raw_capture, sentence, to_second_person

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from jarvis.config import CognitionConfig
    from jarvis.memory.episodic import Episode

__all__ = [
    "EntityCandidate",
    "Extraction",
    "ExtractionError",
    "Extractor",
    "FactCandidate",
    "OllamaExtractor",
    "RelationCandidate",
    "RuleExtractor",
    "remembered_fact",
]


@dataclass(frozen=True, slots=True)
class FactCandidate:
    """Un hecho propuesto."""

    content: str
    importance: float


@dataclass(frozen=True, slots=True)
class EntityCandidate:
    """Una entidad propuesta."""

    name: str
    type: str


@dataclass(frozen=True, slots=True)
class RelationCandidate:
    """Una relacion propuesta entre dos entidades."""

    source: str
    relation: str
    target: str
    source_type: str = "cosa"
    target_type: str = "cosa"


@dataclass(slots=True)
class Extraction:
    """Lo que salio de un tramo de episodios."""

    facts: list[FactCandidate] = field(default_factory=list)
    entities: list[EntityCandidate] = field(default_factory=list)
    relations: list[RelationCandidate] = field(default_factory=list)
    summary: str = ""

    def extend(self, other: Extraction) -> None:
        """Junta otra extraccion en esta."""
        self.facts.extend(other.facts)
        self.entities.extend(other.entities)
        self.relations.extend(other.relations)
        if other.summary:
            self.summary = f"{self.summary} {other.summary}".strip()


class ExtractionError(RuntimeError):
    """El extractor no pudo producir un resultado valido."""


class Extractor(Protocol):
    """Lee episodios y propone que recordar."""

    @property
    def name(self) -> str:
        """Para el reporte: "modelo local" o "reglas"."""
        ...

    async def extract(self, episodes: Sequence[Episode]) -> Extraction:
        """Lanza `ExtractionError` si no pudo."""
        ...


# ---------------------------------------------------------------------------
# Reglas
# ---------------------------------------------------------------------------

_REMEMBER = re.compile(
    r"^\W*(?:(?:oye|hey|jarvis|yarvis|por favor)\W+)*"
    r"(?:recuerda|anota|apunta|no olvides|acuerdate de|guarda|memoriza|toma nota de)"
    r"(?:\s+(?:esto|eso))?\W*(?:\s*que)?\s+(?P<fact>.+)$"
)
_KIN = (
    "hermana|hermano|madre|mama|padre|papa|esposa|esposo|pareja|novia|novio|hijo|hija|"
    "jefe|jefa|amigo|amiga|socio|socia|abuela|abuelo|tia|tio|prima|primo|perro|perra|gato|gata"
)
_END = r"(?:[,.;]|\s+y\s|$)"
"""Donde termina un nombre: puntuacion, " y " o el final de la frase."""
_WHO = r"(?P<who>[\w.'-]+(?: [\w.'-]+)?)"
_TARGET = r"(?:el |la )?(?P<kind>proyecto )?(?P<what>[\w .'-]+?)"
_NAMED = re.compile(rf"\bmi (?P<kin>{_KIN}) se llama (?P<name>[\w .'-]+?){_END}")
_WORKS = re.compile(rf"\b{_WHO} trabaja en {_TARGET}{_END}")
_I_WORK = re.compile(rf"\btrabajo en {_TARGET}{_END}")
_USES_FILE = re.compile(
    rf"\b{_WHO} usa (?:el archivo |el documento )?(?P<file>[\w-]+\.\w{{2,5}})\b"
)
_LIVES = re.compile(rf"\b(?P<who>vivo|[\w.'-]+ vive) en (?P<where>[\w .'-]+?){_END}")
_LIKES = re.compile(r"\b(?P<neg>no )?me (?:gusta|gustan|encanta|encantan) (?P<what>.+?)(?:[,.;]|$)")
_HATES = re.compile(r"\b(?:odio|detesto) (?P<what>.+?)(?:[,.;]|$)")
_PET_KIN = frozenset({"perro", "perra", "gato", "gata"})


def remembered_fact(text: str) -> str | None:
    """El hecho de "recuerda que X", en segunda persona y con sus tildes."""
    raw = raw_capture(text, _REMEMBER, "fact")
    return sentence(to_second_person(raw)) if raw else None


class RuleExtractor:
    """Extraccion por reglas: frases tipicas del espanol hablado."""

    name = "reglas"

    def __init__(self, user_name: Callable[[], str]) -> None:
        """Args: user_name, el nombre del usuario para su nodo en el grafo."""
        self._user_name = user_name

    async def extract(self, episodes: Sequence[Episode]) -> Extraction:
        """Aplica las reglas a lo que dijo el usuario en cada episodio."""
        result = Extraction()
        for episode in episodes:
            if episode.user_text:
                result.extend(self.from_text(episode.user_text))
        return result

    def from_text(self, text: str) -> Extraction:
        """Las reglas sobre una sola frase del usuario."""
        out = Extraction()
        me = self._user_name()
        fact = remembered_fact(text)
        if fact:
            out.facts.append(FactCandidate(fact, 0.9))
        for match in _NAMED.finditer(fold(text)):
            name = _cut(text, match, "name")
            kin = match["kin"]
            if fact is None:
                out.facts.append(FactCandidate(f"Tu {kin} se llama {name}.", 0.9))
            kind = "cosa" if kin in _PET_KIN else "persona"
            out.entities.append(EntityCandidate(name, kind))
            # "Ana es hermana de Rhandy": se lee bien desde cualquiera de los dos.
            out.relations.append(RelationCandidate(name, f"es_{kin}_de", me, kind, "persona"))
        for match in _WORKS.finditer(fold(text)):
            who, what = _cut(text, match, "who"), _cut(text, match, "what")
            if who.lower() in {"yo", "que"}:
                continue
            kind = "proyecto" if match["kind"] else "organizacion"
            out.relations.append(RelationCandidate(who, "trabaja_en", what, "persona", kind))
        for match in _I_WORK.finditer(fold(text)):
            what = _cut(text, match, "what")
            kind = "proyecto" if match["kind"] else "organizacion"
            out.relations.append(RelationCandidate(me, "trabaja_en", what, "persona", kind))
            if fact is None:
                out.facts.append(FactCandidate(f"Trabajas en {what}.", 0.7))
        for match in _USES_FILE.finditer(fold(text)):
            who, file = _cut(text, match, "who"), _cut(text, match, "file")
            out.relations.append(RelationCandidate(who, "usa_archivo", file, "cosa", "archivo"))
        for match in _LIVES.finditer(fold(text)):
            where = _cut(text, match, "where")
            who = me if match["who"] == "vivo" else _cut(text, match, "who")[: -len(" vive")]
            out.relations.append(RelationCandidate(who, "vive_en", where, "persona", "lugar"))
            if fact is None and who == me:
                out.facts.append(FactCandidate(f"Vives en {where}.", 0.7))
        if fact is None:
            for match in _LIKES.finditer(fold(text)):
                what = _cut(text, match, "what")
                verb = "No te gusta" if match["neg"] else "Te gusta"
                out.facts.append(FactCandidate(f"{verb} {what}.", 0.6))
            for match in _HATES.finditer(fold(text)):
                out.facts.append(FactCandidate(f"Odias {_cut(text, match, 'what')}.", 0.6))
        return out


def _cut(text: str, match: re.Match[str], group: str) -> str:
    """El grupo recortado del texto original (los indices son los del plegado)."""
    return text[match.start(group) : match.end(group)].strip(" .,;:")


# ---------------------------------------------------------------------------
# Modelo local
# ---------------------------------------------------------------------------


class _Fact(BaseModel):
    hecho: str = Field(min_length=3)
    importancia: float = Field(ge=0.0, le=1.0)


class _Entity(BaseModel):
    nombre: str = Field(min_length=1)
    tipo: str


class _Relation(BaseModel):
    origen: str = Field(min_length=1)
    relacion: str = Field(min_length=1)
    destino: str = Field(min_length=1)


class _Output(BaseModel):
    hechos: list[_Fact] = Field(default_factory=list)
    entidades: list[_Entity] = Field(default_factory=list)
    relaciones: list[_Relation] = Field(default_factory=list)
    resumen: str = ""


OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "hechos": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"hecho": {"type": "string"}, "importancia": {"type": "number"}},
                "required": ["hecho", "importancia"],
            },
        },
        "entidades": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "nombre": {"type": "string"},
                    "tipo": {"type": "string", "enum": list(ENTITY_TYPES)},
                },
                "required": ["nombre", "tipo"],
            },
        },
        "relaciones": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "origen": {"type": "string"},
                    "relacion": {"type": "string"},
                    "destino": {"type": "string"},
                },
                "required": ["origen", "relacion", "destino"],
            },
        },
        "resumen": {"type": "string"},
    },
    "required": ["hechos", "entidades", "relaciones", "resumen"],
}
_TURN_CHARS = 400
_MAX_OUTPUT_TOKENS = 600
_TIMEOUT_S = 600.0
"""Trabajo por lotes en CPU: un tramo largo puede tardar minutos y esta bien."""


class OllamaExtractor:
    """El modelo local lee un tramo de conversacion y devuelve JSON."""

    name = "modelo local"

    def __init__(
        self,
        config: Callable[[], CognitionConfig],
        user_name: Callable[[], str],
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """Args: config, la seccion de cognicion (url y modelo de Ollama); user_name."""
        self._config = config
        self._user_name = user_name
        self._transport = transport

    async def extract(self, episodes: Sequence[Episode]) -> Extraction:
        """Una llamada a Ollama por tramo. Lanza `ExtractionError` si falla."""
        cfg = self._config()
        me = self._user_name()
        lines: list[str] = []
        for episode in episodes:
            if episode.user_text:
                lines.append(f"{me}: {clip(episode.user_text, _TURN_CHARS)}")
            if episode.answer:
                lines.append(f"Jarvis: {clip(episode.answer, _TURN_CHARS)}")
        if not lines:
            return Extraction()
        body = {
            "model": cfg.local_model,
            "messages": [
                {"role": "system", "content": load_prompt("memory_extract").format(user_name=me)},
                {"role": "user", "content": "\n".join(lines)},
            ],
            "stream": False,
            "think": False,
            "format": OUTPUT_SCHEMA,
            "keep_alive": int(cfg.local_keep_alive)
            if cfg.local_keep_alive.lstrip("-").isdigit()
            else cfg.local_keep_alive,
            "options": {
                "temperature": 0.1,
                "num_predict": _MAX_OUTPUT_TOKENS,
                "num_ctx": cfg.local_num_ctx,
            },
        }
        try:
            async with httpx.AsyncClient(
                base_url=cfg.ollama_url, transport=self._transport, timeout=_TIMEOUT_S
            ) as client:
                response = await client.post("/api/chat", json=body)
                response.raise_for_status()
                content = str(response.json()["message"]["content"])
        except (httpx.HTTPError, KeyError, ValueError) as error:
            raise ExtractionError(f"Ollama no extrajo: {type(error).__name__}: {error}") from error
        try:
            parsed = _Output.model_validate_json(content)
        except ValidationError as error:
            raise ExtractionError(f"El modelo devolvio JSON invalido: {error}") from error
        return _to_extraction(parsed)


_ABOUT_USER = re.compile(
    r"\b(?:tu|tus|te|ti|tienes|eres|estas|vives|trabajas|odias|prefieres|usas|sueles|"
    r"quieres|necesitas|llamas)\b"
)


def about_user(fact: str, people: set[str]) -> bool:
    """Si un hecho habla del usuario o de alguien de su mundo.

    DECISION: lo que el modelo local extrae se filtra con esto. Medido con
    qwen3:4b sobre la semana simulada: guardaba cultura general ("la capital
    de Mongolia es Ulan Bator"), respuestas de Jarvis ("Rust es un lenguaje
    solido") y hasta ordenes ("Pon un temporizador"). Un hecho util para
    recordar habla de "tu..." o nombra a alguien que el usuario conoce.
    """
    folded = fold(fact)
    if _ABOUT_USER.search(folded):
        return True
    return any(re.search(rf"\b{re.escape(name)}\b", folded) for name in people)


def _to_extraction(parsed: _Output) -> Extraction:
    types = {e.nombre.strip().lower(): e.tipo for e in parsed.entidades}
    people = {fold(e.nombre.strip()) for e in parsed.entidades if e.tipo == "persona"}
    return Extraction(
        facts=[
            FactCandidate(sentence(f.hecho), f.importancia)
            for f in parsed.hechos
            if about_user(f.hecho, people)
        ],
        entities=[
            EntityCandidate(e.nombre.strip(), e.tipo if e.tipo in ENTITY_TYPES else "cosa")
            for e in parsed.entidades
        ],
        relations=[
            RelationCandidate(
                r.origen.strip(),
                r.relacion,
                r.destino.strip(),
                types.get(r.origen.strip().lower(), "cosa"),
                types.get(r.destino.strip().lower(), "cosa"),
            )
            for r in parsed.relaciones
        ],
        summary=parsed.resumen.strip(),
    )
