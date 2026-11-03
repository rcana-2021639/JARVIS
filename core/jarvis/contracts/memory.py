"""Contrato de la unidad minima de memoria."""

from __future__ import annotations

import math
from datetime import datetime

from pydantic import Field, JsonValue

from jarvis.contracts.base import BaseContract, new_id, utc_now
from jarvis.contracts.enums import MemoryKind

__all__ = ["ConsolidationReport", "MemoryRecord"]

_HOURS_PER_DAY = 24.0


class MemoryRecord(BaseContract):
    """Una unidad de memoria, de cualquiera de las cinco capas.

    Las cinco capas comparten contrato a proposito: la recuperacion hibrida de
    la Fase 3 corre cuatro busquedas en paralelo sobre almacenes distintos
    (SQLite, LanceDB, el grafo) y las fusiona con Reciprocal Rank Fusion. Esa
    fusion solo es simple si todo lo que sale de los cuatro tiene la misma
    forma. `kind` dice de que capa vino; el resto es identico.
    """

    record_id: str = Field(
        default_factory=new_id,
        description="Identificador estable del registro.",
    )
    kind: MemoryKind = Field(
        description="Capa de memoria a la que pertenece.",
    )
    content: str = Field(
        min_length=1,
        description="El contenido en texto. Es lo que se indexa y lo que ve el modelo.",
    )
    embedding: list[float] | None = Field(
        default=None,
        description=(
            "Vector del contenido. Solo lo llevan los registros de la capa "
            "SEMANTICA; en las demas es None y no se calcula."
        ),
    )
    importance: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Cuanto pesa este recuerdo. 1.0 es permanente y no decae nunca.",
    )
    decay: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Cuanto se ha desvanecido por desuso. La consolidacion nocturna lo sube.",
    )
    timestamp: datetime = Field(
        default_factory=utc_now,
        description="Cuando ocurrio o se aprendio.",
    )
    last_accessed_at: datetime = Field(
        default_factory=utc_now,
        description="Ultima vez que se recupero. El decaimiento se calcula desde aqui.",
    )
    metadata: dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Datos propios de cada capa: entidades citadas, id del episodio, etc.",
    )

    @property
    def is_permanent(self) -> bool:
        """Si esta marcado como permanente y la consolidacion no debe degradarlo."""
        return self.importance >= 1.0

    @property
    def effective_weight(self) -> float:
        """Peso real del recuerdo hoy: importancia menos lo que se ha desvanecido.

        Es el numero con el que la recuperacion hibrida ordena candidatos. Lo
        permanente ignora el decaimiento por definicion.
        """
        if self.is_permanent:
            return 1.0
        return max(0.0, self.importance * (1.0 - self.decay))

    def decayed_at(self, moment: datetime, half_life_days: float) -> float:
        """Calcula que decaimiento tendria este recuerdo en un instante dado.

        Usa decaimiento exponencial con vida media configurable: pasada una vida
        media sin usarse, el recuerdo perdio la mitad de su peso. No muta nada
        -- la consolidacion nocturna es quien decide si persiste el resultado.
        """
        if self.is_permanent:
            return 0.0
        if half_life_days <= 0:
            msg = "half_life_days debe ser mayor que cero"
            raise ValueError(msg)
        elapsed_hours = (moment - self.last_accessed_at).total_seconds() / 3600.0
        if elapsed_hours <= 0:
            return self.decay
        elapsed_half_lives = elapsed_hours / (half_life_days * _HOURS_PER_DAY)
        return min(1.0, 1.0 - math.pow(0.5, elapsed_half_lives))


class ConsolidationReport(BaseContract):
    """Lo que aprendio una consolidacion nocturna. Viaja en `memory.consolidation.finished`.

    Se publica al terminar y otra vez al arrancar si nadie lo ha oido todavia:
    "Jarvis reporta al arrancar que aprendio anoche".
    """

    started_at: datetime = Field(description="Cuando empezo.")
    finished_at: datetime = Field(description="Cuando termino.")
    extractor: str = Field(description="Quien extrajo: 'modelo local' o 'reglas'.")
    episodes: int = Field(default=0, ge=0, description="Episodios leidos.")
    conversations: int = Field(default=0, ge=0, description="Conversaciones en que se agruparon.")
    facts_new: int = Field(default=0, ge=0)
    facts_updated: int = Field(default=0, ge=0, description="Hechos que reemplazaron a otros.")
    facts_reinforced: int = Field(default=0, ge=0, description="Hechos que ya se sabian.")
    entities_new: int = Field(default=0, ge=0)
    relations_new: int = Field(default=0, ge=0)
    summaries: int = Field(default=0, ge=0, description="Conversaciones largas resumidas.")
    decayed: int = Field(default=0, ge=0, description="Hechos cuyo peso bajo por desuso.")
    routines_proposed: int = Field(default=0, ge=0)
    compressed: int = Field(default=0, ge=0, description="Episodios viejos comprimidos.")
    learned: list[str] = Field(
        default_factory=list, description="Los hechos nuevos, en texto, para decirlos."
    )
    errors: list[str] = Field(default_factory=list, description="Lo que fallo, sin detenerse.")

    def spoken(self) -> str:
        """El reporte en una o dos frases."""
        if not (
            self.facts_new or self.facts_updated or self.entities_new or self.routines_proposed
        ):
            return "Anoche repasé la memoria y no aprendí nada nuevo."
        parts: list[str] = []
        if self.facts_new or self.facts_updated:
            total = self.facts_new + self.facts_updated
            parts.append("una cosa nueva" if total == 1 else f"{total} cosas nuevas")
        if self.entities_new:
            parts.append(
                "una persona o cosa más" if self.entities_new == 1
                else f"{self.entities_new} personas o cosas más"
            )  # fmt: skip
        if self.routines_proposed:
            parts.append(
                "propuse una rutina" if self.routines_proposed == 1
                else f"propuse {self.routines_proposed} rutinas"
            )  # fmt: skip
        joined = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " y " + parts[-1]
        text = f"Anoche aprendí {joined}."
        if self.learned:
            text += f" Por ejemplo: {self.learned[0].rstrip('.')}."
        return text
