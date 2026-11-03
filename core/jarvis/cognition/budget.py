"""El gasto del dia en la nube y su tope.

DECISION: el acumulado se guarda en disco (`spend_ledger`). Si viviera solo en
RAM, reiniciar el nucleo pondria el contador a cero y el tope diario dejaria de
significar nada: tres reinicios, tres presupuestos. No es memoria de la
conversacion (eso es la Fase 3), es contabilidad.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING

from jarvis.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from jarvis.config import ModelPrice

__all__ = ["SpendLedger", "SpendResult", "cost_usd"]

_log = get_logger(__name__)
_CACHE_WRITE_FACTOR = 1.25
_CACHE_READ_FACTOR = 0.10
_MTOK = 1_000_000


def cost_usd(
    prices: Mapping[str, ModelPrice],
    model: str,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_write_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> float:
    """Costo de una llamada segun la tabla de precios.

    Un modelo sin precio conocido lanza KeyError: preferible a cobrar cero y
    que el tope diario no proteja nada.
    """
    price = prices[model]
    billed_input = (
        input_tokens
        + cache_write_tokens * _CACHE_WRITE_FACTOR
        + cache_read_tokens * _CACHE_READ_FACTOR
    )
    return (billed_input * price.input_per_mtok + output_tokens * price.output_per_mtok) / _MTOK


@dataclass(frozen=True, slots=True)
class SpendResult:
    """Como quedo el dia tras registrar un gasto."""

    spent_today: float
    budget: float
    exhausted: bool
    just_exhausted: bool
    """True solo en la llamada que cruzo el tope: es cuando se avisa por voz."""


class SpendLedger:
    """Acumula el gasto diario y dice si todavia se puede gastar."""

    def __init__(
        self,
        path: Path | None,
        budget: Callable[[], float],
        today: Callable[[], date] = date.today,
    ) -> None:
        """Args: path, archivo del acumulado (None, solo RAM); budget, tope vigente."""
        self._path = path
        self._budget = budget
        self._today = today
        self._day = today()
        self._spent = 0.0
        self._load()

    @property
    def budget(self) -> float:
        """El tope vigente. Cero es sin limite."""
        return self._budget()

    @property
    def spent_today(self) -> float:
        """Lo gastado hoy. Se reinicia solo al cambiar de dia."""
        self._roll_day()
        return self._spent

    def can_spend(self) -> bool:
        """Si queda presupuesto hoy."""
        budget = self._budget()
        return budget <= 0 or self.spent_today < budget

    def record(self, amount: float) -> SpendResult:
        """Suma un gasto y lo persiste."""
        before = self.can_spend()
        self._spent += max(0.0, amount)
        self._save()
        exhausted = not self.can_spend()
        return SpendResult(
            spent_today=self._spent,
            budget=self._budget(),
            exhausted=exhausted,
            just_exhausted=before and exhausted,
        )

    def _roll_day(self) -> None:
        today = self._today()
        if today != self._day:
            self._day = today
            self._spent = 0.0

    def _load(self) -> None:
        if self._path is None or not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            if data.get("date") == self._day.isoformat():
                self._spent = float(data.get("spent_usd", 0.0))
        except (OSError, ValueError, TypeError) as error:
            # Un archivo ilegible no puede regalar presupuesto: se avisa y se
            # parte de cero, que es lo unico que se puede hacer.
            _log.warning("budget.ledger_unreadable", path=str(self._path), error=str(error))

    def _save(self) -> None:
        if self._path is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"date": self._day.isoformat(), "spent_usd": round(self._spent, 6)}
            self._path.write_text(json.dumps(payload), encoding="utf-8")
        except OSError as error:
            _log.error("budget.ledger_unwritable", path=str(self._path), error=str(error))
