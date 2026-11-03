"""Las dos herramientas de demostracion con las que se prueba el bucle.

El documento pide herramientas simuladas para ejercitar el ciclo antes de que
existan las reales (Fase 6). Estas son de juguete pero honestas: la
calculadora calcula de verdad y el reloj da la hora de verdad. Un asistente que
inventa resultados, aunque sea en una demo, entrena el habito equivocado.
"""

from __future__ import annotations

import ast
import operator
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from jarvis.cognition.tool_loop import ToolSpec

if TYPE_CHECKING:
    from collections.abc import Callable

    from pydantic import JsonValue

__all__ = ["demo_tools", "safe_arithmetic"]

_BINARY: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}
_MAX_EXPONENT = 100
_MAX_LENGTH = 200


def safe_arithmetic(expression: str) -> float:
    """Evalua aritmetica pura sin `eval`: solo numeros y + - * / % **.

    Lanza ValueError ante cualquier otra cosa (nombres, llamadas, atributos).
    """
    if len(expression) > _MAX_LENGTH:
        msg = "Expresion demasiado larga"
        raise ValueError(msg)

    def visit(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            right = visit(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
                msg = "Exponente demasiado grande"
                raise ValueError(msg)
            return _BINARY[type(node.op)](visit(node.left), right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](visit(node.operand))
        msg = f"Operacion no permitida: {type(node).__name__}"
        raise ValueError(msg)

    try:
        tree = ast.parse(expression.replace("^", "**").replace(",", "."), mode="eval")
    except SyntaxError as error:
        msg = f"No es una expresion aritmetica: {expression}"
        raise ValueError(msg) from error
    return visit(tree)


async def _calculator(arguments: dict[str, Any]) -> JsonValue:
    expression = str(arguments["expression"])
    value = safe_arithmetic(expression)
    return {"expression": expression, "result": round(value, 10)}


async def _clock(arguments: dict[str, Any]) -> JsonValue:
    offset = arguments.get("utc_offset_hours")
    if offset is None:
        now = datetime.now().astimezone()
    else:
        now = datetime.now(UTC) + timedelta(hours=float(offset))
    return {"iso": now.isoformat(timespec="minutes"), "weekday": now.strftime("%A")}


def demo_tools() -> list[ToolSpec]:
    """Las herramientas de demostracion, listas para registrar."""
    return [
        ToolSpec(
            server="demo",
            name="calculator",
            description="Calcula una expresion aritmetica exacta. Usala para cualquier cuenta.",
            input_schema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Solo numeros y + - * / % ** y parentesis. Ej: (25*4)+3",
                    }
                },
                "required": ["expression"],
            },
            handler=_calculator,
        ),
        ToolSpec(
            server="demo",
            name="clock",
            description="Fecha y hora actuales, locales o con un desfase UTC dado.",
            input_schema={
                "type": "object",
                "properties": {
                    "utc_offset_hours": {
                        "type": "number",
                        "description": "Desfase respecto de UTC. Omitelo para la hora local.",
                    }
                },
            },
            handler=_clock,
        ),
    ]
