"""Calculator tool: arithmetic through a whitelisted AST walk, never `eval`.

A normalisation pass first accepts what people actually type: `^` for power, `n!` for
factorial, and `× ÷ −`.
"""

from __future__ import annotations

import ast
import math
import operator
import re
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from app.tools.base import AsyncTool, ToolError

_BINARY: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _factorial(n: Any) -> int:
    if isinstance(n, float) and n.is_integer():
        n = int(n)
    if not isinstance(n, int) or n < 0:
        raise ToolError("factorial needs a non-negative integer")
    if n > _MAX_FACTORIAL:
        raise ToolError(f"factorial argument too large (max {_MAX_FACTORIAL})")
    return math.factorial(n)


_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "log": math.log,
    "log10": math.log10,
    "floor": math.floor,
    "ceil": math.ceil,
    "factorial": _factorial,
}
_CONSTANTS: dict[str, float] = {"pi": math.pi, "e": math.e}

_MAX_EXPONENT = 1000
_MAX_FACTORIAL = 500
_MAX_ABS = 1e300


class CalculatorArgs(BaseModel):
    expression: str = Field(
        description="Arithmetic expression, e.g. '17 * 23 + 5', '2^10', '7!/(5!*2!)' or "
        "'sqrt(2) * pi'. Supports + - * / // % ^ ** ! parentheses and "
        "sqrt/abs/round/min/max/log/log10/floor/ceil/factorial.",
        min_length=1,
        max_length=500,
    )


class CalculatorTool(AsyncTool):
    name: str = "calculator"
    description: str = (
        "Evaluate an arithmetic expression exactly. Use this for any calculation instead of "
        "doing the maths yourself."
    )
    args_schema: type[BaseModel] = CalculatorArgs

    async def _arun(self, expression: str) -> dict[str, Any]:
        return {"expression": expression, "result": evaluate(expression)}


_UNICODE = str.maketrans({"×": "*", "÷": "/", "−": "-", "–": "-", "，": ","})
_NUMBER_BANG = re.compile(r"(\d+(?:\.\d+)?|\b[a-z]+)!")


def normalise(expression: str) -> str:
    """Turn calculator-style notation into Python syntax."""
    text = expression.strip().translate(_UNICODE)
    text = text.replace("^", "**")
    text = _NUMBER_BANG.sub(r"factorial(\1)", text)  # 7! -> factorial(7)
    return _wrap_paren_factorials(text)  # (n+1)! -> factorial((n+1))


def _wrap_paren_factorials(text: str) -> str:
    while (end := text.find(")!")) != -1:
        depth, start = 0, end
        while start >= 0:
            if text[start] == ")":
                depth += 1
            elif text[start] == "(":
                depth -= 1
                if depth == 0:
                    break
            start -= 1
        if start < 0:
            raise ToolError("unbalanced parentheses before '!'")
        text = f"{text[:start]}factorial{text[start : end + 1]}{text[end + 2 :]}"
    return text


def evaluate(expression: str) -> float | int:
    try:
        tree = ast.parse(normalise(expression), mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"could not parse expression: {exc.msg}") from exc
    try:
        value = _eval_node(tree.body)
    except ZeroDivisionError as exc:
        raise ToolError("division by zero") from exc
    except (ValueError, OverflowError) as exc:
        raise ToolError(f"math error: {exc}") from exc
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        raise ToolError("result is not a finite number")
    if isinstance(value, float) and value.is_integer() and abs(value) < 1e15:
        return int(value)
    return value


def _eval_node(node: ast.expr) -> float | int:
    match node:
        case ast.Constant(value=v) if isinstance(v, int | float) and not isinstance(v, bool):
            return v
        case ast.Name(id=name) if name in _CONSTANTS:
            return _CONSTANTS[name]
        case ast.UnaryOp(op=op, operand=operand) if type(op) in _UNARY:
            return _UNARY[type(op)](_eval_node(operand))
        case ast.BinOp(left=left, op=op, right=right) if type(op) in _BINARY:
            lhs, rhs = _eval_node(left), _eval_node(right)
            if isinstance(op, ast.Pow) and abs(rhs) > _MAX_EXPONENT:
                raise ToolError(f"exponent too large (max {_MAX_EXPONENT})")
            out = _BINARY[type(op)](lhs, rhs)
            if isinstance(out, int | float) and abs(out) > _MAX_ABS:
                raise ToolError("result too large")
            return out
        case ast.Call(func=ast.Name(id=fname), args=call_args, keywords=[]) if fname in _FUNCTIONS:
            return _FUNCTIONS[fname](*(_eval_node(a) for a in call_args))
        case _:
            raise ToolError(f"unsupported syntax: {type(node).__name__}")
