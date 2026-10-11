import pytest
from langchain_core.utils.function_calling import convert_to_openai_tool

from app.tools.base import ToolError
from app.tools.calculator import CalculatorTool, evaluate


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("2 + 2", 4),
        ("17 * 23 + 5", 396),
        ("(1 + 2) * 3", 9),
        ("10 / 4", 2.5),
        ("10 // 4", 2),
        ("10 % 4", 2),
        ("2 ** 10", 1024),
        ("-5 + 3", -2),
        ("sqrt(16)", 4),
        ("round(pi, 2)", 3.14),
        ("max(1, 7, 3)", 7),
        ("abs(-2.5)", 2.5),
        ("1e3 + 1", 1001),
        # calculator-style notation
        ("2^10", 1024),
        ("2^3^2", 512),  # right-associative like ** (and like a calculator)
        ("5!", 120),
        ("7!/(5!*2!)", 21),
        ("(3+1)!", 24),
        ("3 × 4 ÷ 2 − 1", 5),
        ("0!", 1),
        ("log10(1000)", 3),
    ],
)
def test_evaluate(expr: str, expected: float) -> None:
    assert evaluate(expr) == expected


def test_the_kind_of_expression_people_actually_type() -> None:
    expr = "((17^3 - 13^3) * (29^2 - 11^2)) / (23 - 7)^2 + sqrt(1444 + 225) - 7!/(5!*2!)"
    assert evaluate(expr) == pytest.approx(2716 * 720 / 256 + 1669**0.5 - 21)


@pytest.mark.parametrize(
    ("expr", "message"),
    [
        ("2.5!", "non-negative integer"),
        ("(-3)!", "non-negative integer"),
        ("501!", "too large"),
        ("2+3)!", "unbalanced"),
        ("(2+3!", "could not parse"),
    ],
)
def test_factorial_edge_cases(expr: str, message: str) -> None:
    with pytest.raises(ToolError, match=message):
        evaluate(expr)


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('ls')",
        "open('/etc/passwd')",
        "x + 1",
        "2 ** 100000",
        "1 / 0",
        "1 +",
        "[1, 2]",
        "'a' * 3",
        "sqrt(-1)",
        "print(1)",
        "round(1.5, ndigits=0)",
    ],
)
def test_rejects_unsafe_or_invalid(expr: str) -> None:
    with pytest.raises(ToolError):
        evaluate(expr)


async def test_tool_returns_structured_result() -> None:
    result = await CalculatorTool().ainvoke({"expression": "6 * 7"})
    assert result == {"expression": "6 * 7", "result": 42}


async def test_tool_error_propagates_through_langchain() -> None:
    with pytest.raises(ToolError, match="division by zero"):
        await CalculatorTool().ainvoke({"expression": "1 / 0"})


def test_schema_is_openai_function_shape() -> None:
    schema = convert_to_openai_tool(CalculatorTool())
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "calculator"
    assert schema["function"]["parameters"]["required"] == ["expression"]
    assert "expression" in schema["function"]["parameters"]["properties"]
