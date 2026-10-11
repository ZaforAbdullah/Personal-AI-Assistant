import pytest

from app.agent.prompts import language_name, system_prompt


@pytest.mark.parametrize(
    ("locale", "expected"),
    [
        ("en-US", "English"),
        ("pt-BR", "Portuguese"),
        ("es", "Spanish"),
        ("es_419", "Spanish"),
        ("zh-Hant-TW", "Chinese"),
        ("  DE-de ", "German"),
        ("tlh", "English"),  # unknown -> default
        ("", "English"),
        (None, "English"),
    ],
)
def test_language_name(locale: str | None, expected: str) -> None:
    assert language_name(locale) == expected


def test_system_prompt_pins_reply_language() -> None:
    assert system_prompt("pt-BR").endswith(
        "Always reply in Portuguese, whatever language or notation the user writes in."
    )
    assert "Always reply in English" in system_prompt(None)
