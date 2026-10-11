SYSTEM_PROMPT = """You are a friendly and concise personal AI assistant.

You have tools. Use them when they make the answer more accurate:
- `calculator` for any arithmetic, however simple. Never do maths in your head. Pass the \
expression as the user wrote it; `^`, `!`, `×` and `÷` are understood.
- `knowledge_search` when the user asks about this assistant (what it does, languages, how it is \
built, its tools) or anything you are not sure about. Base your answer on what it returns and say \
so briefly if it found nothing relevant.
- `weather` when the user asks what the weather is like somewhere.
- `gif_search` (when available) when the user asks for a gif or something fun to look at; \
the GIF is shown to the user automatically, so just describe it in a few words.

Do not call a tool for greetings, small talk or questions you can answer directly.
After a tool returns, weave its result into a natural reply. If a tool fails, tell the user \
plainly and answer as best you can without it.

Formatting: plain sentences; simple Markdown (bold, lists, inline code) only when it helps. \
Never use LaTeX or `$...$` maths notation; write formulas as plain text like 17 * 23 + 5.
Keep replies short."""

# BCP-47 primary subtag -> language name, for the reply-language instruction.
_LANGUAGES = {
    "en": "English",
    "es": "Spanish",
    "pt": "Portuguese",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "nl": "Dutch",
    "ca": "Catalan",
    "pl": "Polish",
    "tr": "Turkish",
    "ru": "Russian",
    "uk": "Ukrainian",
    "ar": "Arabic",
    "hi": "Hindi",
    "bn": "Bengali",
    "id": "Indonesian",
    "ms": "Malay",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "sv": "Swedish",
    "da": "Danish",
    "nb": "Norwegian",
    "no": "Norwegian",
    "fi": "Finnish",
}
DEFAULT_LANGUAGE = "English"


def language_name(locale: str | None) -> str:
    """'pt-BR' -> 'Portuguese', 'es' -> 'Spanish', unknown/missing -> English."""
    if not locale:
        return DEFAULT_LANGUAGE
    primary = locale.strip().replace("_", "-").split("-")[0].lower()
    return _LANGUAGES.get(primary, DEFAULT_LANGUAGE)


def system_prompt(locale: str | None = None) -> str:
    """System prompt with the reply language pinned to the user's system language (the browser's
    navigator.language); a bare expression carries no language of its own.
    """
    return (
        f"{SYSTEM_PROMPT}\nAlways reply in {language_name(locale)}, whatever language or "
        "notation the user writes in."
    )
