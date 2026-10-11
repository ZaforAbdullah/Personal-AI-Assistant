# Tools available to the assistant

## calculator

The calculator tool evaluates arithmetic expressions exactly. It supports addition, subtraction, multiplication, division, integer division, modulo, powers, parentheses and the functions sqrt, abs, round, min, max, log, floor and ceil, plus the constants pi and e. It is implemented by walking a parsed syntax tree with a whitelist of allowed operations, so arbitrary code can never run. Exponents and result sizes are capped to keep responses fast.

## knowledge_search

The knowledge_search tool answers questions about the assistant and about how it is built. It turns the question into an embedding vector, compares it with the stored document chunks using cosine similarity in pgvector, and returns the most relevant passages with a score. The assistant then writes its answer from those passages and says so when nothing relevant was found.

## weather

The weather tool reports current conditions for a city or place: temperature, how it feels, wind and a plain-language description such as "light rain". It looks the place up with the Open-Meteo geocoding service and then fetches the current conditions; no API key is needed.

## gif_search

The gif_search tool finds a family-friendly GIF for a short description using GIPHY and the chat interface shows it inline. It is only offered to the assistant when a GIPHY API key is configured.

## Adding a tool

A new tool declares its arguments as a pydantic model, which is also converted into the JSON schema the model sees, and implements an async `run` method. Registering it with the tool registry is enough for the model to start using it.
