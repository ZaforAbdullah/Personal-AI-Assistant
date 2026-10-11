# How this chat app is built

## Overview

Personal AI Assistant is a small chat service with an assistant that can use tools. The backend is a Python FastAPI service. The frontend is a Next.js 16 application written in TypeScript. Conversations and the knowledge base are stored in PostgreSQL with the pgvector extension.

## Streaming

Replies stream to the browser over Server-Sent Events (SSE). The backend emits typed events: `conversation` (which conversation this reply belongs to), `text_delta` (a piece of the reply text), `tool_start`, `tool_result` and `tool_error` (progress of a tool call), `done` and `error`. While a slow tool or a slow model keeps the connection quiet, the server sends a comment line as a heartbeat so proxies and browsers do not think the stream has stalled.

## The agent loop

For each user message the backend streams the model's reply. If the model asks for a tool, the backend runs the tool with a timeout, sends the result back to the model and streams again, up to a fixed number of rounds. On the final round no tools are offered, so the model must answer in prose. Tool failures are reported to the user and to the model, which then explains the problem instead of the stream breaking.

## Providers

The language model is configurable with an environment variable. `LLM_PROVIDER=gemini` uses Google's Gemini (gemini-3.1-flash-lite) through its OpenAI-compatible endpoint. `LLM_PROVIDER=lmstudio` uses a model served locally by LM Studio, such as Gemma 4, with no API key and no data leaving the machine. Both go through the same client code because both speak the OpenAI chat completions protocol.

## Storage

Every user message, assistant reply, tool call and tool result is stored in Postgres so a conversation can be reloaded. Partial replies are also stored when a stream is interrupted. The knowledge base is a set of markdown documents that are chunked, embedded and stored in a pgvector column; the knowledge_search tool embeds the question and runs a cosine similarity search. The database can be a local Docker Postgres or a Supabase project, selected with `DB_PROVIDER`.

## Conversations

The sidebar lists past conversations, most recently active first. A conversation is named after its first message unless the user renames it, and it can be deleted. All of this is served by a small REST API next to the streaming chat endpoint.

## Cache

An in-process cache keeps each active conversation's recent history and the embeddings of recent questions, so a turn does not re-read the database and a repeated question does not re-embed. The health endpoint reports the cache hit rate. With several backend replicas the same cache interface would sit in front of Redis.
