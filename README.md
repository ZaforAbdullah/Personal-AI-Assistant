# Personal AI Assistant

A streaming chat assistant that can use tools. You send a message; the reply streams back token by token, and when a question needs a calculation, a lookup, the weather or a GIF, the assistant calls the tool and works the result into its answer, with a card in the UI while the tool runs.

**Live demo:** [your-app.vercel.app](https://your-app.vercel.app) (placeholder, replace with the deployed frontend URL)

- **Backend:** FastAPI and a LangGraph agent. The chat model is Gemini (`gemini-3.1-flash-lite`) or any model served by LM Studio, chosen with one environment variable.
- **Frontend:** Next.js 16, React 19, TypeScript, Tailwind. Sidebar of past conversations, rename and delete, streaming Markdown, tool cards, a Stop button, light and dark themes.
- **Storage:** PostgreSQL with pgvector, either local Docker or Supabase. Holds conversations, every tool call and result, and the embedded knowledge base.
- **Tools:** calculator, knowledge search (vector lookup), weather (Open-Meteo, no key), GIF search (GIPHY, only offered when a key is set).
- **Tests:** run offline against a scripted model and in-memory storage. No keys, no network.

## Run it

You need Docker with Compose v2 and one model: a free [Gemini API key](https://aistudio.google.com/apikey), or [LM Studio](https://lmstudio.ai) with its server on port 1234 and a chat model and an embedding model loaded (defaults: `google/gemma-4-e4b` and `text-embedding-nomic-embed-text-v1.5`).

```bash
cp .env.example .env     # for Gemini set LLM_PROVIDER=gemini and GEMINI_API_KEY; skip for LM Studio
docker compose up --build
```

Open http://localhost:3000 and try:

| Send | What happens |
|---|---|
| `what is 2^10 + 7!/(5!*2!)?` | a calculator card appears, then the answer streams |
| `which languages does the assistant reply in?` | a knowledge search card, and the answer cites what it found |
| `what's the weather in Madrid?` | a live weather lookup |
| `1/0` | the tool error shows on the card and the assistant explains it |
| `hi!` | a plain streamed reply, no tool |

Press **Stop** mid-reply and reload: the partial reply is kept and marked as interrupted. With a `GIPHY_API_KEY` set, `show me a gif of a happy cat` renders a GIF inline.

If port 3000, 8000 or 5432 is taken, use for example `FRONTEND_PORT=3001 docker compose up --build` (`BACKEND_PORT` and `DB_PORT` work the same way; if you change the backend port, also set `NEXT_PUBLIC_API_URL` in `.env` before building). Without a `.env` the stack starts with the LM Studio defaults. If the model server is unreachable the app still starts, and chat reports the model as unavailable.

### Without Docker for the app

```bash
docker compose up -d db                      # only Postgres in Docker

cd backend
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/uvicorn app.main:app --reload --port 8000

cd frontend                                  # Node 20+ and pnpm (`corepack enable`)
pnpm install
pnpm dev                                     # http://localhost:3000
```

The backend reads `.env` from the repository root. The frontend reads `NEXT_PUBLIC_API_URL` (default `http://localhost:8000`).

## Tests

```bash
cd backend && .venv/bin/pytest         # 147 tests, about 3 s
cd frontend && pnpm test               # 17 tests
cd frontend && pnpm lint && pnpm typecheck
```

The backend suite is offline except for the Postgres integration tests, which run when a database is reachable on `localhost:5432` (`docker compose up -d db`) and are skipped otherwise. They use their own `assistant_test` database, created and truncated automatically, so application data is never touched. Point them elsewhere with `TEST_DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/dbname`.

A soak script drives the real app for thousands of rounds (tool turns, follow-ups, rename and delete, streams the client abandons) and exits non-zero if allocated memory, asyncio tasks, live generators or cache size keep growing. `--simulate-leak` checks that it can fail.

```bash
cd backend && .venv/bin/python -m scripts.soak --rounds 3000    # about 45 s, offline
```

CI (`.github/workflows/ci.yml`) runs the backend suite against a pgvector service, the soak script, the frontend lint, typecheck, tests and build, and builds both Docker images.

## Architecture

```mermaid
flowchart LR
    subgraph Browser
        UI[Next.js chat UI<br/>reducer + SSE reader]
    end
    subgraph Backend["FastAPI backend"]
        API[/POST /api/chat<br/>SSE/]
        SVC[ChatService<br/>history · persistence]
        G{{LangGraph<br/>agent ⇄ tools}}
        CACHE[(in-process cache<br/>history · embeddings)]
        T1[[calculator]]
        T2[[knowledge_search]]
        T3[[weather]]
        T4[[gif_search]]
    end
    LLM[(Gemini · LM Studio)]
    PG[(Postgres + pgvector<br/>conversations · messages · chunks)]
    EXT[(Open-Meteo · GIPHY)]

    UI -- "message" --> API --> SVC --> G
    G -- "stream tokens" --> LLM
    G --> T1
    G --> T2 -- "cosine search" --> PG
    T2 -- "embed query" --> LLM
    G --> T3 & T4 -- "https, 5s timeout" --> EXT
    SVC <--> CACHE <--> PG
    G -. "typed events" .-> API -. "text_delta · tool_start · tool_result · done" .-> UI
```

One turn: the user message is stored, recent history is loaded (from the cache after the first turn), and the graph streams the model with tools bound. If the model asks for a tool, the tool runs under a timeout, its result goes back to the model, and the model streams again, at most `MAX_TOOL_ROUNDS` times. Every step is a typed event on one SSE stream, which the browser's reducer turns into bubbles and tool cards. Whatever the loop produced, partial output included, is written back in one batch.

Three boundaries keep it testable:

- The graph depends only on `BaseChatModel` and `BaseTool`. Switching providers is a change in `llm/factory.py`.
- The service depends only on repository protocols. Postgres, Supabase, the in-memory test double and the cache wrapper are interchangeable.
- The API depends only on the event union, which the frontend mirrors in TypeScript.

### Code map

```
backend/app/
  agent/   events.py (wire contract) · graph.py (LangGraph agent ⇄ tools) · service.py (persistence around it)
  llm/     factory.py (Gemini and LM Studio clients) · embeddings.py · fake.py (scripted model for tests)
  tools/   calculator.py · knowledge_search.py · weather.py · gif.py · base.py · http.py
  db/      models.py · postgres.py · repositories.py (protocols and in-memory versions) · cached.py · seed.py
  knowledge/   the markdown documents behind knowledge search
  api/     chat.py · health.py          sse.py · cache.py · config.py · main.py
backend/tests/      offline; one Postgres integration file
frontend/src/
  lib/         sse.ts (stream reader and API calls) · chatReducer.ts (UI state) · types.ts
  components/  Chat · Sidebar · Composer · MessageBubble · ToolCallCard
```

If you read four files, read `agent/graph.py`, `agent/service.py`, `sse.py` and `tests/test_graph.py`.

## Configuration

Every variable is documented in [.env.example](.env.example). The ones that matter:

| Variable | Default | Notes |
|---|---|---|
| `LLM_PROVIDER` | `lmstudio` | `lmstudio` or `gemini`. Picks base URL, key and model. |
| `GEMINI_API_KEY`, `GEMINI_MODEL` | | Gemini uses its native client. Default model `gemini-3.1-flash-lite`. |
| `LMSTUDIO_BASE_URL`, `LMSTUDIO_MODEL`, `LMSTUDIO_EMBEDDING_MODEL` | | Inside Docker the host is `host.docker.internal`. |
| `LLM_TEMPERATURE`, `LLM_MAX_TOKENS` | `0.2`, `1024` | Sampling temperature and reply cap, both providers. |
| `EMBEDDING_DIM` | `768` | Preferred vector width. Gemini is asked for it; a provider that returns another width is accepted, since every row is embedded together. |
| `DB_PROVIDER` | `postgres` | `supabase` uses `SUPABASE_DB_URL` with TLS. Connection strings may start with `postgres://`, `postgresql://` or a driver prefix; they are converted for asyncpg. Enable the `vector` extension in the Supabase dashboard if the app cannot. |
| `GIPHY_API_KEY` | | Free key from developers.giphy.com. Without it `gif_search` is not offered. |
| `MAX_TOOL_ROUNDS`, `TOOL_TIMEOUT_S`, `HISTORY_LIMIT` | `4`, `15`, `30` | Agent loop limits. |
| `FRONTEND_PORT` | `3000` | CORS allows browser calls from any host on this port, and no other port. Under compose it is also the published port. |
| `CORS_ORIGINS` | | Extra exact origins allowed to call the API, comma separated, for a deployed frontend (for example `https://my-app.up.railway.app`). |
| `CACHE_ENABLED`, `CACHE_TTL_S`, `CACHE_MAX_ENTRIES` | `true`, `300`, `2000` | In-process cache for history, conversation lists and query embeddings. Hit and miss counts appear in `/health`. |

Changing the embedding model re-embeds the knowledge base on the next start.

## API

| Method | Path | |
|---|---|---|
| `POST` | `/api/chat` | Body `{"message": "...", "conversation_id": "<uuid>?", "locale": "pt-BR"?}`. `locale` is the user's system language (the UI sends `navigator.language`) and the reply is written in it; English when absent. Responds with `text/event-stream`. |
| `GET` | `/api/conversations?limit=50` | Conversation list: title, last activity, message count. |
| `GET` | `/api/conversations/{id}/messages` | Full stored history (user, assistant and tool rows). |
| `PATCH` | `/api/conversations/{id}` | Body `{"title": "..."}` (1 to 80 characters). The title otherwise defaults to the first message. |
| `DELETE` | `/api/conversations/{id}` | Removes a conversation and its messages. |
| `GET` | `/health` | Provider, model, database, tools and cache statistics. |
| `GET` | `/` | Points at `/docs`, FastAPI's interactive Swagger UI. |

SSE events arrive as `conversation`, then any mix of `text_delta`, `tool_start` followed by `tool_result` or `tool_error`, then `done` (or `error`). A `: ping` comment is sent after 15 s of silence.

```bash
curl -N localhost:8000/api/chat -H 'content-type: application/json' -d '{"message":"what is 2+2?"}'
```

## Design

**The agent is a small graph.** `agent/graph.py` is a LangGraph `StateGraph` with two nodes and one decision. `agent` streams the model, with tools bound while the round budget lasts and none on the last round, so the model must answer in prose. If the reply contains tool calls, `tools` runs each one under a timeout and reports progress on LangGraph's custom stream channel, then control returns to `agent`. State is the message list and a round counter. The nodes are written by hand rather than with the prebuilt `create_react_agent` and `ToolNode`, because per-tool timeouts, failure reporting and when tools stop being offered are exactly what the prebuilt versions hide. There is no HTTP or database code in the graph, so it is tested with a scripted chat model (`llm/fake.py`) that streams tool-call fragments the way a real provider does. Persistence lives in `ChatService`, wrapped around the graph.

**Two providers, one interface.** The graph only sees `BaseChatModel`. LM Studio uses `ChatOpenAI` against its OpenAI-compatible server. Gemini uses the native `ChatGoogleGenerativeAI`: Gemini 3 attaches a thought signature to every function call and rejects the next request unless it is echoed back, and the OpenAI-compatible endpoint carries it in a vendor field that generic clients drop, so tool calls fail on the second turn. Embeddings use the OpenAI-compatible endpoint for both providers. That endpoint returns `index: null` on Gemini, so response order is kept rather than sorted, and a rejected `dimensions` parameter is retried without it.

**Own message table instead of a LangGraph checkpointer.** The UI needs a queryable history per conversation, with every tool call and result as a row and partial replies marked, and a checkpoint blob is the wrong shape for that. A checkpointer could be added alongside the table for resumable runs.

**Typed events over SSE.** The server emits a small union: `conversation`, `text_delta`, `tool_start`, `tool_result`, `tool_error`, `done`, `error`. SSE over a POST `fetch` fits a one-directional stream and passes through any HTTP infrastructure; WebSockets would only pay off with client-to-server traffic mid-stream. After 15 s of silence the server sends a comment line so a slow tool or slow first token is not mistaken for a dead connection. The producer runs in its own task behind a bounded queue, so a heartbeat never cancels it and a slow client applies backpressure instead of letting the buffer grow.

**Failures are events.** A tool can fail four ways, and none of them breaks the stream:

- An expected failure (`ToolError`, such as division by zero) is shown as is and passed back to the model.
- Invalid or unparsable arguments are reported with the offending field.
- A timeout is enforced with `asyncio.wait_for`.
- An unexpected exception is logged and reported generically, so internals do not leak.

In every case the model receives a tool message describing the failure and continues in prose. If the model itself fails mid-reply, the partial text is stored with status `error`. If the client leaves (Stop, closed tab) or the server shuts down, the partial reply is stored as `interrupted`. When history is rebuilt for the next turn, tool calls with no stored result and results with no call are dropped, so the transcript is always valid for the model API.

**Tools.** `calculator` walks a parsed syntax tree against a whitelist (never `eval`) with capped exponents, factorials and magnitudes, and first accepts the notation people type: `2^10`, `7!`, `×`, `÷`. `knowledge_search` embeds the question and runs a cosine search in pgvector over about 20 chunks seeded from the markdown documents. `weather` and `gif_search` share one `httpx` client and one error mapping, so a timeout, bad key, rate limit or outage becomes a readable `ToolError`. `gif_search` is registered only when a key is configured, and the GIF is rendered from the tool result rather than from whatever the model writes. Every tool is a LangChain `BaseTool`: the pydantic `args_schema` is both the schema the model sees and the validator for what it sends back, so adding a tool is one class and one list entry.

**Storage.** Postgres holds conversations, every message (ordered by an identity column, not a timestamp) and the knowledge chunks. Supabase is Postgres with pgvector, so `DB_PROVIDER=supabase` only changes the connection string and TLS. The knowledge base is seeded by content: the stored chunks are compared with the markdown (source, text and embedding model), so an edited document or a changed embedding model is picked up and an unchanged store costs one query. The compare, re-embed and swap run in one transaction under a Postgres advisory lock, so replicas starting together embed once and cannot create duplicates, and a failed embedding leaves the old rows in place. The schema is created with `create_all` at startup.

**Cache.** Two things sit on the hot path of every message: the recent history a turn is built from, and the embedding of a knowledge search query. A small in-process LRU and TTL cache fronts both. History windows are extended in place on every write, so a conversation being talked to never re-reads its messages after the first load; conversation lists are invalidated on write; embeddings are keyed by model, kind and text. It is in-process because there is one backend replica; with several, the same interface would sit in front of Redis.

**Frontend.** A pure reducer turns events into UI state (streaming text, tool cards, stopped and failed turns) and folds stored history into one bubble per turn. The reducer and the SSE reader are unit tested and the components are thin. Replies are rendered as Markdown, and deltas are flushed once per animation frame instead of per token. The reply language follows the device: the browser sends `navigator.language` and the system prompt pins the reply to it, because a bare arithmetic expression has no language of its own.

### Lifecycle and shutdown

The app uses FastAPI's [lifespan](https://fastapi.tiangolo.com/advanced/events/) hook, with an `AsyncExitStack` so each resource (database pool, HTTP clients, provider clients, cache) registers its cleanup the moment it is acquired. Shutdown releases exactly what startup acquired, in reverse order, even when startup fails half-way, and only resources the app created are closed; injected ones belong to the caller.

- **Early exits close everything.** `async for` does not close an inner generator when the loop ends early, so every hand-off (the SSE layer, the service, the agent stream, the route) uses `contextlib.aclosing`. When a client leaves, the whole chain down to the model's token stream is closed at once and nothing keeps generating for nobody.
- **Cancellation.** In the SSE layer's cleanup the producer task is awaited with `asyncio.wait`, not `gather`. Under Starlette's cancel scope, `gather` forwards its own cancellation to the producer as a second `cancel()`, which interrupts LangGraph half-way through its cleanup.
- **Shutdown drains replies.** uvicorn cancels open requests and starts the lifespan shutdown without waiting for them. `ChatService.aclose()` waits for in-flight turns and the writes of their partial replies before the connection pool is disposed. The container runs uvicorn with `--timeout-graceful-shutdown 5` and compose sets a 15 s stop grace period, so `docker stop` exits cleanly with the reply stored as `interrupted`.
- **Memory stays flat.** The soak script runs 3,000 rounds against the real app; allocated blocks go from 1,079,185 at round 500 to 1,079,190 at round 3,000, with no lingering tasks, and a deliberate leak (about 8 blocks per round) fails it. The cache is bounded by entry count, by age (with an amortised sweep, so expired entries are reclaimed and not just hidden) and by messages held per conversation. The check measures Python-level memory, not RSS.

## Tradeoffs

- **Hand-built graph with only LangGraph's primitives.** `StateGraph`, `add_messages`, the custom stream channel and `bind_tools`. Control flow stays visible, and interrupts, resumable runs and subgraphs remain available later. For two nodes a plain loop would do; the graph earns its place if the agent grows.
- **`create_all` instead of migrations, and no vector index.** About 20 chunks do not need an index. A real corpus needs Alembic, a fixed dimension and HNSW.
- **No authentication or rate limiting.** CORS admits any hostname on the frontend's port and no other port. There are no cookies or credentials on this API, so restricting the hostname protects nothing while breaking whichever of `localhost`, `127.0.0.1`, `0.0.0.0` or a LAN address the browser uses; the port is what keeps another local app's page from calling the API. Real auth needs real credentials and a real origin allowlist together.
- **Conversation id in `localStorage`** rather than user accounts.
- **In-process cache.** Correct for one replica only.
- **Concurrent sends.** Two simultaneous messages in one conversation can interleave in its history; a per-conversation lock would fix it.
- **Local model latency.** With a 4B model in LM Studio on a laptop, the first token of a tool-calling turn takes about 20 s (prompt processing of the system prompt and tool schemas). Gemini Flash-Lite answers in under a second. The UI shows a thinking state and the heartbeat keeps the connection open, but a local deployment would want prompt caching or a smaller tool prompt.

## Scaling notes

**Fast responses when expensive operations spike.** Keep the chat path off the expensive path. A tool enqueues the job and returns a job id immediately; the reply says the work is under way and the result arrives later, as an event on the same stream if it is still open and as a push message otherwise. Workers pull from a queue with per-user concurrency limits and a priority lane for interactive requests. Admission control (queue depth, token buckets per user) degrades to a "high demand" message instead of timeouts. Cache identical requests, serve a downscaled preview before the full render, and autoscale workers on queue latency rather than CPU.

**Long-running tasks that don't block chat.** A persisted task record (`pending`, `running`, `done`, `failed`) is executed by a worker, with progress written to the record. To the chat loop, "task started" is just a tool result, so the user keeps chatting, and completion arrives as a new assistant message. Idempotency keys and a visibility timeout make retries safe, and a dead-letter queue catches poison jobs. The seam exists here: a `start_job` tool plus a `task_update` event type is the whole chat-side change.

**Conversation data for millions of users.** Messages are append-only and read by conversation in recency order, which suits Postgres partitioned by time or by conversation hash, with a `(conversation_id, seq)` index, the shape used here. The last N messages of hot conversations sit in Redis so building model context needs no query; the database stays the source of truth, with one batch insert per turn. Old partitions move to cheap storage, and per-user and per-country retention and deletion (GDPR) become partition operations. Analytics go to a warehouse through change data capture. Beyond tens of terabytes, the same access pattern fits a wide-column store (Cassandra, DynamoDB) keyed by conversation.

**Cheaper versus more capable model.** Route by task: chit-chat and tool dispatch go to the cheap model, long reasoning and low-confidence answers to the capable one, escalating when a cheap answer fails a check. Before switching, build an offline eval set from anonymised real conversations: rubric grading by an LLM judge calibrated against human labels, tool-call accuracy (checked deterministically, as the tests here do), refusal rates, and cost and latency per turn. Then A/B test on thumbs, regeneration rate and retention, with the eval set and judge versioned so a regression shows up as a diff.

## Next steps

1. Alembic migrations, an HNSW index with a fixed embedding dimension, and a route to re-seed the knowledge base.
2. Per-conversation locking, authentication and rate limiting.
3. A job table and a `task_update` event for long-running work, plus a LangGraph checkpointer for resumable and human-in-the-loop turns.
4. Structured logs with a request id on every event, and OpenTelemetry spans around model and tool calls.
5. Prompt caching for the local provider and a latency panel in the UI.
6. An end-to-end browser test against the scripted model.
