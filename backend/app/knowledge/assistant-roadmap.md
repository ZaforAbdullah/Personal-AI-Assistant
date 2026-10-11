# What comes next

## Near term

Database migrations in place of create-all at startup, a vector index on the knowledge table, and a route to re-seed the knowledge base without a restart.

## Longer term

Per-conversation locking so two simultaneous messages cannot interleave, authentication and rate limiting, and a job table with a task update event for long-running work such as image generation. A LangGraph checkpointer would make turns resumable, and OpenTelemetry spans around model and tool calls would show where time goes.
