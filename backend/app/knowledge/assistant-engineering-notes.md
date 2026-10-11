# Engineering notes

## Design principles

Keep the agent small and visible. The agent is a two-node LangGraph with no HTTP or database inside it, so every branch can be tested with a scripted model and no network. Persistence, streaming and rendering wrap it rather than live inside it.

## Performance

The first event of a reply is sent before the model is called, text is streamed as it arrives, and a tool call is announced before the tool runs, so the interface moves within milliseconds even when the first token is slow. Recent history and query embeddings are cached in process, and everything that can grow has a bound: tool timeouts, tool rounds, reply length, the cache and the streaming queue.

## Reliability

A tool can fail four ways: an expected error, invalid arguments, a timeout or an unexpected exception. None of them breaks the stream. The failure is shown to the user and passed back to the model, which explains it in prose. If a client disconnects or the server shuts down mid-reply, the partial reply is stored and the connection to the model is closed so nothing keeps generating for nobody.

## Testing

The test suite runs offline. A scripted chat model streams tool-call fragments the way a real provider does, repositories have in-memory versions, and HTTP tools run against a mock transport. A soak script drives the whole app for thousands of rounds and fails if memory, tasks or generators keep growing.
