/** SSE reader over fetch + ReadableStream (EventSource is GET-only and we need a POST body).
 * Frames end at a blank line; comment lines (": ping") are heartbeats and skipped. */

import type { AgentEvent, ConversationSummary, StoredMessage } from "./types";

export interface SSEFrame {
  event: string;
  data: string;
}

export function parseFrame(frame: string): SSEFrame | null {
  let event = "message";
  const data: string[] = [];
  for (const rawLine of frame.split("\n")) {
    const line = rawLine.replace(/\r$/, "");
    if (line === "" || line.startsWith(":")) continue;
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "event") event = value;
    else if (field === "data") data.push(value);
  }
  if (data.length === 0) return null;
  return { event, data: data.join("\n") };
}

export async function* readSSE(body: ReadableStream<Uint8Array>): AsyncGenerator<SSEFrame> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let split: number;
      while ((split = buffer.search(/\r?\n\r?\n/)) !== -1) {
        const sepLen = buffer.slice(split).match(/^\r?\n\r?\n/)![0].length;
        const frame = buffer.slice(0, split);
        buffer = buffer.slice(split + sepLen);
        const parsed = parseFrame(frame);
        if (parsed) yield parsed;
      }
    }
    const tail = parseFrame(buffer);
    if (tail) yield tail;
  } finally {
    reader.releaseLock();
  }
}

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

export interface StreamChatOptions {
  apiUrl: string;
  message: string;
  conversationId: string | null;
  signal?: AbortSignal;
}

/** POST a message and yield typed agent events as they arrive. */
export async function* streamChat(opts: StreamChatOptions): AsyncGenerator<AgentEvent> {
  const res = await fetch(`${opts.apiUrl}/api/chat`, {
    method: "POST",
    headers: { "content-type": "application/json", accept: "text/event-stream" },
    body: JSON.stringify({
      message: opts.message,
      conversation_id: opts.conversationId,
      // The device's language decides the reply language (see backend prompts.py).
      locale: typeof navigator !== "undefined" ? navigator.language : undefined,
    }),
    signal: opts.signal,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* not JSON */
    }
    throw new ApiError(res.status, typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  if (!res.body) throw new ApiError(res.status, "empty response body");
  for await (const frame of readSSE(res.body)) {
    yield JSON.parse(frame.data) as AgentEvent;
  }
}

export async function fetchHistory(apiUrl: string, conversationId: string): Promise<StoredMessage[]> {
  const res = await fetch(`${apiUrl}/api/conversations/${conversationId}/messages`);
  if (!res.ok) throw new ApiError(res.status, `could not load conversation (${res.status})`);
  return (await res.json()) as StoredMessage[];
}

export async function fetchConversations(apiUrl: string): Promise<ConversationSummary[]> {
  const res = await fetch(`${apiUrl}/api/conversations`);
  if (!res.ok) throw new ApiError(res.status, `could not list conversations (${res.status})`);
  return (await res.json()) as ConversationSummary[];
}

export async function renameConversation(
  apiUrl: string,
  conversationId: string,
  title: string,
): Promise<ConversationSummary> {
  const res = await fetch(`${apiUrl}/api/conversations/${conversationId}`, {
    method: "PATCH",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ title }),
  });
  if (!res.ok) throw new ApiError(res.status, `could not rename conversation (${res.status})`);
  return (await res.json()) as ConversationSummary;
}

export async function deleteConversation(apiUrl: string, conversationId: string): Promise<void> {
  const res = await fetch(`${apiUrl}/api/conversations/${conversationId}`, { method: "DELETE" });
  if (!res.ok && res.status !== 404) {
    throw new ApiError(res.status, `could not delete conversation (${res.status})`);
  }
}
