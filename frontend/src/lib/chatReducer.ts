/** UI state for the chat, driven by agent events. Pure so it can be unit-tested. */

import type { AgentEvent, StoredMessage } from "./types";

export type ToolStatus = "running" | "done" | "error";

export interface ToolCallView {
  callId: string;
  name: string;
  args: Record<string, unknown>;
  status: ToolStatus;
  result?: unknown;
  error?: string;
}

export type MessageStatus = "streaming" | "complete" | "error" | "interrupted";

export interface MessageView {
  id: string;
  role: "user" | "assistant";
  text: string;
  toolCalls: ToolCallView[];
  status: MessageStatus;
  /** Wall time of the turn as reported by the server's `done` event. */
  elapsedMs?: number;
}

export interface ChatState {
  conversationId: string | null;
  messages: MessageView[];
  streaming: boolean;
  error: string | null;
}

export type ChatAction =
  | { type: "send"; text: string; userId: string; assistantId: string }
  | { type: "event"; event: AgentEvent }
  | { type: "stream_failed"; message: string }
  | { type: "stream_aborted" }
  | { type: "history_loaded"; conversationId: string; messages: StoredMessage[] }
  | { type: "reset" };

export const initialState: ChatState = {
  conversationId: null,
  messages: [],
  streaming: false,
  error: null,
};

export function chatReducer(state: ChatState, action: ChatAction): ChatState {
  switch (action.type) {
    case "send":
      return {
        ...state,
        error: null,
        streaming: true,
        messages: [
          ...state.messages,
          { id: action.userId, role: "user", text: action.text, toolCalls: [], status: "complete" },
          { id: action.assistantId, role: "assistant", text: "", toolCalls: [], status: "streaming" },
        ],
      };
    case "event":
      return applyEvent(state, action.event);
    case "stream_failed":
      return {
        ...state,
        streaming: false,
        error: action.message,
        messages: updateLastAssistant(state.messages, (m) => ({
          ...m,
          status: "error",
          toolCalls: m.toolCalls.map((t) =>
            t.status === "running" ? { ...t, status: "error", error: "interrupted" } : t,
          ),
        })),
      };
    case "stream_aborted":
      return {
        ...state,
        streaming: false,
        messages: updateLastAssistant(state.messages, (m) => ({
          ...m,
          status: "interrupted",
          toolCalls: m.toolCalls.map((t) =>
            t.status === "running" ? { ...t, status: "error", error: "stopped" } : t,
          ),
        })),
      };
    case "history_loaded":
      return {
        ...initialState,
        conversationId: action.conversationId,
        messages: foldHistory(action.messages),
      };
    case "reset":
      return initialState;
  }
}

function applyEvent(state: ChatState, event: AgentEvent): ChatState {
  switch (event.type) {
    case "conversation":
      return { ...state, conversationId: event.conversation_id };
    case "text_delta":
      return {
        ...state,
        messages: updateLastAssistant(state.messages, (m) => ({ ...m, text: m.text + event.text })),
      };
    case "tool_start":
      return {
        ...state,
        messages: updateLastAssistant(state.messages, (m) => ({
          ...m,
          toolCalls: [
            ...m.toolCalls,
            { callId: event.call_id, name: event.name, args: event.args, status: "running" },
          ],
        })),
      };
    case "tool_result":
      return {
        ...state,
        messages: updateLastAssistant(state.messages, (m) => ({
          ...m,
          toolCalls: m.toolCalls.map((t) =>
            t.callId === event.call_id ? { ...t, status: "done", result: event.result } : t,
          ),
        })),
      };
    case "tool_error":
      return {
        ...state,
        messages: updateLastAssistant(state.messages, (m) => ({
          ...m,
          toolCalls: m.toolCalls.map((t) =>
            t.callId === event.call_id ? { ...t, status: "error", error: event.error } : t,
          ),
        })),
      };
    case "error":
      return {
        ...state,
        error: event.message,
        messages: updateLastAssistant(state.messages, (m) => ({ ...m, status: "error" })),
      };
    case "done":
      return {
        ...state,
        streaming: false,
        messages: updateLastAssistant(state.messages, (m) => ({
          ...m,
          id: event.message_id ?? m.id,
          status: m.status === "streaming" ? "complete" : m.status,
          elapsedMs: event.elapsed_ms ?? m.elapsedMs,
        })),
      };
  }
}

function updateLastAssistant(
  messages: MessageView[],
  update: (m: MessageView) => MessageView,
): MessageView[] {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === "assistant") {
      const next = messages.slice();
      next[i] = update(messages[i]);
      return next;
    }
  }
  return messages;
}

/**
 * Stored history is flat (user / assistant-with-tool-calls / tool / assistant ...). The UI
 * shows one assistant bubble per user turn, so consecutive non-user rows are folded
 * together: tool calls become cards, tool rows fill in their results, text concatenates.
 */
export function foldHistory(rows: StoredMessage[]): MessageView[] {
  const out: MessageView[] = [];
  let current: MessageView | null = null;

  for (const row of rows) {
    if (row.role === "user") {
      current = null;
      out.push({ id: row.id, role: "user", text: row.content, toolCalls: [], status: "complete" });
      continue;
    }
    if (!current) {
      current = { id: row.id, role: "assistant", text: "", toolCalls: [], status: "complete" };
      out.push(current);
    }
    if (row.role === "assistant") {
      current.id = row.id;
      if (row.content) current.text = current.text ? `${current.text}\n${row.content}` : row.content;
      for (const call of row.tool_calls ?? []) {
        current.toolCalls.push({
          callId: call.id,
          name: call.name,
          args: safeParse(call.arguments),
          status: "error",
          error: "no result recorded",
        });
      }
      if (row.status !== "complete") current.status = row.status;
    } else if (row.role === "tool") {
      const call = current.toolCalls.find((t) => t.callId === row.tool_call_id);
      const payload = safeParse(row.content);
      if (call) {
        if (payload && typeof payload === "object" && "error" in payload) {
          call.status = "error";
          call.error = String((payload as { error: unknown }).error);
        } else {
          call.status = "done";
          call.result = payload;
          call.error = undefined;
        }
      }
    }
  }
  return out;
}

function safeParse(text: string): Record<string, unknown> {
  try {
    const value = JSON.parse(text);
    return value && typeof value === "object" ? (value as Record<string, unknown>) : { value };
  } catch {
    return { raw: text };
  }
}
