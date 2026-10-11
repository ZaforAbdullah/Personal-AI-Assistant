import { describe, expect, it } from "vitest";

import { chatReducer, foldHistory, initialState, type ChatState } from "./chatReducer";
import type { AgentEvent, StoredMessage } from "./types";

function run(state: ChatState, events: AgentEvent[]): ChatState {
  return events.reduce((s, event) => chatReducer(s, { type: "event", event }), state);
}

const sent = chatReducer(initialState, {
  type: "send",
  text: "2+2?",
  userId: "u1",
  assistantId: "a1",
});

describe("chatReducer streaming", () => {
  it("adds a user message and a streaming assistant placeholder", () => {
    expect(sent.streaming).toBe(true);
    expect(sent.messages.map((m) => [m.role, m.status])).toEqual([
      ["user", "complete"],
      ["assistant", "streaming"],
    ]);
  });

  it("applies a full tool-using reply", () => {
    const state = run(sent, [
      { type: "conversation", conversation_id: "c1" },
      { type: "tool_start", call_id: "k", name: "calculator", args: { expression: "2+2" } },
      { type: "tool_result", call_id: "k", name: "calculator", result: { result: 4 } },
      { type: "text_delta", text: "It's " },
      { type: "text_delta", text: "4." },
      { type: "done", message_id: "m9" },
    ]);
    expect(state.conversationId).toBe("c1");
    expect(state.streaming).toBe(false);
    const assistant = state.messages[1];
    expect(assistant.id).toBe("m9");
    expect(assistant.text).toBe("It's 4.");
    expect(assistant.status).toBe("complete");
    expect(assistant.toolCalls).toEqual([
      { callId: "k", name: "calculator", args: { expression: "2+2" }, status: "done", result: { result: 4 } },
    ]);
  });

  it("records tool errors on the right call", () => {
    const state = run(sent, [
      { type: "tool_start", call_id: "a", name: "knowledge_search", args: { query: "x" } },
      { type: "tool_start", call_id: "b", name: "calculator", args: { expression: "1/0" } },
      { type: "tool_error", call_id: "b", name: "calculator", error: "division by zero" },
      { type: "tool_result", call_id: "a", name: "knowledge_search", result: [] },
    ]);
    const [a, b] = state.messages[1].toolCalls;
    expect(a.status).toBe("done");
    expect(b).toMatchObject({ status: "error", error: "division by zero" });
  });

  it("keeps partial text and surfaces the error on a server error event", () => {
    const state = run(sent, [
      { type: "text_delta", text: "Half" },
      { type: "error", message: "The model is unavailable right now." },
      { type: "done", message_id: null },
    ]);
    expect(state.error).toBe("The model is unavailable right now.");
    expect(state.messages[1]).toMatchObject({ text: "Half", status: "error" });
    expect(state.streaming).toBe(false);
  });

  it("marks running tools when the stream fails or is stopped", () => {
    const mid = run(sent, [
      { type: "tool_start", call_id: "k", name: "calculator", args: {} },
    ]);
    const failed = chatReducer(mid, { type: "stream_failed", message: "network down" });
    expect(failed.error).toBe("network down");
    expect(failed.messages[1].status).toBe("error");
    expect(failed.messages[1].toolCalls[0].status).toBe("error");

    const stopped = chatReducer(mid, { type: "stream_aborted" });
    expect(stopped.error).toBeNull();
    expect(stopped.messages[1].status).toBe("interrupted");
    expect(stopped.messages[1].toolCalls[0].error).toBe("stopped");
  });
});

describe("foldHistory", () => {
  const rows: StoredMessage[] = [
    row("1", "user", "2+2?"),
    { ...row("2", "assistant", ""), tool_calls: [{ id: "k", name: "calculator", arguments: '{"expression":"2+2"}' }] },
    { ...row("3", "tool", '{"result":4}'), tool_call_id: "k", tool_name: "calculator" },
    row("4", "assistant", "It's 4."),
    row("5", "user", "and 1/0?"),
    { ...row("6", "assistant", ""), tool_calls: [{ id: "z", name: "calculator", arguments: '{"expression":"1/0"}' }] },
    { ...row("7", "tool", '{"error":"division by zero"}'), tool_call_id: "z", tool_name: "calculator" },
    { ...row("8", "assistant", "Can't divide by zero."), status: "interrupted" },
  ];

  it("folds tool exchanges into one assistant bubble per turn", () => {
    const view = foldHistory(rows);
    expect(view.map((m) => m.role)).toEqual(["user", "assistant", "user", "assistant"]);
    expect(view[1]).toMatchObject({
      id: "4",
      text: "It's 4.",
      status: "complete",
      toolCalls: [{ callId: "k", name: "calculator", args: { expression: "2+2" }, status: "done", result: { result: 4 } }],
    });
    expect(view[3].status).toBe("interrupted");
    expect(view[3].toolCalls[0]).toMatchObject({ status: "error", error: "division by zero" });
  });

  it("flags tool calls that never got a result", () => {
    const view = foldHistory(rows.slice(0, 2));
    expect(view[1].toolCalls[0]).toMatchObject({ status: "error", error: "no result recorded" });
  });
});

function row(id: string, role: StoredMessage["role"], content: string): StoredMessage {
  return {
    id,
    role,
    content,
    tool_calls: null,
    tool_call_id: null,
    tool_name: null,
    status: "complete",
    created_at: "2026-01-01T00:00:00Z",
  };
}
