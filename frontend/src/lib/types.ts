/** Wire contract with the backend. Mirrors backend/app/agent/events.py. */

export type AgentEvent =
  | { type: "conversation"; conversation_id: string }
  | { type: "text_delta"; text: string }
  | { type: "tool_start"; call_id: string; name: string; args: Record<string, unknown> }
  | { type: "tool_result"; call_id: string; name: string; result: unknown }
  | { type: "tool_error"; call_id: string; name: string; error: string }
  | { type: "done"; message_id: string | null; elapsed_ms?: number | null; tool_calls?: number }
  | { type: "error"; message: string };

export type AgentEventType = AgentEvent["type"];

/** A stored message as returned by GET /api/conversations/{id}/messages. */
export interface StoredMessage {
  id: string;
  role: "user" | "assistant" | "tool";
  content: string;
  tool_calls: { id: string; name: string; arguments: string }[] | null;
  tool_call_id: string | null;
  tool_name: string | null;
  status: "complete" | "error" | "interrupted";
  created_at: string;
}

/** One row of GET /api/conversations. */
export interface ConversationSummary {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
}

export interface HealthInfo {
  status: string;
  llm_provider: string;
  llm_model: string;
  db_provider: string;
  db_ok: boolean;
  /** Names of the tools the assistant can currently use (gif_search only with a key). */
  tools?: string[];
}
