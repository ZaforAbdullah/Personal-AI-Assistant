/** Backend base URL, baked in at build time (see .env.example / docker-compose.yml). */
export const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(
  /\/$/,
  "",
);

export const CONVERSATION_KEY = "assistant.conversationId";
