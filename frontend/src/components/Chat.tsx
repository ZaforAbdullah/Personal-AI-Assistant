"use client";

import { useCallback, useEffect, useReducer, useRef, useState } from "react";

import { chatReducer, initialState } from "@/lib/chatReducer";
import { API_URL, CONVERSATION_KEY } from "@/lib/config";
import {
  ApiError,
  deleteConversation,
  fetchConversations,
  fetchHistory,
  renameConversation,
  streamChat,
} from "@/lib/sse";
import type { ConversationSummary, HealthInfo } from "@/lib/types";

import { Composer } from "./Composer";
import { AssistantMark, MenuIcon, PlusIcon } from "./Icons";
import { MessageBubble } from "./MessageBubble";
import { Sidebar } from "./Sidebar";

/** Suggestion chips; the ones tied to a tool appear only when /health lists that tool. */
const SUGGESTIONS: { text: string; tool?: string }[] = [
  { text: "What is 17 × 23 + 5?" },
  { text: "Which languages does the assistant reply in?" },
  { text: "What's the weather in Madrid right now?", tool: "weather" },
  { text: "Show me a gif of a happy cat", tool: "gif_search" },
  { text: "How does this app stream replies?" },
];

export function Chat() {
  const [state, dispatch] = useReducer(chatReducer, initialState);
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [health, setHealth] = useState<HealthInfo | null>(null);
  const [loadingHistory, setLoadingHistory] = useState(true);
  const [menuOpen, setMenuOpen] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  const refreshConversations = useCallback(async () => {
    try {
      setConversations(await fetchConversations(API_URL));
    } catch {
      /* the sidebar just stays as it was */
    }
  }, []);

  // Restore the last conversation (id kept in localStorage), load the list, show the provider.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const saved = window.localStorage.getItem(CONVERSATION_KEY);
        if (saved) {
          const messages = await fetchHistory(API_URL, saved);
          if (!cancelled) dispatch({ type: "history_loaded", conversationId: saved, messages });
        }
      } catch (err) {
        // Unknown id (e.g. database reset): start fresh instead of failing.
        if (err instanceof ApiError && err.status === 404) {
          window.localStorage.removeItem(CONVERSATION_KEY);
        } else {
          dispatch({ type: "stream_failed", message: describe(err) });
        }
      } finally {
        if (!cancelled) setLoadingHistory(false);
      }
    })();
    fetchConversations(API_URL)
      .then((list) => !cancelled && setConversations(list))
      .catch(() => {});
    fetch(`${API_URL}/health`)
      .then((r) => r.json())
      .then((h: HealthInfo) => !cancelled && setHealth(h))
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (state.conversationId) window.localStorage.setItem(CONVERSATION_KEY, state.conversationId);
  }, [state.conversationId]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ block: "end" });
  }, [state.messages]);

  const send = useCallback(
    async (text: string) => {
      const controller = new AbortController();
      abortRef.current = controller;
      dispatch({ type: "send", text, userId: uid(), assistantId: uid() });
      // deltas arrive per token; re-rendering Markdown for each is wasted work, so they are
      // flushed once per animation frame. Other events are dispatched immediately, in order.
      let pending = "";
      let frame: number | null = null;
      const flush = () => {
        frame = null;
        if (pending) {
          dispatch({ type: "event", event: { type: "text_delta", text: pending } });
          pending = "";
        }
      };
      try {
        for await (const event of streamChat({
          apiUrl: API_URL,
          message: text,
          conversationId: state.conversationId,
          signal: controller.signal,
        })) {
          if (event.type === "text_delta") {
            pending += event.text;
            frame ??= requestAnimationFrame(flush);
          } else {
            if (frame !== null) cancelAnimationFrame(frame);
            flush();
            dispatch({ type: "event", event });
          }
        }
        if (frame !== null) cancelAnimationFrame(frame);
        flush();
      } catch (err) {
        if (frame !== null) cancelAnimationFrame(frame);
        flush();
        if (controller.signal.aborted) {
          dispatch({ type: "stream_aborted" });
        } else if (err instanceof ApiError && err.status === 404) {
          window.localStorage.removeItem(CONVERSATION_KEY);
          dispatch({
            type: "stream_failed",
            message: "That conversation no longer exists. Start a new chat.",
          });
        } else {
          dispatch({ type: "stream_failed", message: describe(err) });
        }
      } finally {
        abortRef.current = null;
        void refreshConversations();
      }
    },
    [state.conversationId, refreshConversations],
  );

  function stop() {
    abortRef.current?.abort();
  }

  function newChat() {
    abortRef.current?.abort();
    window.localStorage.removeItem(CONVERSATION_KEY);
    dispatch({ type: "reset" });
    setMenuOpen(false);
  }

  async function openConversation(id: string) {
    if (id === state.conversationId) {
      setMenuOpen(false);
      return;
    }
    abortRef.current?.abort();
    setMenuOpen(false);
    try {
      const messages = await fetchHistory(API_URL, id);
      dispatch({ type: "history_loaded", conversationId: id, messages });
    } catch (err) {
      dispatch({ type: "stream_failed", message: describe(err) });
      void refreshConversations();
    }
  }

  async function retitleConversation(id: string, title: string) {
    // Optimistic: show the new name at once, then settle on what the server stored.
    setConversations((list) => list.map((c) => (c.id === id ? { ...c, title } : c)));
    try {
      const saved = await renameConversation(API_URL, id, title);
      setConversations((list) => list.map((c) => (c.id === id ? saved : c)));
    } catch (err) {
      dispatch({ type: "stream_failed", message: describe(err) });
      void refreshConversations();
    }
  }

  async function removeConversation(id: string) {
    setConversations((list) => list.filter((c) => c.id !== id));
    if (id === state.conversationId) newChat();
    try {
      await deleteConversation(API_URL, id);
    } finally {
      void refreshConversations();
    }
  }

  return (
    <div className="flex h-dvh w-full overflow-hidden bg-canvas">
      <Sidebar
        conversations={conversations}
        activeId={state.conversationId}
        health={health}
        open={menuOpen}
        onClose={() => setMenuOpen(false)}
        onNew={newChat}
        onSelect={openConversation}
        onRename={retitleConversation}
        onDelete={removeConversation}
      />

      <div className="flex min-w-0 flex-1 flex-col">
        {/* Small-screen top bar */}
        <header className="flex items-center gap-2 border-b border-line bg-surface/80 px-3 py-2 backdrop-blur md:hidden">
          <button
            type="button"
            onClick={() => setMenuOpen(true)}
            className="rounded-lg p-2 text-ink hover:bg-surface-2"
            aria-label="Open menu"
          >
            <MenuIcon />
          </button>
          <span className="flex items-center gap-2 font-semibold text-brand-600">
            <AssistantMark size={22} /> Personal AI Assistant
          </span>
          <button
            type="button"
            onClick={newChat}
            className="ml-auto rounded-lg p-2 text-ink hover:bg-surface-2"
            aria-label="New chat"
          >
            <PlusIcon />
          </button>
        </header>

        {/* The scroll region spans the whole column, so the scrollbar sits at the window edge. */}
        <main className="min-h-0 flex-1 overflow-y-auto">
          <div className="mx-auto w-full max-w-3xl px-4 pt-6 pb-4 sm:px-6">
            {loadingHistory ? (
              <p className="py-10 text-center text-sm text-muted">Loading…</p>
            ) : state.messages.length === 0 ? (
              <EmptyState onPick={send} tools={health?.tools} />
            ) : (
              <div className="space-y-6">
                {state.messages.map((m) => (
                  <MessageBubble key={m.id} message={m} />
                ))}
              </div>
            )}
            <div ref={bottomRef} className="h-2" />
          </div>
        </main>

        <footer className="bg-gradient-to-t from-canvas via-canvas to-transparent px-4 pt-2 pb-4 sm:px-6">
          <div className="mx-auto w-full max-w-3xl">
            {state.error && (
              <div
                role="alert"
                className="mb-2 rounded-xl border border-coral-300 bg-coral-100 px-3 py-2 text-sm text-coral-700 dark:border-coral-700/50 dark:bg-coral-700/15 dark:text-coral-300"
              >
                {state.error}
              </div>
            )}
            <Composer
              disabled={loadingHistory}
              streaming={state.streaming}
              onSend={send}
              onStop={stop}
            />
            <p className="mt-2 text-center text-[11px] text-muted">
              The assistant can use tools; you&apos;ll see a card whenever it does. Enter to send,
              Shift+Enter for a new line.
            </p>
          </div>
        </footer>
      </div>
    </div>
  );
}

function EmptyState({ onPick, tools }: { onPick: (text: string) => void; tools?: string[] }) {
  const chips = SUGGESTIONS.filter((s) => !s.tool || !tools || tools.includes(s.tool));
  return (
    <div className="flex flex-col items-center px-2 pt-[10vh] text-center">
      <AssistantMark size={56} />
      <h1 className="mt-5 text-2xl font-semibold tracking-tight text-ink">
        Hi, I&apos;m your personal AI assistant.
      </h1>
      <p className="mt-2 max-w-md text-[15px] text-muted">
        Ask me anything. I can do exact calculations and look things up in my knowledge base,
        and I&apos;ll show you when I use a tool.
      </p>
      <div className="mt-8 flex flex-wrap justify-center gap-2">
        {chips.map((s) => (
          <button
            key={s.text}
            type="button"
            onClick={() => onPick(s.text)}
            className="rounded-full border border-line bg-surface px-4 py-2 text-sm text-ink shadow-soft transition hover:border-brand-300 hover:text-brand-700"
          >
            {s.text}
          </button>
        ))}
      </div>
    </div>
  );
}

function uid(): string {
  return typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random()}`;
}

function describe(err: unknown): string {
  if (err instanceof ApiError) return err.message;
  if (err instanceof TypeError) return `Could not reach the backend at ${API_URL}.`;
  return err instanceof Error ? err.message : "Something went wrong.";
}
