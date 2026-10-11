"use client";

import { useEffect, useRef, useState, type KeyboardEvent } from "react";

import type { ConversationSummary, HealthInfo } from "@/lib/types";
import { formatRelative } from "@/lib/time";

import { ChatIcon, CloseIcon, PencilIcon, PlusIcon, TrashIcon, Wordmark } from "./Icons";

export const TITLE_MAX_CHARS = 80;

interface Props {
  conversations: ConversationSummary[];
  activeId: string | null;
  health: HealthInfo | null;
  open: boolean;
  onClose: () => void;
  onNew: () => void;
  onSelect: (id: string) => void;
  onRename: (id: string, title: string) => void;
  onDelete: (id: string) => void;
}

export function Sidebar({
  conversations,
  activeId,
  health,
  open,
  onClose,
  onNew,
  onSelect,
  onRename,
  onDelete,
}: Props) {
  const [editingId, setEditingId] = useState<string | null>(null);

  return (
    <>
      {/* backdrop on small screens */}
      <div
        onClick={onClose}
        className={`fixed inset-0 z-20 bg-ink/30 backdrop-blur-[2px] transition-opacity md:hidden ${
          open ? "opacity-100" : "pointer-events-none opacity-0"
        }`}
        aria-hidden
      />
      <aside
        className={`fixed inset-y-0 left-0 z-30 flex w-[280px] flex-col border-r border-line bg-surface transition-transform md:static md:z-auto md:translate-x-0 ${
          open ? "translate-x-0" : "-translate-x-full"
        }`}
        aria-label="Conversations"
      >
        <div className="flex items-center justify-between px-4 pt-4 pb-2">
          <Wordmark />
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg p-1.5 text-muted hover:bg-surface-2 md:hidden"
            aria-label="Close menu"
          >
            <CloseIcon />
          </button>
        </div>

        <div className="px-3 pt-2 pb-3">
          <button
            type="button"
            onClick={onNew}
            className="flex w-full items-center gap-2 rounded-xl bg-brand-600 px-3.5 py-2.5 text-sm font-semibold text-white shadow-soft transition hover:bg-brand-700 active:scale-[0.99]"
          >
            <PlusIcon width={18} height={18} />
            New chat
          </button>
        </div>

        <nav className="flex-1 overflow-y-auto px-2 pb-2">
          {conversations.length === 0 ? (
            <p className="px-3 py-6 text-center text-sm text-muted">
              No conversations yet. Say hi to start one.
            </p>
          ) : (
            <ul className="space-y-0.5">
              {conversations.map((c) =>
                editingId === c.id ? (
                  <li key={c.id}>
                    <TitleEditor
                      initial={c.title}
                      onSave={(title) => {
                        setEditingId(null);
                        if (title !== c.title) onRename(c.id, title);
                      }}
                      onCancel={() => setEditingId(null)}
                    />
                  </li>
                ) : (
                  <ConversationRow
                    key={c.id}
                    conversation={c}
                    active={c.id === activeId}
                    onSelect={() => onSelect(c.id)}
                    onEdit={() => setEditingId(c.id)}
                    onDelete={() => onDelete(c.id)}
                  />
                ),
              )}
            </ul>
          )}
        </nav>

        <footer className="border-t border-line px-4 py-3 text-xs text-muted">
          {health ? (
            <span className="flex items-center gap-2">
              <span
                className={`inline-block size-2 rounded-full ${health.db_ok ? "bg-mint-600" : "bg-coral-500"}`}
              />
              <span className="truncate">
                {health.llm_provider} · {health.llm_model.split("/").pop()}
              </span>
            </span>
          ) : (
            <span className="flex items-center gap-2">
              <span className="inline-block size-2 rounded-full bg-coral-500" />
              backend unreachable
            </span>
          )}
        </footer>
      </aside>
    </>
  );
}

function ConversationRow({
  conversation: c,
  active,
  onSelect,
  onEdit,
  onDelete,
}: {
  conversation: ConversationSummary;
  active: boolean;
  onSelect: () => void;
  onEdit: () => void;
  onDelete: () => void;
}) {
  const action =
    "rounded-lg p-1.5 text-muted opacity-0 transition group-hover:opacity-100 focus:opacity-100";
  return (
    <li className="group relative">
      <button
        type="button"
        onClick={onSelect}
        onDoubleClick={onEdit}
        title="Double-click to rename"
        className={`flex w-full items-start gap-2.5 rounded-xl px-3 py-2.5 text-left transition ${
          active
            ? "bg-brand-50 text-brand-800 dark:bg-brand-600/20 dark:text-brand-100"
            : "text-ink hover:bg-surface-2"
        }`}
      >
        <ChatIcon
          width={18}
          height={18}
          className={`mt-0.5 shrink-0 ${active ? "text-brand-600" : "text-muted"}`}
        />
        <span className="min-w-0 flex-1">
          <span className="block truncate pr-14 text-sm font-medium">{c.title}</span>
          <span className="block text-xs text-muted">
            {formatRelative(c.updated_at)} · {c.message_count}{" "}
            {c.message_count === 1 ? "message" : "messages"}
          </span>
        </span>
      </button>
      <span className="absolute top-2 right-2 flex">
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            onEdit();
          }}
          className={`${action} hover:bg-brand-50 hover:text-brand-700 dark:hover:bg-brand-600/20`}
          aria-label={`Rename "${c.title}"`}
          title="Rename"
        >
          <PencilIcon width={16} height={16} />
        </button>
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            onDelete();
          }}
          className={`${action} hover:bg-coral-100 hover:text-coral-700 dark:hover:bg-coral-700/30`}
          aria-label={`Delete "${c.title}"`}
          title="Delete"
        >
          <TrashIcon width={16} height={16} />
        </button>
      </span>
    </li>
  );
}

/** Inline rename: Enter saves, Escape cancels, blur saves (a common expectation). */
function TitleEditor({
  initial,
  onSave,
  onCancel,
}: {
  initial: string;
  onSave: (title: string) => void;
  onCancel: () => void;
}) {
  const [value, setValue] = useState(initial);
  const ref = useRef<HTMLInputElement>(null);
  const done = useRef(false);

  useEffect(() => {
    ref.current?.focus();
    ref.current?.select();
  }, []);

  function commit() {
    if (done.current) return;
    done.current = true;
    const title = value.trim().replace(/\s+/g, " ");
    if (title) onSave(title);
    else onCancel();
  }

  function cancel() {
    if (done.current) return;
    done.current = true;
    onCancel();
  }

  function onKeyDown(e: KeyboardEvent<HTMLInputElement>) {
    if (e.key === "Enter") {
      e.preventDefault();
      commit();
    } else if (e.key === "Escape") {
      e.preventDefault();
      cancel();
    }
  }

  return (
    <div className="flex items-center gap-2 rounded-xl bg-brand-50 px-2 py-1.5 ring-2 ring-brand-300 dark:bg-brand-600/20">
      <input
        ref={ref}
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={onKeyDown}
        onBlur={commit}
        maxLength={TITLE_MAX_CHARS}
        aria-label="Conversation name"
        className="min-w-0 flex-1 bg-transparent px-1 text-sm font-medium text-ink outline-none"
      />
      <span className="text-[10px] text-muted">Enter · Esc</span>
    </div>
  );
}
