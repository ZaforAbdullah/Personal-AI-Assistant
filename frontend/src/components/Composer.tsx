"use client";

import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from "react";

import { SendIcon, StopIcon } from "./Icons";

interface Props {
  disabled: boolean;
  streaming: boolean;
  onSend: (text: string) => void;
  onStop: () => void;
}

export function Composer({ disabled, streaming, onSend, onStop }: Props) {
  const [text, setText] = useState("");
  const ref = useRef<HTMLTextAreaElement>(null);

  // Grow with content up to a few lines, then scroll.
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "0px";
    el.style.height = `${Math.min(el.scrollHeight, 180)}px`;
  }, [text]);

  function submit(e?: FormEvent) {
    e?.preventDefault();
    const trimmed = text.trim();
    if (!trimmed || streaming || disabled) return;
    onSend(trimmed);
    setText("");
  }

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) submit(e);
  }

  return (
    <form
      onSubmit={submit}
      className="flex items-end gap-2 rounded-2xl border border-line bg-surface p-2 pl-4 shadow-soft transition focus-within:border-brand-300 focus-within:ring-4 focus-within:ring-brand-600/10"
    >
      <textarea
        ref={ref}
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={onKeyDown}
        placeholder="Message your assistant…"
        rows={1}
        autoFocus
        disabled={disabled}
        aria-label="Message"
        className="max-h-[180px] min-h-[24px] flex-1 resize-none bg-transparent py-2 text-[15px] leading-6 text-ink outline-none placeholder:text-muted disabled:opacity-50"
      />
      {streaming ? (
        <button
          type="button"
          onClick={onStop}
          className="inline-flex size-10 shrink-0 items-center justify-center rounded-xl bg-ink text-surface transition hover:opacity-90"
          aria-label="Stop generating"
          title="Stop"
        >
          <StopIcon width={18} height={18} />
        </button>
      ) : (
        <button
          type="submit"
          disabled={disabled || !text.trim()}
          className="inline-flex size-10 shrink-0 items-center justify-center rounded-xl bg-brand-600 text-white transition hover:bg-brand-700 disabled:bg-surface-2 disabled:text-muted"
          aria-label="Send"
          title="Send (Enter)"
        >
          <SendIcon width={18} height={18} />
        </button>
      )}
    </form>
  );
}
