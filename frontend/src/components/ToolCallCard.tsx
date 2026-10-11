import type { ToolCallView } from "@/lib/chatReducer";

import { CalcIcon, CloudIcon, ImageIcon, SearchIcon, ToolIcon } from "./Icons";

const LABELS: Record<string, { running: string; done: string }> = {
  calculator: { running: "Calculating", done: "Calculated" },
  knowledge_search: { running: "Searching the knowledge base", done: "Searched the knowledge base" },
  weather: { running: "Checking the weather", done: "Checked the weather" },
  gif_search: { running: "Looking for a GIF", done: "Found a GIF" },
};

function labelFor(call: ToolCallView): string {
  const l = LABELS[call.name];
  if (!l) return call.status === "running" ? `Using ${call.name}…` : `Used ${call.name}`;
  return call.status === "running" ? `${l.running}…` : l.done;
}

function Icon({ name, className }: { name: string; className?: string }) {
  const p = { width: 16, height: 16, className };
  if (name === "calculator") return <CalcIcon {...p} />;
  if (name === "knowledge_search") return <SearchIcon {...p} />;
  if (name === "weather") return <CloudIcon {...p} />;
  if (name === "gif_search") return <ImageIcon {...p} />;
  return <ToolIcon {...p} />;
}

export function ToolCallCard({ call }: { call: ToolCallView }) {
  const summary = summarizeArgs(call);
  const tone =
    call.status === "error"
      ? "border-coral-300 bg-coral-100/60 text-coral-700 dark:border-coral-700/50 dark:bg-coral-700/15 dark:text-coral-300"
      : call.status === "done"
        ? "border-mint-300 bg-mint-100/70 text-mint-600 dark:border-mint-600/40 dark:bg-mint-600/15 dark:text-mint-300"
        : "border-brand-200 bg-brand-50 text-brand-700 dark:border-brand-600/40 dark:bg-brand-600/15 dark:text-brand-200";

  return (
    <details className={`group my-2 max-w-xl rounded-xl border text-sm ${tone}`} data-testid="tool-call">
      <summary className="flex cursor-pointer list-none items-center gap-2.5 px-3 py-2 select-none">
        <span className="relative inline-flex size-6 shrink-0 items-center justify-center rounded-lg bg-surface/70">
          {call.status === "running" ? (
            <span
              aria-label="running"
              className="inline-block size-3.5 animate-spin rounded-full border-2 border-current border-t-transparent"
            />
          ) : (
            <Icon name={call.name} />
          )}
        </span>
        <span className="font-medium">{labelFor(call)}</span>
        {summary && (
          <span className="truncate font-mono text-xs opacity-80" title={summary}>
            {summary}
          </span>
        )}
        <span className="ml-auto text-[11px] tracking-wide uppercase opacity-60 group-open:hidden">
          details
        </span>
      </summary>
      <div className="space-y-2 border-t border-current/15 px-3 py-2.5 font-mono text-xs text-ink">
        <div>
          <div className="mb-1 text-[11px] tracking-wide text-muted uppercase">
            {call.name} · arguments
          </div>
          <pre className="overflow-x-auto rounded-lg bg-surface/70 p-2 whitespace-pre-wrap">
            {JSON.stringify(call.args, null, 2)}
          </pre>
        </div>
        {call.status === "done" && (
          <div>
            <div className="mb-1 text-[11px] tracking-wide text-muted uppercase">result</div>
            <pre className="max-h-64 overflow-auto rounded-lg bg-surface/70 p-2 whitespace-pre-wrap">
              {JSON.stringify(call.result, null, 2)}
            </pre>
          </div>
        )}
        {call.status === "error" && (
          <div className="rounded-lg bg-surface/70 p-2 text-coral-700 dark:text-coral-300">
            {call.error}
          </div>
        )}
      </div>
    </details>
  );
}

function summarizeArgs(call: ToolCallView): string {
  const first = Object.values(call.args)[0];
  if (typeof first === "string") return first.length > 48 ? `${first.slice(0, 45)}…` : first;
  return "";
}
