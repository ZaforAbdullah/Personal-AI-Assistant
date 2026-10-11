import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import type { MessageView, ToolCallView } from "@/lib/chatReducer";
import { delatex } from "@/lib/markdown";

import { AssistantMark } from "./Icons";
import { ToolCallCard } from "./ToolCallCard";

export function MessageBubble({ message }: { message: MessageView }) {
  if (message.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[80%] rounded-2xl rounded-br-md bg-user-bubble px-4 py-2.5 text-[15px] leading-relaxed whitespace-pre-wrap text-user-bubble-text shadow-soft">
          {message.text}
        </div>
      </div>
    );
  }

  const thinking = message.status === "streaming" && !message.text && message.toolCalls.length === 0;

  return (
    <div className="flex gap-3">
      <div className="pt-0.5">
        <AssistantMark />
      </div>
      <div className="min-w-0 flex-1">
        {message.toolCalls.map((call) => (
          <ToolCallCard key={call.callId} call={call} />
        ))}
        {message.toolCalls.filter(isGif).map((call) => (
          <GifPreview key={`gif-${call.callId}`} gif={call.result as GifResult} />
        ))}
        {thinking ? (
          <span className="inline-flex items-center gap-1 py-2 text-brand-500" aria-label="thinking">
            <span className="dot inline-block size-1.5 rounded-full bg-current" />
            <span className="dot inline-block size-1.5 rounded-full bg-current" />
            <span className="dot inline-block size-1.5 rounded-full bg-current" />
          </span>
        ) : (
          <div className="md text-[15px] leading-relaxed text-ink">
            {/* Models write Markdown; render it (GFM: lists, tables, strikethrough).
                Streaming re-renders per delta, fine at chat-reply sizes. */}
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{delatex(message.text)}</ReactMarkdown>
            {message.status === "streaming" && (
              <span className="caret ml-0.5 inline-block h-[1em] w-[2px] translate-y-[2px] bg-brand-500" />
            )}
          </div>
        )}
        {message.status === "complete" && message.elapsedMs != null && (
          <p className="mt-1 text-[11px] text-muted" title="Server-side time for this turn">
            {message.toolCalls.length > 0
              ? `${message.toolCalls.length} tool call${message.toolCalls.length > 1 ? "s" : ""} · `
              : ""}
            {(message.elapsedMs / 1000).toFixed(1)}s
          </p>
        )}
        {message.status === "interrupted" && (
          <p className="mt-1 text-xs text-muted">Stopped.</p>
        )}
        {message.status === "error" && (
          <p className="mt-1 text-xs text-coral-700 dark:text-coral-300">This reply did not finish.</p>
        )}
      </div>
    </div>
  );
}

interface GifResult {
  url: string;
  title?: string;
  page?: string;
  width?: number;
  height?: number;
}

function isGif(call: ToolCallView): boolean {
  const r = call.result as Partial<GifResult> | undefined;
  return call.name === "gif_search" && call.status === "done" && typeof r?.url === "string";
}

/** The GIF a gif_search call found, rendered where the reply is (the tool result is the
 * source of truth; the model only describes it). GIPHY's terms ask for attribution. */
function GifPreview({ gif }: { gif: GifResult }) {
  return (
    <figure className="my-2 w-fit max-w-xs">
      {/* eslint-disable-next-line @next/next/no-img-element -- remote GIF, no optimisation wanted */}
      <img
        src={gif.url}
        alt={gif.title ?? "GIF"}
        width={gif.width || undefined}
        height={gif.height || undefined}
        loading="lazy"
        className="h-auto max-w-full rounded-xl shadow-soft"
      />
      <figcaption className="mt-1 text-[11px] text-muted">
        {gif.page ? (
          <a href={gif.page} target="_blank" rel="noreferrer" className="hover:underline">
            {gif.title ?? "GIF"} · via GIPHY
          </a>
        ) : (
          <>{gif.title ?? "GIF"} · via GIPHY</>
        )}
      </figcaption>
    </figure>
  );
}
