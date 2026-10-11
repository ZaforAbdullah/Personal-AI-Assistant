import { describe, expect, it } from "vitest";

import { parseFrame, readSSE } from "./sse";

function streamOfBytes(chunks: Uint8Array[]): ReadableStream<Uint8Array> {
  return new ReadableStream({
    start(controller) {
      for (const c of chunks) controller.enqueue(c);
      controller.close();
    },
  });
}

function streamOf(chunks: string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return streamOfBytes(chunks.map((c) => encoder.encode(c)));
}

async function collect(body: ReadableStream<Uint8Array>) {
  const out = [];
  for await (const frame of readSSE(body)) out.push(frame);
  return out;
}

describe("parseFrame", () => {
  it("reads event and data fields", () => {
    expect(parseFrame('event: text_delta\ndata: {"text":"hi"}')).toEqual({
      event: "text_delta",
      data: '{"text":"hi"}',
    });
  });

  it("ignores comments and returns null without data", () => {
    expect(parseFrame(": ping")).toBeNull();
    expect(parseFrame("event: done")).toBeNull();
  });

  it("joins multi-line data and tolerates CRLF", () => {
    expect(parseFrame("data: a\r\ndata: b\r\n")).toEqual({ event: "message", data: "a\nb" });
  });
});

describe("readSSE", () => {
  it("yields frames split at arbitrary byte boundaries, skipping heartbeats", async () => {
    const text =
      'event: conversation\ndata: {"id":1}\n\n: ping\n\nevent: text_delta\ndata: {"text":"héllo"}\n\nevent: done\ndata: {}\n\n';
    const bytes = new TextEncoder().encode(text);
    // Cut mid-line and in the middle of the two-byte "é" to exercise buffering/decoding.
    const cut = new TextEncoder().encode(text.slice(0, text.indexOf("é"))).length + 1;
    const frames = await collect(
      streamOfBytes([bytes.slice(0, 7), bytes.slice(7, cut), bytes.slice(cut)]),
    );
    expect(frames.map((f) => f.event)).toEqual(["conversation", "text_delta", "done"]);
    expect(JSON.parse(frames[1].data).text).toBe("héllo");
  });

  it("flushes a trailing frame without a final blank line", async () => {
    const frames = await collect(streamOf(["event: done\n", "data: {}"]));
    expect(frames).toEqual([{ event: "done", data: "{}" }]);
  });
});
