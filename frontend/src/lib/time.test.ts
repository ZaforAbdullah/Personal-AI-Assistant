import { describe, expect, it } from "vitest";

import { formatRelative } from "./time";

const now = new Date("2026-09-17T12:00:00Z");
const ago = (ms: number) => new Date(now.getTime() - ms).toISOString();

describe("formatRelative", () => {
  it("buckets by the largest sensible unit", () => {
    expect(formatRelative(ago(5_000), now)).toBe("just now");
    expect(formatRelative(ago(3 * 60_000), now)).toBe("3m");
    expect(formatRelative(ago(5 * 3_600_000), now)).toBe("5h");
    expect(formatRelative(ago(2 * 86_400_000), now)).toBe("2d");
  });

  it("falls back to a date after a week and to empty for junk", () => {
    expect(formatRelative(ago(30 * 86_400_000), now)).toMatch(/Aug/);
    expect(formatRelative("not a date", now)).toBe("");
  });
});
