import { describe, expect, it } from "vitest";

import { delatex } from "./markdown";

describe("delatex", () => {
  it("leaves plain text untouched", () => {
    expect(delatex("The answer is 1045.")).toBe("The answer is 1045.");
  });

  it("rewrites the constructs a model reaches for", () => {
    expect(delatex("The result of $2^{10} + \\frac{7!}{5!2!}$ is 1045.")).toBe(
      "The result of 2^10 + 7!/5!2! is 1045.",
    );
    expect(delatex("so $17 \\times 23 + 5 = 396$")).toBe("so 17 × 23 + 5 = 396");
    expect(delatex("\\frac{a + b}{2}")).toBe("(a + b)/2");
    expect(delatex("$$\\sqrt{1444 + 225} \\approx 40.85$$")).toBe("√(1444 + 225) ≈ 40.85");
    expect(delatex("\\( x_{i} \\le y \\)")).toBe(" x_i ≤ y ");
  });

  it("does not eat dollar amounts", () => {
    expect(delatex("costs $5 and $10")).toBe("costs $5 and $10");
    expect(delatex("between $5 and $ 10")).toBe("between $5 and $ 10");
  });
});
