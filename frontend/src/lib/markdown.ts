/** Models answer maths in LaTeX even when told not to. There is no math renderer here, so the
 * common constructs are rewritten to plain notation; anything else just loses its `$` fences. */
export function delatex(text: string): string {
  if (!text.includes("\\") && !text.includes("$")) return text;
  let out = text;
  // \frac{a}{b} -> a/b   (parenthesise multi-token parts)
  out = out.replace(/\\(?:d|t)?frac\{([^{}]*)\}\{([^{}]*)\}/g, (_, a: string, b: string) => {
    const wrap = (s: string) => (/[+\-*/ ]/.test(s.trim()) ? `(${s.trim()})` : s.trim());
    return `${wrap(a)}/${wrap(b)}`;
  });
  out = out.replace(/\\sqrt\{([^{}]*)\}/g, "√($1)");
  out = out.replace(/\\text\{([^{}]*)\}/g, "$1");
  out = out.replace(/\\(?:times|cdot)\b/g, "×");
  out = out.replace(/\\div\b/g, "÷");
  out = out.replace(/\\pm\b/g, "±");
  out = out.replace(/\\(?:le|leq)\b/g, "≤").replace(/\\(?:ge|geq)\b/g, "≥").replace(/\\neq?\b/g, "≠");
  out = out.replace(/\\approx\b/g, "≈").replace(/\\infty\b/g, "∞").replace(/\\pi\b/g, "π");
  out = out.replace(/\\(?:left|right)\b/g, "");
  // x^{10} -> x^10, x_{i} -> x_i ; keep single-char scripts as they are
  out = out.replace(/\^\{([^{}]*)\}/g, "^$1").replace(/_\{([^{}]*)\}/g, "_$1");
  // math fences: $$...$$, \[...\], \(...\), $...$
  out = out.replace(/\$\$([\s\S]*?)\$\$/g, "$1");
  out = out.replace(/\\\[([\s\S]*?)\\\]/g, "$1").replace(/\\\(([\s\S]*?)\\\)/g, "$1");
  out = out.replace(/(^|[^\\$])\$(?!\s)([^$\n]+?)(?<!\s)\$(?!\d)/g, "$1$2");
  return out;
}
