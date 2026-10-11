import type { SVGProps } from "react";

type P = SVGProps<SVGSVGElement>;
const base = (p: P) => ({
  width: 20,
  height: 20,
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.8,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
  "aria-hidden": true,
  ...p,
});

export const PlusIcon = (p: P) => (
  <svg {...base(p)}>
    <path d="M12 5v14M5 12h14" />
  </svg>
);
export const MenuIcon = (p: P) => (
  <svg {...base(p)}>
    <path d="M4 7h16M4 12h16M4 17h16" />
  </svg>
);
export const CloseIcon = (p: P) => (
  <svg {...base(p)}>
    <path d="M6 6l12 12M18 6L6 18" />
  </svg>
);
export const TrashIcon = (p: P) => (
  <svg {...base(p)}>
    <path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3" />
  </svg>
);
export const PencilIcon = (p: P) => (
  <svg {...base(p)}>
    <path d="M4 20h4l10.5-10.5a2.1 2.1 0 0 0-3-3L5 17v3zM13.5 6.5l3 3" />
  </svg>
);
export const CheckIcon = (p: P) => (
  <svg {...base(p)}>
    <path d="M5 12.5l4.5 4.5L19 7.5" />
  </svg>
);
export const SendIcon = (p: P) => (
  <svg {...base(p)}>
    <path d="M12 19V5M5 12l7-7 7 7" />
  </svg>
);
export const StopIcon = (p: P) => (
  <svg {...base(p)} fill="currentColor" stroke="none">
    <rect x="7" y="7" width="10" height="10" rx="2" />
  </svg>
);
export const CalcIcon = (p: P) => (
  <svg {...base(p)}>
    <rect x="5" y="3" width="14" height="18" rx="2" />
    <path d="M8 7h8M8 12h2M12 12h2M16 12h0M8 16h2M12 16h2M16 16h0" />
  </svg>
);
export const SearchIcon = (p: P) => (
  <svg {...base(p)}>
    <circle cx="11" cy="11" r="6" />
    <path d="M20 20l-4.5-4.5" />
  </svg>
);
export const CloudIcon = (p: P) => (
  <svg {...base(p)}>
    <path d="M7 18a4 4 0 0 1-.6-7.95A6 6 0 0 1 18 9.5 3.5 3.5 0 0 1 17.5 18H7z" />
  </svg>
);
export const ImageIcon = (p: P) => (
  <svg {...base(p)}>
    <rect x="3" y="5" width="18" height="14" rx="2" />
    <circle cx="9" cy="10" r="1.5" />
    <path d="M21 16l-5-5-8 8" />
  </svg>
);
export const ToolIcon = (p: P) => (
  <svg {...base(p)}>
    <path d="M14.5 5.5a4 4 0 0 0 5 5L9 21H4v-5L14.5 5.5z" />
  </svg>
);
export const ChatIcon = (p: P) => (
  <svg {...base(p)}>
    <path d="M4 6a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H9l-5 4V6z" />
  </svg>
);

/** The assistant mark: a soft gradient disc in the brand blue, used as the assistant avatar. */
export function AssistantMark({ size = 28 }: { size?: number }) {
  return (
    <span
      aria-hidden
      className="inline-flex shrink-0 items-center justify-center rounded-full text-white"
      style={{
        width: size,
        height: size,
        background: "linear-gradient(135deg, #6f8afb 0%, #4060f8 55%, #3049d6 100%)",
        boxShadow: "0 2px 6px rgb(64 96 248 / 0.35)",
        fontSize: Math.round(size * 0.5),
        fontWeight: 700,
        letterSpacing: "-0.02em",
      }}
    >
      L
    </span>
  );
}

export function Wordmark() {
  return (
    <span className="flex items-center gap-2">
      <AssistantMark size={30} />
      <span className="text-[17px] font-bold leading-tight tracking-tight text-brand-600" style={{ letterSpacing: "-0.03em" }}>
        Personal AI Assistant
      </span>
    </span>
  );
}
