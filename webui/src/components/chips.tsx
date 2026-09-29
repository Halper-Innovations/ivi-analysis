/**
 * Chips: the two boxed voices the UI allows itself.
 *
 * StatusChip — state-machine values (watchlist status, gate verdicts,
 * health states) in mono caps: the one place uppercase survives.
 * CopyChip — the read-only bridge to the CLI.
 *
 * Everything else that used to be a chip is now typography: conviction is
 * a colored word (GradeWord), event flags are humanized prose. Identity is
 * never color-alone — every treatment carries its text.
 */

import { useState } from "react";

import { humanizeToken } from "../lib/floor";
import { glossaryDefinition } from "../lib/glossary";

const STATUS_TONES: Record<string, string> = {
  BUY_CONFIRMED: "text-jade border-jade/20 border-l-jade bg-jade/5",
  DEPLOY_READY: "text-jade border-jade/20 border-l-jade bg-jade/5",
  EVENT_PENDING: "text-amber border-amber/20 border-l-amber bg-amber/5",
  ACTIVE: "text-sounding border-glass-tint/15 border-l-glass-tint bg-glass-tint/5",
  UNCERTAIN: "text-amber border-amber/20 border-l-amber bg-amber/5",
  PRICE_DATA_SUSPECT: "text-rose border-rose/20 border-l-rose bg-rose/5",
  QUARANTINE: "text-slate-mist border-slate-mist/20 border-l-slate-mist bg-slate-mist/5",
  REMOVED: "text-slate-mist border-slate-mist/20 border-l-slate-mist bg-slate-mist/5",
  GREEN: "text-jade border-jade/20 border-l-jade bg-jade/5",
  AMBER: "text-amber border-amber/20 border-l-amber bg-amber/5",
  RED: "text-rose border-rose/20 border-l-rose bg-rose/5",
  PROCEED: "text-jade border-jade/20 border-l-jade bg-jade/5",
  ADJUST: "text-amber border-amber/20 border-l-amber bg-amber/5",
  BLOCK: "text-rose border-rose/20 border-l-rose bg-rose/5",
};

export function StatusChip({ value, title }: { value: string | null; title?: string }) {
  if (!value) return null;
  const tone =
    STATUS_TONES[value] ??
    "text-sounding border-glass-tint/15 border-l-glass-tint bg-glass-tint/5";
  return (
    <span
      title={title ?? value}
      className={`font-data inline-block whitespace-nowrap rounded-[2px] border border-l-2 px-2 py-0.5 text-[10px] uppercase tracking-[0.08em] ${tone}`}
    >
      {value.replaceAll("_", " ")}
    </span>
  );
}

export function Term({ token, label }: { token: string; label: string }) {
  const definition = glossaryDefinition(token);
  if (!definition) return <>{label}</>;
  return (
    <span
      title={definition}
      aria-label={`${label}: ${definition}`}
      className="underline decoration-dotted underline-offset-2"
    >
      {label}
      <sup className="ml-0.5 text-[9px]" aria-hidden="true">
        ⓘ
      </sup>
    </span>
  );
}

const GRADE_TONES: Record<string, string> = {
  ACTIONABLE: "text-jade",
  WATCHLIST_ONLY: "text-sounding",
  DATA_INCOMPLETE: "text-amber",
  AVOID: "text-rose",
};

/** Conviction grade as a quiet colored word — not a chip. The raw value
 *  stays in the tooltip. */
export function GradeWord({ value }: { value: string | null }) {
  if (!value) return null;
  const tone = GRADE_TONES[value] ?? "text-sounding";
  return (
    <span
      title={value}
      className={`whitespace-nowrap text-xs lowercase tracking-wide ${tone}`}
    >
      {humanizeToken(value)}
    </span>
  );
}

export function CopyChip({
  command,
  label,
  disabled,
  disabledHint,
}: {
  command: string;
  label?: string;
  disabled?: boolean;
  disabledHint?: string;
}) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      type="button"
      disabled={disabled}
      title={disabled ? disabledHint : command}
      onClick={() => {
        void navigator.clipboard.writeText(command).then(() => {
          setCopied(true);
          window.setTimeout(() => setCopied(false), 1600);
        });
      }}
      className={`font-data max-w-full truncate rounded-[2px] border px-2.5 py-1.5 text-left text-[11px] transition-colors ${
        disabled
          ? "cursor-not-allowed border-glass-tint/10 text-sounding/40"
          : copied
            ? "border-jade/50 text-jade"
            : "border-glass-tint/25 text-glass-tint hover:border-glass-tint/50 hover:text-moonlight"
      }`}
    >
      {copied ? "copied ✓" : (label ?? command)}
    </button>
  );
}
