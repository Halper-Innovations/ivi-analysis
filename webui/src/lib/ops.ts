/**
 * Pure helpers for the Events and Ops decks — status glyph maps (shape +
 * color, never color alone), byte/age formatting, and the weekly cost
 * chart layout. Everything here is vitest-covered; components stay thin.
 */

import type { CostWeek, EventCard } from "./api";

export interface StatusGlyph {
  glyph: string;
  tone: "jade" | "amber" | "rose" | "slate";
  label: string;
}

/** Heartbeat cell: shape + tone + word — readable without color. */
export function heartbeatGlyph(status: string): StatusGlyph {
  switch (status) {
    case "COMPLETE":
      return { glyph: "●", tone: "jade", label: "complete" };
    case "FAILED":
      return { glyph: "✕", tone: "rose", label: "failed" };
    case "STARTED":
      return { glyph: "◐", tone: "amber", label: "started, never completed" };
    default:
      return { glyph: "·", tone: "slate", label: "no run recorded" };
  }
}

/** Scan-strip cell tone: OK is jade; a missing business day is dark water;
 *  anything else recorded is a failure. */
export function scanDayGlyph(status: string): StatusGlyph {
  if (status === "OK") return { glyph: "●", tone: "jade", label: "scanned" };
  if (status === "MISSING") return { glyph: "·", tone: "slate", label: "no scan — a hole in the record" };
  return { glyph: "✕", tone: "rose", label: status };
}

export function formatBytes(size: number): string {
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
  return `${(size / (1024 * 1024)).toFixed(1)} MB`;
}

export function formatAgeDays(days: number | null): string {
  if (days === null) return "—";
  if (days === 0) return "today";
  return `${days}d`;
}

/** Small detail_json dicts spoken as "key value" fragments, capped. */
export function detailHighlights(
  detail: Record<string, unknown>,
  max = 3,
): string[] {
  const out: string[] = [];
  for (const [key, value] of Object.entries(detail)) {
    if (out.length >= max) break;
    if (value === null || value === undefined || value === "") continue;
    const rendered = Array.isArray(value)
      ? value.slice(0, 4).join(", ")
      : typeof value === "object"
        ? null // nested blobs stay in the tooltip, not the card
        : String(value);
    if (rendered === null || rendered === "") continue;
    out.push(`${key.replace(/_/g, " ")}: ${rendered}`);
  }
  return out;
}

/** Search text for the client-side event filter (server q covers ticker +
 *  company; this covers what's already on screen). */
export function eventSearchText(card: EventCard): string {
  return `${card.ticker ?? ""} ${card.company_name ?? ""} ${card.event_type}`.toLowerCase();
}

export interface CostChartRow {
  week: string;
  scan: number;
  research: number;
  total: number;
  /** 0..1 of the tallest week, for bar heights. */
  scanFrac: number;
  researchFrac: number;
}

/** Newest-first API weeks → chronological chart rows scaled to the max.
 *  Weeks with zero recorded spend keep their slot — a quiet week is data. */
export function costChartRows(weeks: CostWeek[], limit = 16): CostChartRow[] {
  const recent = weeks.slice(0, limit).slice().reverse();
  const max = Math.max(...recent.map((week) => week.total_usd), 0);
  return recent.map((week) => ({
    week: week.week,
    scan: week.scan_cost_usd,
    research: week.research_cost_usd,
    total: week.total_usd,
    scanFrac: max > 0 ? week.scan_cost_usd / max : 0,
    researchFrac: max > 0 ? week.research_cost_usd / max : 0,
  }));
}

/** "2026-W12" → "W12" (the year lives in the tooltip). */
export function shortWeek(week: string): string {
  const idx = week.indexOf("-W");
  return idx === -1 ? week : week.slice(idx + 1);
}
