/**
 * Watchlist-floor logic — pure functions, no DOM, unit-tested.
 *
 * Everything the watchlist and Today views compute client-side lives here:
 * age math for the frost contract, GaugeInput assembly from API rows,
 * board grouping, leaderboard sorting, and the filter predicate.
 */

import type { WatchlistRow, WaterlineItem } from "./api";
import type { GaugeInput } from "./gauge";

/** Hours since an ISO timestamp; null when absent or unparseable. */
export function hoursSince(iso: string | null, now: Date): number | null {
  if (!iso) return null;
  const then = new Date(iso.replace(" ", "T"));
  const ms = now.getTime() - then.getTime();
  if (Number.isNaN(ms)) return null;
  return Math.round((ms / 3_600_000) * 100) / 100;
}

/** Compact age label: 3h, 26h, 4.2d — the defrost hover's exact-age text. */
export function formatAge(hours: number | null): string {
  if (hours === null) return "age unknown";
  if (hours < 48) return `${Math.round(hours)}h`;
  return `${Math.round((hours / 24) * 10) / 10}d`;
}

export function formatMoneyCompact(value: number | null): string {
  if (value === null) return "—";
  const sign = value < 0 ? "-" : "";
  const abs = Math.abs(value);
  const digits = abs >= 1000 ? 0 : 2;
  return `${sign}$${abs.toLocaleString("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  })}`;
}

/** Millions → $1.9B / $311M / $4.5M (market cap; net debt can go negative). */
export function formatMillions(mm: number | null): string {
  if (mm === null) return "—";
  const sign = mm < 0 ? "-" : "";
  const abs = Math.abs(mm);
  if (abs >= 1000) return `${sign}$${(abs / 1000).toFixed(1)}B`;
  if (abs >= 10) return `${sign}$${Math.round(abs)}M`;
  return `${sign}$${abs.toFixed(1)}M`;
}

/** Raw units (quarterly fact values are plain dollars / share counts;
 *  valuation extras mix ratios in). */
export function formatRawCompact(value: number | null, money = true): string {
  if (value === null) return "—";
  const sign = value < 0 ? "-" : "";
  const prefix = money ? `${sign}$` : sign;
  const abs = Math.abs(value);
  if (abs >= 1e9) return `${prefix}${(abs / 1e9).toFixed(2)}B`;
  if (abs >= 1e6) return `${prefix}${(abs / 1e6).toFixed(1)}M`;
  if (abs >= 1e3) return `${prefix}${(abs / 1e3).toFixed(1)}k`;
  if (money) return `${prefix}${abs.toFixed(2)}`;
  if (Number.isInteger(value)) return String(value);
  // Ratios keep their signal: 0.0013 must not render as "0".
  return abs < 1 ? `${sign}${abs.toPrecision(3)}` : `${sign}${abs.toFixed(2)}`;
}

export function formatPct(value: number | null, digits = 1): string {
  if (value === null) return "—";
  return `${value >= 0 ? "+" : ""}${value.toFixed(digits)}%`;
}

/* ---------------------------------------------------------- humanizing --
   Raw platform codes (EVENT_PENDING:UNREVIEWED_8K, payments_fintech,
   SECULAR_DECLINE_HEADWIND) are state-machine values, not prose. The UI
   speaks them in lowercase; the raw value stays reachable via tooltips.
   Mono-uppercase is reserved for status chips and CLI commands. */

/** Tokens that keep their true case when humanized. */
const TOKEN_CASE: Record<string, string> = {
  "8k": "8-K",
  "10k": "10-K",
  "10q": "10-Q",
  ch11: "Chapter 11",
  ipo: "IPO",
  dcf: "DCF",
  epv: "EPV",
  ncav: "NCAV",
  fy: "FY",
  ttm: "TTM",
  yoy: "YoY",
  wacc: "WACC",
  mos: "MoS",
  adv: "ADV",
  sec: "SEC",
  llm: "LLM",
  etf: "ETF",
  reit: "REIT",
  spac: "SPAC",
  us: "US",
};

/** SNAKE_OR_MIXED_case → "snake or mixed case" (acronyms keep their case). */
export function humanizeToken(raw: string | null): string {
  if (!raw) return "";
  return raw
    .split(/[_\s]+/)
    .filter(Boolean)
    .map((word) => TOKEN_CASE[word.toLowerCase()] ?? word.toLowerCase())
    .join(" ");
}

/** Headwind/support codes drop their suffix: SECULAR_DECLINE_HEADWIND →
 *  "secular decline". */
export function humanizeSignalCode(raw: string): string {
  return humanizeToken(raw.replace(/_(HEADWIND|SUPPORT)$/i, ""));
}

/** The event_pending column carries comma-joined flags, each prefixed
 *  EVENT_PENDING:. → "material agreement · unreviewed 8-K" (deduped). */
export function humanizeEventPending(raw: string | null): string {
  if (!raw) return "";
  const seen = new Set<string>();
  const labels: string[] = [];
  for (const part of raw.split(",")) {
    const label = humanizeToken(part.trim().replace(/^EVENT_PENDING:/i, ""));
    if (label && !seen.has(label)) {
      seen.add(label);
      labels.push(label);
    }
  }
  return labels.join(" · ");
}

/** Margin of safety at the current price against the valuation anchor. */
export function marginOfSafetyPct(
  price: number | null,
  anchorValue: number | null,
): number | null {
  if (price === null || anchorValue === null || anchorValue <= 0) return null;
  return ((anchorValue - price) / anchorValue) * 100;
}

/** Statuses where an in-zone name is clear to act rather than held. Jade
 *  distance/price treatments key off this — a held name in the zone stays
 *  neutral; its status chip carries the hold. */
export const CLEAR_STATUSES = new Set(["DEPLOY_READY", "BUY_CONFIRMED"]);

export function clearToAct(status: string | null | undefined): boolean {
  return CLEAR_STATUSES.has(status ?? "");
}

/** GaugeInput for a queue row's micro gauge: the anchor shelf, the target,
 *  the price, and whatever wake the caller has. */
export function rowGaugeInput(row: WatchlistRow, now: Date, wake?: number[]): GaugeInput {
  return {
    ticker: row.ticker,
    shelves:
      row.valuation_anchor_value !== null
        ? [
            {
              method: "anchor",
              label: row.valuation_anchor_method ?? "anchor",
              value: row.valuation_anchor_value,
              emphasized: true,
            },
          ]
        : [],
    buyTarget: row.buy_price_target,
    price: row.latest_price,
    wake: wake ?? [],
    priceAgeHours: hoursSince(row.latest_price_checked_at, now),
    priceSuspect: row.status === "PRICE_DATA_SUSPECT",
  };
}

/** GaugeInput for a waterline strip item (server supplies age + wake). */
export function waterlineGaugeInput(item: WaterlineItem): GaugeInput {
  return {
    ticker: item.ticker,
    shelves:
      item.valuation_anchor_value !== null
        ? [
            {
              method: "anchor",
              label: item.valuation_anchor_method ?? "anchor",
              value: item.valuation_anchor_value,
              emphasized: true,
            },
          ]
        : [],
    buyTarget: item.buy_price_target,
    price: item.latest_price,
    wake: item.wake,
    priceAgeHours: item.price_age_hours,
    priceSuspect: item.price_suspect,
  };
}

/** Board columns in the platform's own priority order. */
export const BOARD_COLUMNS = [
  "BUY_CONFIRMED",
  "DEPLOY_READY",
  "EVENT_PENDING",
  "ACTIVE",
  "UNCERTAIN",
  "PRICE_DATA_SUSPECT",
  "QUARANTINE",
] as const;

export function groupForBoard(rows: WatchlistRow[]): Map<string, WatchlistRow[]> {
  const groups = new Map<string, WatchlistRow[]>();
  for (const column of BOARD_COLUMNS) groups.set(column, []);
  for (const row of rows) {
    const key = row.presented_status ?? "UNKNOWN";
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key)!.push(row);
  }
  for (const [key, group] of groups) {
    if (group.length === 0 && !BOARD_COLUMNS.includes(key as never)) groups.delete(key);
  }
  return groups;
}

/** Statuses with no place in the race to the zone: the gate ate the
 *  quarantined rows, and a suspect price is not a real distance — the same
 *  eligibility rule the waterline read model applies. */
const LADDER_EXCLUDED = new Set(["QUARANTINE", "PRICE_DATA_SUSPECT"]);

/** Distance-to-zone leaderboard: rows with a real distance, closest first. */
export function leaderboardRows(rows: WatchlistRow[]): WatchlistRow[] {
  return rows
    .filter(
      (row) =>
        row.distance_from_buy_pct !== null &&
        !LADDER_EXCLUDED.has(row.presented_status ?? ""),
    )
    .sort((a, b) => a.distance_from_buy_pct! - b.distance_from_buy_pct!);
}

export interface FloorFilters {
  q?: string;
  band?: string;
  grade?: string;
  family?: string;
  status?: string;
}

export function filterRows(rows: WatchlistRow[], filters: FloorFilters): WatchlistRow[] {
  const q = (filters.q ?? "").trim().toUpperCase();
  return rows.filter((row) => {
    if (q && !row.ticker.toUpperCase().includes(q)) return false;
    if (filters.band && row.cap_band_label !== filters.band) return false;
    if (filters.grade && row.conviction_grade !== filters.grade) return false;
    if (filters.family && row.scan_family !== filters.family) return false;
    if (filters.status && row.presented_status !== filters.status) return false;
    return true;
  });
}

export function distinctValues(
  rows: WatchlistRow[],
  key: "cap_band_label" | "conviction_grade" | "scan_family" | "presented_status",
): string[] {
  const values = new Set<string>();
  for (const row of rows) {
    const value = row[key];
    if (value) values.add(value);
  }
  return [...values].sort();
}
