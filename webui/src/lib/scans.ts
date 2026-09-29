/**
 * Pure logic for the Scans gallery, run detail, and Coverage atlas.
 * Same contract as floor.ts: everything here is renderer-free and
 * vitest-covered; components stay thin.
 */

import type { CoverageCell, RunFunnelStage, RunSummary, SweepGroup } from "./api";

/** Verdict → reserved status tone. Shape/text always accompanies color. */
export function verdictTone(verdict: string | null): "jade" | "amber" | "rose" | "slate" {
  switch (verdict) {
    case "SELECTED":
      return "jade";
    case "WATCHLIST":
      return "amber";
    case "NO_SELECTION":
      return "slate";
    default:
      return verdict ? "rose" : "slate";
  }
}

/**
 * Funnel stage → tone. v2 terminal states are genuine platform states and
 * wear the reserved status colors; v1 process stages (loaded/examined/
 * ranked) are neutral water — only the selection outcome earns jade.
 */
export function stageTone(key: string): "jade" | "amber" | "rose" | "slate" | "water" {
  switch (key) {
    case "UNDERWRITTEN":
    case "READY_FOR_UNDERWRITING":
    case "SELECTED":
      return "jade";
    case "NEEDS_DATA":
      return "amber";
    case "SCREENED_OUT":
      return "rose";
    case "LOADED":
    case "EXAMINED":
    case "RANKED":
      return "water";
    default:
      return "slate";
  }
}

export interface FunnelBar {
  key: string;
  label: string;
  count: number;
  tickers: string[];
  /** Width fraction of the widest stage; 0-count stages keep a hairline. */
  frac: number;
}

export function funnelLayout(stages: RunFunnelStage[]): FunnelBar[] {
  const max = Math.max(1, ...stages.map((s) => s.count));
  return stages.map((stage) => ({
    ...stage,
    frac: stage.count === 0 ? 0 : Math.max(0.04, stage.count / max),
  }));
}

/** Cost in dollars for display; sub-cent runs keep enough precision to read. */
export function formatUsd(value: number | null): string {
  if (value === null || !Number.isFinite(value)) return "—";
  if (value === 0) return "$0";
  if (value < 0.01) return `$${value.toFixed(4)}`;
  return `$${value.toFixed(2)}`;
}

/** The exact CLI to fill (or refresh) a coverage cell. */
export function fillCommand(sector: string, band: string): string {
  return `ivi autonomous-sector-run --sector ${sector} --market-cap-focus ${band}`;
}

/**
 * Recency → cell strength for the atlas, on 0..1.
 * Fresh coverage glows; anything past a quarter fades toward dark water.
 */
export const COVERAGE_FRESH_DAYS = 30;
export const COVERAGE_DARK_DAYS = 120;

export function coverageStrength(ageDays: number | null): number {
  if (ageDays === null || !Number.isFinite(ageDays)) return 0;
  if (ageDays <= COVERAGE_FRESH_DAYS) return 1;
  if (ageDays >= COVERAGE_DARK_DAYS) return 0.18;
  const span = COVERAGE_DARK_DAYS - COVERAGE_FRESH_DAYS;
  return 1 - (0.82 * (ageDays - COVERAGE_FRESH_DAYS)) / span;
}

export function cellFor(
  cells: CoverageCell[],
  sector: string,
  band: string,
): CoverageCell | undefined {
  return cells.find((cell) => cell.sector === sector && cell.band === band);
}

export interface RunFilters {
  q?: string;
  sector?: string;
  band?: string;
  verdict?: string;
  pipeline?: string;
  history?: "all";
}

export function filterRuns(runs: RunSummary[], filters: RunFilters): RunSummary[] {
  const q = (filters.q ?? "").trim().toLowerCase();
  return runs.filter((run) => {
    if (filters.history !== "all" && !run.decision_eligible) return false;
    if (filters.sector && run.sector !== filters.sector) return false;
    if (filters.band && run.market_cap_focus !== filters.band) return false;
    if (filters.verdict && (run.final_verdict ?? "UNKNOWN") !== filters.verdict) return false;
    if (filters.pipeline && (run.pipeline_version ?? "v1") !== filters.pipeline) return false;
    if (
      q &&
      !`${run.run_id} ${run.sector ?? ""} ${run.selected_ticker ?? ""}`
        .toLowerCase()
        .includes(q)
    )
      return false;
    return true;
  });
}

export function runHistoryLabel(run: RunSummary): string | null {
  if (run.decision_eligible) return null;
  if (run.parse_error) return "excluded history · unreadable artifact";
  return `excluded history · ${run.integrity_status.toLowerCase().replaceAll("_", " ")}`;
}

export function distinctRunValues(
  runs: RunSummary[],
  key: "sector" | "market_cap_focus" | "final_verdict" | "pipeline_version",
): string[] {
  const values = new Set<string>();
  for (const run of runs) {
    const value = run[key];
    if (value) values.add(value);
  }
  return [...values].sort();
}

/** "2026-W28 · Jun 8 – Jun 14" without pulling in a date library. */
export function weekCaption(group: SweepGroup): string {
  const day = (iso: string) => {
    const date = new Date(`${iso}T00:00:00Z`);
    return date.toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      timeZone: "UTC",
    });
  };
  return `${day(group.week_start)} – ${day(group.week_end)}`;
}

/** Lane spend as a fraction of its cap, when both sides are known. */
export function laneBudgetFrac(
  cost: number | null,
  cap: number | null,
): number | null {
  if (cost === null || cap === null || cap <= 0) return null;
  return Math.min(1, cost / cap);
}
