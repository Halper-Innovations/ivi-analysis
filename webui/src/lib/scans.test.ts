import { describe, expect, it } from "vitest";

import type { RunFunnelStage, RunSummary, SweepGroup } from "./api";
import {
  cellFor,
  coverageStrength,
  distinctRunValues,
  fillCommand,
  filterRuns,
  formatUsd,
  funnelLayout,
  laneBudgetFrac,
  runHistoryLabel,
  stageTone,
  verdictTone,
  weekCaption,
} from "./scans";

const stage = (key: string, count: number, tickers: string[] = []): RunFunnelStage => ({
  key,
  label: key.toLowerCase(),
  count,
  tickers,
});

describe("funnelLayout", () => {
  it("scales widths to the widest stage", () => {
    const bars = funnelLayout([stage("LOADED", 40), stage("EXAMINED", 10), stage("SELECTED", 1)]);
    // 1/40 = 0.025 clamps to the 0.04 hairline floor.
    expect(bars.map((b) => b.frac)).toEqual([1, 0.25, 0.04]);
  });

  it("keeps zero-count stages at zero and never divides by zero", () => {
    const bars = funnelLayout([stage("SELECTED", 0)]);
    expect(bars).toEqual([{ ...stage("SELECTED", 0), frac: 0 }]);
  });
});

describe("tones", () => {
  it("maps verdicts to reserved status tones", () => {
    expect(verdictTone("SELECTED")).toBe("jade");
    expect(verdictTone("WATCHLIST")).toBe("amber");
    expect(verdictTone("NO_SELECTION")).toBe("slate");
    expect(verdictTone(null)).toBe("slate");
    expect(verdictTone("FAILED")).toBe("rose");
  });

  it("maps v2 terminal states to reserved status tones", () => {
    expect(stageTone("UNDERWRITTEN")).toBe("jade");
    expect(stageTone("READY_FOR_UNDERWRITING")).toBe("jade");
    expect(stageTone("NEEDS_DATA")).toBe("amber");
    expect(stageTone("SCREENED_OUT")).toBe("rose");
    expect(stageTone("OUT_OF_SCOPE")).toBe("slate");
  });

  it("keeps v1 process stages neutral water", () => {
    expect(stageTone("LOADED")).toBe("water");
    expect(stageTone("EXAMINED")).toBe("water");
    expect(stageTone("RANKED")).toBe("water");
    expect(stageTone("SELECTED")).toBe("jade");
  });
});

describe("formatUsd", () => {
  it("formats the observed cost magnitudes", () => {
    expect(formatUsd(null)).toBe("—");
    expect(formatUsd(0)).toBe("$0");
    expect(formatUsd(0.0042)).toBe("$0.0042");
    expect(formatUsd(1.234567)).toBe("$1.23");
    expect(formatUsd(17.4)).toBe("$17.40");
  });
});

describe("fillCommand", () => {
  it("composes the exact CLI the platform accepts", () => {
    expect(fillCommand("healthcare_pharma", "small_cap")).toBe(
      "ivi autonomous-sector-run --sector healthcare_pharma --market-cap-focus small_cap",
    );
  });
});

describe("coverageStrength", () => {
  it("is full inside the fresh window and floors in dark water", () => {
    expect(coverageStrength(0)).toBe(1);
    expect(coverageStrength(30)).toBe(1);
    expect(coverageStrength(120)).toBe(0.18);
    expect(coverageStrength(400)).toBe(0.18);
    expect(coverageStrength(null)).toBe(0);
  });

  it("fades linearly between the fresh and dark boundaries", () => {
    expect(coverageStrength(75)).toBeCloseTo(0.59, 10);
  });
});

const run = (overrides: Partial<RunSummary>): RunSummary =>
  ({
    run_id: "r",
    slug: "s/r",
    path: "/p",
    kind: "autonomous_sector",
    contract_version: null,
    pipeline_version: "v1",
    sector: "biotech",
    market_cap_focus: "mid_cap",
    scan_family: "normal",
    as_of_date: null,
    created_at: null,
    completed_at: null,
    status: null,
    execution_status: null,
    decision_status: null,
    final_verdict: "NO_SELECTION",
    selected_ticker: null,
    no_selection_reason: null,
    examined_count: null,
    disposition_counts_json: null,
    cost_microdollars: null,
    cost_usd: null,
    report_path: null,
    parse_error: null,
    integrity_status: "PASS",
    decision_eligible: true,
    indexed_at: "now",
    ...overrides,
  }) as RunSummary;

describe("filterRuns", () => {
  const runs = [
    run({ run_id: "a", sector: "biotech", final_verdict: "SELECTED", selected_ticker: "AAA" }),
    run({ run_id: "b", sector: "utilities", market_cap_focus: "micro_cap" }),
    run({ run_id: "c", pipeline_version: null, final_verdict: null }),
    run({
      run_id: "history",
      integrity_status: "INVALID",
      decision_eligible: false,
      final_verdict: null,
    }),
  ];

  it("filters by facets with v1/UNKNOWN defaults", () => {
    expect(filterRuns(runs, { sector: "utilities" }).map((r) => r.run_id)).toEqual(["b"]);
    expect(filterRuns(runs, { pipeline: "v1" }).map((r) => r.run_id)).toEqual(["a", "b", "c"]);
    expect(filterRuns(runs, { verdict: "UNKNOWN" }).map((r) => r.run_id)).toEqual(["c"]);
    expect(filterRuns(runs, { history: "all" }).map((r) => r.run_id)).toEqual([
      "a",
      "b",
      "c",
      "history",
    ]);
  });

  it("free-text matches run id, sector, and selected ticker", () => {
    expect(filterRuns(runs, { q: "aaa" }).map((r) => r.run_id)).toEqual(["a"]);
    expect(filterRuns(runs, { q: "utilit" }).map((r) => r.run_id)).toEqual(["b"]);
  });

  it("lists distinct facet values sorted", () => {
    expect(distinctRunValues(runs, "sector")).toEqual(["biotech", "utilities"]);
    expect(distinctRunValues(runs, "market_cap_focus")).toEqual(["micro_cap", "mid_cap"]);
  });

  it("labels excluded history without surfacing a decision", () => {
    expect(runHistoryLabel(runs[0])).toBeNull();
    expect(runHistoryLabel(runs[3])).toBe("excluded history · invalid");
    expect(runHistoryLabel(run({
      decision_eligible: false,
      integrity_status: "UNAUDITED",
      parse_error: "JSONDecodeError: boom",
    }))).toBe("excluded history · unreadable artifact");
  });
});

describe("cellFor and laneBudgetFrac", () => {
  it("finds a cell by axes", () => {
    const cells = [
      {
        sector: "biotech",
        band: "mid_cap",
        tickers: 3,
        last_loaded_at: null,
        age_days: null,
        run_count: 1,
        pipelines: ["v1"],
      },
    ];
    expect(cellFor(cells, "biotech", "mid_cap")?.tickers).toBe(3);
    expect(cellFor(cells, "biotech", "micro_cap")).toBeUndefined();
  });

  it("caps lane spend fraction at 1 and rejects unknowable caps", () => {
    expect(laneBudgetFrac(500000, 1000000)).toBe(0.5);
    expect(laneBudgetFrac(2000000, 1000000)).toBe(1);
    expect(laneBudgetFrac(null, 1000000)).toBeNull();
    expect(laneBudgetFrac(500000, null)).toBeNull();
  });
});

describe("weekCaption", () => {
  it("renders the week span compactly", () => {
    const group = {
      week: "2026-W24",
      week_start: "2026-06-08",
      week_end: "2026-06-14",
    } as SweepGroup;
    expect(weekCaption(group)).toBe("Jun 8 – Jun 14");
  });
});
