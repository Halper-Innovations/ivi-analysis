import { describe, expect, it } from "vitest";

import type { CostWeek, EventCard } from "./api";
import {
  costChartRows,
  detailHighlights,
  eventSearchText,
  formatAgeDays,
  formatBytes,
  heartbeatGlyph,
  scanDayGlyph,
  shortWeek,
} from "./ops";

describe("status glyphs", () => {
  it("maps heartbeat statuses to shape + tone", () => {
    expect(heartbeatGlyph("COMPLETE")).toEqual({ glyph: "●", tone: "jade", label: "complete" });
    expect(heartbeatGlyph("FAILED")).toEqual({ glyph: "✕", tone: "rose", label: "failed" });
    expect(heartbeatGlyph("STARTED").tone).toBe("amber");
    expect(heartbeatGlyph("MISSING")).toEqual({
      glyph: "·",
      tone: "slate",
      label: "no run recorded",
    });
  });

  it("maps scan-day statuses; unknown recorded statuses read as failure", () => {
    expect(scanDayGlyph("OK").tone).toBe("jade");
    expect(scanDayGlyph("MISSING").tone).toBe("slate");
    expect(scanDayGlyph("FAILED")).toEqual({ glyph: "✕", tone: "rose", label: "FAILED" });
    expect(scanDayGlyph("PARTIAL").label).toBe("PARTIAL");
  });
});

describe("formatting", () => {
  it("formats bytes at sensible units", () => {
    expect(formatBytes(512)).toBe("512 B");
    expect(formatBytes(2048)).toBe("2.0 KB");
    expect(formatBytes(3 * 1024 * 1024)).toBe("3.0 MB");
  });

  it("formats ages", () => {
    expect(formatAgeDays(null)).toBe("—");
    expect(formatAgeDays(0)).toBe("today");
    expect(formatAgeDays(12)).toBe("12d");
  });

  it("shortens ISO weeks", () => {
    expect(shortWeek("2026-W07")).toBe("W07");
    expect(shortWeek("nonsense")).toBe("nonsense");
  });
});

describe("detailHighlights", () => {
  it("speaks flat keys, skips empties and nested blobs, caps at max", () => {
    expect(
      detailHighlights(
        {
          watchlist_ticker: "BBWI",
          items: ["7.01", "9.01"],
          empty: "",
          nothing: null,
          nested: { deep: true },
          form: "10-12B",
          extra: "over the cap",
        },
        3,
      ),
    ).toEqual(["watchlist ticker: BBWI", "items: 7.01, 9.01", "form: 10-12B"]);
  });
});

describe("eventSearchText", () => {
  it("concatenates ticker, company, and type lowercased", () => {
    const card = {
      ticker: "TGT1",
      company_name: "Target Corp",
      event_type: "merger",
    } as EventCard;
    expect(eventSearchText(card)).toBe("tgt1 target corp merger");
  });
});

describe("costChartRows", () => {
  const week = (w: string, scan: number, research: number): CostWeek => ({
    week: w,
    scan_cost_usd: scan,
    scan_runs: scan > 0 ? 1 : 0,
    research_cost_usd: research,
    research_calls: research > 0 ? 2 : 0,
    total_usd: scan + research,
  });

  it("reverses newest-first weeks and scales to the tallest total", () => {
    const rows = costChartRows([week("2026-W03", 1, 1), week("2026-W02", 0, 4), week("2026-W01", 2, 0)]);
    expect(rows.map((r) => r.week)).toEqual(["2026-W01", "2026-W02", "2026-W03"]);
    expect(rows[1].total).toBe(4);
    expect(rows[1].researchFrac).toBe(1);
    expect(rows[0].scanFrac).toBe(0.5);
    expect(rows[2].scanFrac).toBe(0.25);
  });

  it("keeps zero weeks and survives an all-zero ledger", () => {
    const rows = costChartRows([week("2026-W02", 0, 0), week("2026-W01", 0, 0)]);
    expect(rows).toHaveLength(2);
    expect(rows[0].scanFrac).toBe(0);
    expect(rows[0].researchFrac).toBe(0);
  });

  it("caps at the limit, keeping the newest weeks", () => {
    const rows = costChartRows(
      [week("2026-W05", 1, 0), week("2026-W04", 1, 0), week("2026-W03", 1, 0)],
      2,
    );
    expect(rows.map((r) => r.week)).toEqual(["2026-W04", "2026-W05"]);
  });
});
