import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { StatusChip } from "../components/chips";
import type { WatchlistRow, WaterlineItem } from "./api";
import { glossaryDefinition } from "./glossary";
import {
  distinctValues,
  filterRows,
  formatAge,
  formatMillions,
  formatMoneyCompact,
  formatPct,
  formatRawCompact,
  groupForBoard,
  hoursSince,
  humanizeEventPending,
  humanizeSignalCode,
  humanizeToken,
  leaderboardRows,
  marginOfSafetyPct,
  rowGaugeInput,
  waterlineGaugeInput,
} from "./floor";

const NOW = new Date("2026-07-21T12:00:00Z");

describe("StatusChip", () => {
  it("displays an underscore-free label while preserving the raw enum tooltip", () => {
    const html = renderToStaticMarkup(createElement(StatusChip, { value: "DEPLOY_READY" }));

    expect(html).toContain('title="DEPLOY_READY"');
    expect(html).toContain(">DEPLOY READY</span>");
    expect(html).not.toContain(">DEPLOY_READY</span>");
  });
});

describe("glossaryDefinition", () => {
  it("matches actual event tokens and EVENT_PENDING flag spellings", () => {
    expect(glossaryDefinition("busted_ipo")).toBe(
      "an IPO from the last two years trading at least 50% below its debut close — a forced-seller setup",
    );
    expect(glossaryDefinition("ch11_emergence")).toBe(
      "a company exiting bankruptcy with a restructured balance sheet and fresh equity",
    );
    expect(glossaryDefinition("EVENT_PENDING:UNREVIEWED_8K")).toBe(
      "an SEC filing companies must publish within days of a material corporate event",
    );
    expect(glossaryDefinition("merger")).toBeUndefined();
  });
});

function row(overrides: Partial<WatchlistRow>): WatchlistRow {
  return {
    id: 1,
    ticker: "AAA",
    status: "ACTIVE",
    presented_status: "ACTIVE",
    event_pending: null,
    conviction_grade: "WATCHLIST_ONLY",
    confidence: "MODERATE",
    conviction_source: null,
    pipeline_version: null,
    candidate_disposition: null,
    decision_basis: null,
    selection_validation_status: null,
    price_trigger_eligible: true,
    scan_family: "normal",
    latest_price: 95,
    latest_price_source: null,
    latest_price_checked_at: "2026-07-21T09:00:00Z",
    buy_price_target: 80,
    distance_from_buy_pct: 18.75,
    valuation_anchor_method: "DCF",
    valuation_anchor_value: 106.67,
    source_sector: "industrial_tech",
    status_reason: null,
    added_at: null,
    last_evaluated_at: null,
    falsifiers: [],
    market_cap_mm: null,
    cap_source: null,
    cap_band: null,
    cap_band_label: "UNKNOWN_CAP",
    adv_dollar_20d: null,
    adv_dollar_60d: null,
    adv_asof: null,
    capacity_class: "ADV_UNKNOWN",
    cheapness: "n/a",
    ...overrides,
  };
}

describe("hoursSince", () => {
  it("computes exact hours from ISO to now", () => {
    expect(hoursSince("2026-07-21T09:00:00Z", NOW)).toBe(3);
    expect(hoursSince("2026-07-17T12:00:00Z", NOW)).toBe(96);
  });
  it("is null for missing or junk input", () => {
    expect(hoursSince(null, NOW)).toBeNull();
    expect(hoursSince("not-a-date", NOW)).toBeNull();
  });
});

describe("formatting", () => {
  it("formats ages across the frost thresholds", () => {
    expect(formatAge(3)).toBe("3h");
    expect(formatAge(47.6)).toBe("48h");
    expect(formatAge(100.8)).toBe("4.2d");
    expect(formatAge(null)).toBe("age unknown");
  });
  it("formats money compactly above $1000", () => {
    expect(formatMoneyCompact(78.5)).toBe("$78.50");
    expect(formatMoneyCompact(2181.37)).toBe("$2,181");
    expect(formatMoneyCompact(null)).toBe("—");
  });
  it("formats millions into M/B with leading sign", () => {
    expect(formatMillions(1944.9)).toBe("$1.9B");
    expect(formatMillions(311.2)).toBe("$311M");
    expect(formatMillions(4.53)).toBe("$4.5M");
    expect(formatMillions(-487.2)).toBe("-$487M");
    expect(formatMillions(null)).toBe("—");
  });
  it("formats raw dollar/count magnitudes", () => {
    expect(formatRawCompact(144602000)).toBe("$144.6M");
    expect(formatRawCompact(1944901000)).toBe("$1.94B");
    expect(formatRawCompact(53000000, false)).toBe("53.0M");
    expect(formatRawCompact(-2500000)).toBe("-$2.5M");
    expect(formatRawCompact(null)).toBe("—");
  });
  it("keeps ratio signal in non-money small numbers", () => {
    expect(formatRawCompact(0.0013471, false)).toBe("0.00135");
    expect(formatRawCompact(34.4108, false)).toBe("34.41");
    expect(formatRawCompact(34, false)).toBe("34");
    expect(formatRawCompact(-0.153, false)).toBe("-0.153");
  });
  it("formats signed percents", () => {
    expect(formatPct(18.75)).toBe("+18.8%");
    expect(formatPct(-2.5)).toBe("-2.5%");
    expect(formatPct(null)).toBe("—");
  });
});

describe("marginOfSafetyPct", () => {
  it("is the discount to the anchor at the current price", () => {
    expect(marginOfSafetyPct(80, 100)).toBe(20);
    expect(marginOfSafetyPct(120, 100)).toBeCloseTo(-20);
  });
  it("is null without a positive anchor", () => {
    expect(marginOfSafetyPct(80, null)).toBeNull();
    expect(marginOfSafetyPct(null, 100)).toBeNull();
    expect(marginOfSafetyPct(80, 0)).toBeNull();
  });
});

describe("rowGaugeInput", () => {
  it("builds the anchor shelf and frost inputs from a queue row", () => {
    const input = rowGaugeInput(row({}), NOW, [96, 95]);
    expect(input).toEqual({
      ticker: "AAA",
      shelves: [{ method: "anchor", label: "DCF", value: 106.67, emphasized: true }],
      buyTarget: 80,
      price: 95,
      wake: [96, 95],
      priceAgeHours: 3,
      priceSuspect: false,
    });
  });
  it("ices suspect rows and tolerates a missing anchor", () => {
    const input = rowGaugeInput(
      row({ status: "PRICE_DATA_SUSPECT", valuation_anchor_value: null }),
      NOW,
    );
    expect(input.priceSuspect).toBe(true);
    expect(input.shelves).toEqual([]);
    expect(input.wake).toEqual([]);
  });
});

describe("waterlineGaugeInput", () => {
  it("uses the server's age and wake untouched", () => {
    const item: WaterlineItem = {
      watchlist_id: 7,
      ticker: "CCC",
      presented_status: "DEPLOY_READY",
      conviction_grade: "ACTIONABLE",
      confidence: "HIGH",
      latest_price: 78,
      latest_price_checked_at: "2026-07-21T11:00:00Z",
      price_age_hours: 1,
      price_suspect: false,
      buy_price_target: 80,
      distance_from_buy_pct: -2.5,
      valuation_anchor_method: "DCF",
      valuation_anchor_value: 106.67,
      source_sector: "industrial_tech",
      wake: [80, 79, 78],
    };
    const input = waterlineGaugeInput(item);
    expect(input.priceAgeHours).toBe(1);
    expect(input.wake).toEqual([80, 79, 78]);
    expect(input.buyTarget).toBe(80);
  });
});

describe("groupForBoard", () => {
  it("keeps the platform's column order and appends unknown statuses", () => {
    const rows = [
      row({ id: 1, presented_status: "ACTIVE" }),
      row({ id: 2, presented_status: "DEPLOY_READY" }),
      row({ id: 3, presented_status: "EVENT_PENDING" }),
      row({ id: 4, presented_status: "SOMETHING_NEW" }),
    ];
    const groups = groupForBoard(rows);
    expect([...groups.keys()]).toEqual([
      "BUY_CONFIRMED",
      "DEPLOY_READY",
      "EVENT_PENDING",
      "ACTIVE",
      "UNCERTAIN",
      "PRICE_DATA_SUSPECT",
      "QUARANTINE",
      "SOMETHING_NEW",
    ]);
    expect(groups.get("DEPLOY_READY")!.map((r) => r.id)).toEqual([2]);
  });
});

describe("leaderboardRows", () => {
  it("sorts by distance ascending and drops null distances", () => {
    const rows = [
      row({ id: 1, distance_from_buy_pct: 18.75 }),
      row({ id: 2, distance_from_buy_pct: -2.5 }),
      row({ id: 3, distance_from_buy_pct: null }),
      row({ id: 4, distance_from_buy_pct: 4 }),
    ];
    expect(leaderboardRows(rows).map((r) => r.id)).toEqual([2, 4, 1]);
  });
  it("excludes quarantined and price-suspect rows from the race", () => {
    const rows = [
      row({ id: 1, distance_from_buy_pct: -99.7, presented_status: "QUARANTINE" }),
      row({ id: 2, distance_from_buy_pct: -50, presented_status: "PRICE_DATA_SUSPECT" }),
      row({ id: 3, distance_from_buy_pct: 4, presented_status: "DEPLOY_READY" }),
    ];
    expect(leaderboardRows(rows).map((r) => r.id)).toEqual([3]);
  });
});

describe("filterRows", () => {
  const rows = [
    row({ id: 1, ticker: "AAA", cap_band_label: "mid", conviction_grade: "ACTIONABLE" }),
    row({ id: 2, ticker: "BBB", cap_band_label: "micro", scan_family: "pearl" }),
    row({ id: 3, ticker: "ABC", presented_status: "QUARANTINE" }),
  ];
  it("matches ticker substring case-insensitively", () => {
    expect(filterRows(rows, { q: "ab" }).map((r) => r.id)).toEqual([3]);
  });
  it("combines facet filters", () => {
    expect(filterRows(rows, { band: "mid", grade: "ACTIONABLE" }).map((r) => r.id)).toEqual([1]);
    expect(filterRows(rows, { family: "pearl" }).map((r) => r.id)).toEqual([2]);
    expect(filterRows(rows, { status: "QUARANTINE" }).map((r) => r.id)).toEqual([3]);
  });
  it("collects distinct facet values sorted", () => {
    expect(distinctValues(rows, "cap_band_label")).toEqual(["UNKNOWN_CAP", "micro", "mid"]);
  });
});

describe("humanizing", () => {
  it("lowercases snake_case and preserves known acronyms", () => {
    expect(humanizeToken("payments_fintech")).toBe("payments fintech");
    expect(humanizeToken("UNKNOWN_CAP")).toBe("unknown cap");
    expect(humanizeToken("UNREVIEWED_8K")).toBe("unreviewed 8-K");
    expect(humanizeToken("CH11_EMERGENCE")).toBe("Chapter 11 emergence");
    expect(humanizeToken("DCF")).toBe("DCF");
    expect(humanizeToken(null)).toBe("");
  });
  it("strips headwind/support suffixes from signal codes", () => {
    expect(humanizeSignalCode("SECULAR_DECLINE_HEADWIND")).toBe("secular decline");
    expect(humanizeSignalCode("STRONG_CASH_CONVERSION_SUPPORT")).toBe("strong cash conversion");
    expect(humanizeSignalCode("NET_DILUTION_HEADWIND")).toBe("net dilution");
  });
  it("renders event_pending flag lists as deduped prose", () => {
    expect(
      humanizeEventPending("EVENT_PENDING:MATERIAL_AGREEMENT,EVENT_PENDING:UNREVIEWED_8K"),
    ).toBe("material agreement · unreviewed 8-K");
    expect(
      humanizeEventPending(
        "EVENT_PENDING:UNREVIEWED_8K,EVENT_PENDING:UNREVIEWED_8K,EVENT_PENDING:RESTRUCTURING",
      ),
    ).toBe("unreviewed 8-K · restructuring");
    expect(humanizeEventPending("merger")).toBe("merger");
    expect(humanizeEventPending(null)).toBe("");
  });
});
