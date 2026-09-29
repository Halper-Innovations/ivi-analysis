import { describe, expect, it } from "vitest";

import type { SearchResult } from "./api";
import { presentRegistrantName, searchResultHint } from "./palette";

describe("presentRegistrantName", () => {
  it("title-cases filing-caps names", () => {
    expect(presentRegistrantName("BOYD GAMING CORP")).toBe("Boyd Gaming Corp");
  });

  it("title-cases after hyphens and parens", () => {
    expect(presentRegistrantName("SMITH-JONES (HOLDINGS) INC")).toBe(
      "Smith-Jones (Holdings) Inc",
    );
  });

  it("leaves mixed-case names untouched", () => {
    expect(presentRegistrantName("Boyd Group Services Inc.")).toBe(
      "Boyd Group Services Inc.",
    );
  });

  it("passes null through", () => {
    expect(presentRegistrantName(null)).toBeNull();
  });
});

describe("searchResultHint", () => {
  const base: SearchResult = {
    ticker: "BYD",
    name: "BOYD GAMING CORP",
    sector: "consumer_gaming",
    covered: true,
    presented_status: null,
  };

  it("shows name and humanized status when tracked", () => {
    expect(searchResultHint({ ...base, presented_status: "DEPLOY_READY" })).toBe(
      "Boyd Gaming Corp · deploy ready",
    );
  });

  it("falls back to the sector when untracked", () => {
    expect(searchResultHint(base)).toBe("Boyd Gaming Corp · consumer gaming");
  });

  it("shows just the ticker context when name and sector are missing", () => {
    expect(
      searchResultHint({ ...base, name: null, sector: null, presented_status: null }),
    ).toBe("");
  });
});

