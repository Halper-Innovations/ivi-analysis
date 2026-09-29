import { describe, expect, it } from "vitest";

import {
  depthTicks,
  distanceToZonePct,
  formatMoney,
  frostLevel,
  gaugeDomain,
  gaugeLayout,
  resolveLabelYs,
  zoneState,
  type GaugeGeometry,
  type GaugeInput,
} from "./gauge";

const INPUT: GaugeInput = {
  ticker: "HELIO",
  shelves: [
    { method: "dcf", label: "DCF", low: 70, base: 100, high: 120, emphasized: true },
    { method: "epv", label: "EPV", value: 90 },
  ],
  buyTarget: 80,
  price: 96,
  wake: [100, 96],
  priceAgeHours: 3,
};

// height 240 with 4px pads over domain [66, 124] → exactly 4px per dollar.
const GEOMETRY: GaugeGeometry = {
  width: 200,
  height: 240,
  padTop: 4,
  padBottom: 4,
  wakeLeft: 4,
  wakeRight: 40,
};

describe("gaugeDomain", () => {
  it("pads the collected min/max by 8%", () => {
    expect(gaugeDomain(INPUT)).toEqual([66, 124]);
  });

  it("handles a degenerate single value", () => {
    const input: GaugeInput = {
      ticker: "X",
      shelves: [],
      buyTarget: null,
      price: 50,
      wake: [],
      priceAgeHours: null,
    };
    expect(gaugeDomain(input)).toEqual([46, 54]);
  });

  it("handles no values at all", () => {
    const input: GaugeInput = {
      ticker: "X",
      shelves: [],
      buyTarget: null,
      price: null,
      wake: [],
      priceAgeHours: null,
    };
    expect(gaugeDomain(input)).toEqual([0, 1]);
  });
});

describe("zoneState", () => {
  it("is in-zone at or below the buy target", () => {
    expect(zoneState(78, 80)).toBe("in-zone");
    expect(zoneState(80, 80)).toBe("in-zone");
  });
  it("is above when price floats over the target", () => {
    expect(zoneState(85, 80)).toBe("above");
  });
  it("is unknown without both numbers", () => {
    expect(zoneState(null, 80)).toBe("unknown");
    expect(zoneState(78, null)).toBe("unknown");
    expect(zoneState(78, 0)).toBe("unknown");
  });
});

describe("frostLevel", () => {
  it("maps age to the freshness contract", () => {
    expect(frostLevel(0)).toBe(0);
    expect(frostLevel(23.9)).toBe(0);
    expect(frostLevel(24)).toBe(1);
    expect(frostLevel(95.9)).toBe(1);
    expect(frostLevel(96)).toBe(2);
    expect(frostLevel(null)).toBe(2);
  });
  it("ices over PRICE_DATA_SUSPECT regardless of age", () => {
    expect(frostLevel(1, true)).toBe(3);
  });
});

describe("distanceToZonePct", () => {
  it("is percent above the target", () => {
    expect(distanceToZonePct(96, 80)).toBe(20);
    expect(distanceToZonePct(72, 80)).toBe(-10);
    expect(distanceToZonePct(null, 80)).toBeNull();
  });
});

describe("gaugeLayout", () => {
  const layout = gaugeLayout(INPUT, GEOMETRY);

  it("maps prices to y with higher dollars higher up", () => {
    expect(layout.domain).toEqual([66, 124]);
    expect(layout.y(124)).toBe(4);
    expect(layout.y(66)).toBe(236);
    expect(layout.y(80)).toBe(180);
    expect(layout.priceY).toBe(116);
  });

  it("lays out band and line shelves with direct labels", () => {
    expect(layout.shelves).toHaveLength(2);
    const [dcf, epv] = layout.shelves;
    expect(dcf.kind).toBe("band");
    expect(dcf.label).toBe("DCF");
    expect(dcf.emphasized).toBe(true);
    expect(dcf.yHigh).toBe(20); // y(120)
    expect(dcf.yLow).toBe(220); // y(70)
    expect(dcf.y).toBe(100); // y(base 100)
    expect(epv.kind).toBe("line");
    expect(epv.y).toBe(140); // y(90)
  });

  it("defaults a band's base to the midpoint when absent", () => {
    const input: GaugeInput = {
      ...INPUT,
      shelves: [{ method: "dcf", label: "DCF", low: 70, high: 120 }],
    };
    const mid = gaugeLayout(input, GEOMETRY).shelves[0];
    expect(mid.value).toBe(95);
    expect(mid.y).toBe(120); // y(95)
  });

  it("places the waterline and submerged zone band", () => {
    expect(layout.waterlineY).toBe(180);
    expect(layout.zone).toEqual({ y: 180, height: 60 }); // to the frame floor (240)
  });

  it("draws the wake oldest to newest across the wake band", () => {
    expect(layout.wake).toEqual([
      { x: 4, y: 100 }, // y(100)
      { x: 40, y: 116 }, // y(96)
    ]);
  });

  it("carries zone state, frost, and distance", () => {
    expect(layout.zoneState).toBe("above");
    expect(layout.frost).toBe(0);
    expect(layout.distancePct).toBe(20);
  });

  it("goes nacre when price submerges into the zone", () => {
    const submerged = gaugeLayout({ ...INPUT, price: 78 }, GEOMETRY);
    expect(submerged.zoneState).toBe("in-zone");
    expect(submerged.distancePct).toBe(-2.5);
  });
});

describe("resolveLabelYs", () => {
  it("pushes colliding labels apart while keeping order", () => {
    const resolved = resolveLabelYs(
      [
        { id: "price", y: 100 },
        { id: "graham", y: 100.5 },
        { id: "epv", y: 140 },
      ],
      13,
    );
    expect(resolved.get("price")).toBe(100);
    expect(resolved.get("graham")).toBe(113);
    expect(resolved.get("epv")).toBe(140);
  });

  it("cascades through a pile-up", () => {
    const resolved = resolveLabelYs(
      [
        { id: "a", y: 50 },
        { id: "b", y: 51 },
        { id: "c", y: 52 },
      ],
      10,
    );
    expect(resolved.get("a")).toBe(50);
    expect(resolved.get("b")).toBe(60);
    expect(resolved.get("c")).toBe(70);
  });
});

describe("depthTicks", () => {
  it("returns round in-domain ticks", () => {
    expect(depthTicks([66, 124])).toEqual([70, 80, 90, 100, 110, 120]);
  });
});

describe("formatMoney", () => {
  it("formats with cents and thousands separators", () => {
    expect(formatMoney(1234.5)).toBe("$1,234.50");
    expect(formatMoney(78)).toBe("$78.00");
    expect(formatMoney(-4.01)).toBe("-$4.01");
  });
});

describe("gaugeLayout shelf fallbacks", () => {
  it("draws a base-only shelf (the durable DCF) as a line at the base, never at $0", () => {
    const layout = gaugeLayout(
      {
        ticker: "X",
        shelves: [{ method: "dcf", label: "DCF", base: 60 }],
        buyTarget: 50,
        price: 55,
        wake: [],
        priceAgeHours: 1,
      },
      GEOMETRY,
    );
    expect(layout.shelves).toHaveLength(1);
    expect(layout.shelves[0].kind).toBe("line");
    expect(layout.shelves[0].value).toBe(60);
    expect(layout.shelves[0].y).toBe(layout.y(60));
  });

  it("omits a shelf with no number at all", () => {
    const layout = gaugeLayout(
      {
        ticker: "X",
        shelves: [{ method: "epv", label: "EPV" }],
        buyTarget: 50,
        price: 55,
        wake: [],
        priceAgeHours: 1,
      },
      GEOMETRY,
    );
    expect(layout.shelves).toEqual([]);
  });
});
