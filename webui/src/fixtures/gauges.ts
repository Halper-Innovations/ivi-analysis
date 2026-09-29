/**
 * Gauge Lab fixtures — invented tickers, realistic shapes. Three states the
 * platform actually produces: drifting toward the zone, submerged in it
 * (the nacre moment), and frosted over by stale prices.
 */

import type { GaugeInput } from "../lib/gauge";

function drift(from: number, to: number, n: number, wobble: number): number[] {
  const points: number[] = [];
  for (let i = 0; i < n; i += 1) {
    const t = i / (n - 1);
    const base = from + (to - from) * t;
    // Deterministic wobble — fixtures must render identically every load.
    const w = Math.sin(i * 1.7) * wobble * (1 - t * 0.4);
    points.push(Math.round((base + w) * 100) / 100);
  }
  return points;
}

export const APPROACHING: GaugeInput = {
  ticker: "HELIO",
  shelves: [
    { method: "dcf", label: "DCF", low: 74, base: 102, high: 128, emphasized: true },
    { method: "epv", label: "EPV", value: 88 },
    { method: "graham", label: "Graham", value: 96.5 },
  ],
  buyTarget: 80,
  price: 96.4,
  wake: drift(104.2, 96.4, 30, 1.6),
  priceAgeHours: 3,
};

export const IN_ZONE: GaugeInput = {
  ticker: "MERID",
  shelves: [
    { method: "dcf", label: "DCF", low: 71, base: 98.5, high: 117, emphasized: true },
    { method: "epv", label: "EPV", value: 84.2 },
    { method: "graham", label: "Graham", value: 91 },
  ],
  buyTarget: 80,
  price: 77.9,
  wake: drift(89.5, 77.9, 30, 1.2),
  priceAgeHours: 6,
};

export const FROSTED: GaugeInput = {
  ticker: "BOREA",
  shelves: [
    { method: "dcf", label: "DCF", low: 33, base: 46, high: 58, emphasized: true },
    { method: "epv", label: "EPV", value: 39.8 },
  ],
  buyTarget: 36,
  price: 41.2,
  wake: drift(43.8, 41.2, 30, 0.7),
  priceAgeHours: 8 * 24,
};

export const SUSPECT: GaugeInput = {
  ...FROSTED,
  ticker: "PELAG",
  priceAgeHours: 12 * 24,
  priceSuspect: true,
};
