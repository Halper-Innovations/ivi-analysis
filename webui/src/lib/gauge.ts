/**
 * Depth Gauge math — pure functions, no DOM.
 *
 * The gauge renders a vertical valuation ladder as water depth: fair-value
 * estimates are fair-value lines, the buy zone below `buyTarget` is the
 * submerged waterline band, and the current price floats with a 30-day
 * wake. Everything here is deterministic and unit-tested; the component
 * only draws what this module lays out.
 */

import { scaleLinear } from "d3-scale";

export interface GaugeShelf {
  /** Stable method id, e.g. "dcf" | "epv" | "graham". */
  method: string;
  /** Direct label — identity is never color-alone. */
  label: string;
  /** Single fair-value line (EPV, Graham, …). */
  value?: number;
  /** Band shelf (DCF low/base/high). */
  low?: number;
  base?: number;
  high?: number;
  /** The watchlist's valuation anchor gets the emphasized shelf. */
  emphasized?: boolean;
}

export interface GaugeInput {
  ticker: string;
  shelves: GaugeShelf[];
  buyTarget: number | null;
  price: number | null;
  /** Oldest → newest trailing closes (the wake). */
  wake: number[];
  /** Hours since the price snapshot; null = age unknown. */
  priceAgeHours: number | null;
  /** PRICE_DATA_SUSPECT rows render iced over regardless of age. */
  priceSuspect?: boolean;
}

export type ZoneState = "in-zone" | "above" | "unknown";

/** Frost levels map to the platform's freshness contract:
 *  0 = fresh (< 24h), 1 = aging (24–96h, inside the deadman ceiling),
 *  2 = past the 96h price-snapshot ceiling (or age unknown),
 *  3 = PRICE_DATA_SUSPECT — iced over. */
export const FROST_FRESH_HOURS = 24;
export const FROST_CEILING_HOURS = 96;

export function frostLevel(priceAgeHours: number | null, priceSuspect = false): 0 | 1 | 2 | 3 {
  if (priceSuspect) return 3;
  if (priceAgeHours === null || priceAgeHours >= FROST_CEILING_HOURS) return 2;
  if (priceAgeHours >= FROST_FRESH_HOURS) return 1;
  return 0;
}

export function zoneState(price: number | null, buyTarget: number | null): ZoneState {
  if (price === null || buyTarget === null || buyTarget <= 0) return "unknown";
  return price <= buyTarget ? "in-zone" : "above";
}

/** Percent above the buy target (negative = submerged in the zone). */
export function distanceToZonePct(price: number | null, buyTarget: number | null): number | null {
  if (price === null || buyTarget === null || buyTarget <= 0) return null;
  return ((price - buyTarget) / buyTarget) * 100;
}

function collectValues(input: GaugeInput): number[] {
  const values: number[] = [];
  for (const shelf of input.shelves) {
    for (const v of [shelf.value, shelf.low, shelf.base, shelf.high]) {
      if (typeof v === "number" && Number.isFinite(v)) values.push(v);
    }
  }
  if (input.buyTarget !== null) values.push(input.buyTarget);
  if (input.price !== null) values.push(input.price);
  for (const v of input.wake) if (Number.isFinite(v)) values.push(v);
  return values;
}

/**
 * Price domain [floor, ceiling] with 8% padding each side, so the deepest
 * shelf and the highest mark both sit clear of the frame and the zone band
 * always has visible water below the waterline.
 */
export function gaugeDomain(input: GaugeInput): [number, number] {
  const values = collectValues(input);
  if (values.length === 0) return [0, 1];
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  if (lo === hi) {
    if (lo === 0) return [-1, 1];
    return [lo * 0.92, hi * 1.08];
  }
  const pad = (hi - lo) * 0.08;
  return [lo - pad, hi + pad];
}

export interface GaugeShelfLayout {
  method: string;
  label: string;
  emphasized: boolean;
  kind: "line" | "band";
  /** y of value (line) or base (band). */
  y: number;
  yLow?: number;
  yHigh?: number;
  value: number;
  low?: number;
  high?: number;
}

export interface WakePoint {
  x: number;
  y: number;
}

export interface GaugeLayout {
  width: number;
  height: number;
  domain: [number, number];
  /** y for a given price (exposed for ticks/tests). */
  y: (price: number) => number;
  shelves: GaugeShelfLayout[];
  /** Waterline (buy target) y, or null when no target exists. */
  waterlineY: number | null;
  /** The submerged jade band: from the waterline down to the frame floor. */
  zone: { y: number; height: number } | null;
  priceY: number | null;
  wake: WakePoint[];
  zoneState: ZoneState;
  frost: 0 | 1 | 2 | 3;
  distancePct: number | null;
}

export interface GaugeGeometry {
  width: number;
  height: number;
  padTop: number;
  padBottom: number;
  /** Horizontal band the wake occupies, in px from the left edge. */
  wakeLeft: number;
  wakeRight: number;
}

export const GAUGE_SIZES: Record<"card" | "hero", GaugeGeometry> = {
  card: { width: 200, height: 240, padTop: 18, padBottom: 14, wakeLeft: 16, wakeRight: 108 },
  hero: { width: 400, height: 480, padTop: 28, padBottom: 22, wakeLeft: 36, wakeRight: 210 },
};

export function gaugeLayout(input: GaugeInput, geometry: GaugeGeometry): GaugeLayout {
  const domain = gaugeDomain(input);
  const scale = scaleLinear()
    .domain(domain)
    .range([geometry.height - geometry.padBottom, geometry.padTop]);
  // Sub-pixel noise is meaningless at these sizes; 2dp keeps layouts stable.
  const y = (price: number) => Math.round(scale(price) * 100) / 100;

  const shelves: GaugeShelfLayout[] = input.shelves.flatMap((shelf): GaugeShelfLayout[] => {
    if (typeof shelf.low === "number" && typeof shelf.high === "number") {
      const base = typeof shelf.base === "number" ? shelf.base : (shelf.low + shelf.high) / 2;
      return [{
        method: shelf.method,
        label: shelf.label,
        emphasized: Boolean(shelf.emphasized),
        kind: "band" as const,
        y: y(base),
        yLow: y(shelf.low),
        yHigh: y(shelf.high),
        value: base,
        low: shelf.low,
        high: shelf.high,
      }];
    }
    // A base with no band (the durable DCF) is drawn as a line at the base.
    // A shelf with no number at all is not drawn: $0 is not a fair value.
    const value = typeof shelf.value === "number" ? shelf.value : shelf.base;
    if (typeof value !== "number" || !Number.isFinite(value)) return [];
    return [{
      method: shelf.method,
      label: shelf.label,
      emphasized: Boolean(shelf.emphasized),
      kind: "line" as const,
      y: y(value),
      value,
    }];
  });

  const waterlineY = input.buyTarget !== null ? y(input.buyTarget) : null;
  const floorY = geometry.height;
  const zone =
    waterlineY !== null ? { y: waterlineY, height: Math.max(0, floorY - waterlineY) } : null;

  const wake: WakePoint[] = [];
  const n = input.wake.length;
  if (n >= 2) {
    const step = (geometry.wakeRight - geometry.wakeLeft) / (n - 1);
    for (let i = 0; i < n; i += 1) {
      wake.push({ x: geometry.wakeLeft + step * i, y: y(input.wake[i]) });
    }
  }

  return {
    width: geometry.width,
    height: geometry.height,
    domain,
    y,
    shelves,
    waterlineY,
    zone,
    priceY: input.price !== null ? y(input.price) : null,
    wake,
    zoneState: zoneState(input.price, input.buyTarget),
    frost: frostLevel(input.priceAgeHours, input.priceSuspect ?? false),
    distancePct: distanceToZonePct(input.price, input.buyTarget),
  };
}

export interface LabelEntry {
  id: string;
  y: number;
}

/**
 * De-overlap a column of labels: sorted by y, each label is pushed down to
 * keep at least `minGap` px from its predecessor. Marks stay at their true
 * y; only the text moves.
 */
export function resolveLabelYs(entries: LabelEntry[], minGap: number): Map<string, number> {
  const sorted = [...entries].sort((a, b) => a.y - b.y);
  const resolved = new Map<string, number>();
  let previous = Number.NEGATIVE_INFINITY;
  for (const entry of sorted) {
    const y = Math.max(entry.y, previous + minGap);
    resolved.set(entry.id, y);
    previous = y;
  }
  return resolved;
}

/** Round depth ticks (price gridlines) for the hero gauge. */
export function depthTicks(domain: [number, number], count = 5): number[] {
  return scaleLinear().domain(domain).nice().ticks(count).filter((t) => t >= domain[0] && t <= domain[1]);
}

export function formatMoney(value: number): string {
  const sign = value < 0 ? "-" : "";
  return `${sign}$${Math.abs(value).toLocaleString("en-US", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })}`;
}
