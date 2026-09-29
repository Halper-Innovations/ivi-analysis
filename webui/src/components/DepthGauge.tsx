/**
 * The Depth Gauge — a vertical valuation ladder rendered as water depth.
 *
 * Fair-value estimates are horizontal lines; the buy zone below the target is
 * the submerged jade band (the waterline); price floats with a 30-day wake.
 * When price submerges into the zone, the band takes the nacre shimmer —
 * the only place the reserved accent appears.
 *
 * Hand-built SVG over d3-scale (no chart-library look). Identity is never
 * color-alone: every shelf carries a direct label, and the whole SVG has
 * an aria description. (Table rows use the DepthPip reduction instead.)
 */

import { useId, useState } from "react";

import {
  GAUGE_SIZES,
  depthTicks,
  formatMoney,
  gaugeLayout,
  resolveLabelYs,
  type GaugeInput,
  type GaugeShelfLayout,
  type LabelEntry,
} from "../lib/gauge";

export type GaugeSize = "card" | "hero";

interface DepthGaugeProps {
  input: GaugeInput;
  size: GaugeSize;
}

interface ShelfHover {
  shelf: GaugeShelfLayout;
  x: number;
  y: number;
}

const ZONE_LABELS: Record<string, string> = {
  "in-zone": "price in the buy zone",
  above: "price above the buy zone",
  unknown: "buy zone unknown",
};

export function DepthGauge({ input, size }: DepthGaugeProps) {
  const geometry = GAUGE_SIZES[size];
  const layout = gaugeLayout(input, geometry);
  const uid = useId();
  const [hover, setHover] = useState<ShelfHover | null>(null);

  const { width, height } = geometry;
  const inZone = layout.zoneState === "in-zone";
  const shelfRight = width - (size === "hero" ? 118 : 64);
  const labelX = shelfRight + 8;
  const frostClass = `frost-${layout.frost}`;

  // The label column de-overlaps as one set: marks keep their true y, text
  // gives way (price wins ties by being listed first at the same y).
  const labelEntries: LabelEntry[] = [];
  if (layout.priceY !== null) labelEntries.push({ id: "price", y: layout.priceY });
  for (const shelf of layout.shelves) labelEntries.push({ id: shelf.method, y: shelf.y });
  if (layout.zone) labelEntries.push({ id: "buy", y: layout.zone.y });
  const labelYs = resolveLabelYs(labelEntries, size === "hero" ? 14 : 12);
  const labelY = (id: string, fallback: number) => labelYs.get(id) ?? fallback;

  const description =
    `${input.ticker}: ${ZONE_LABELS[layout.zoneState]}` +
    (layout.distancePct !== null ? `, ${layout.distancePct.toFixed(1)}% from target` : "");

  const frostWrap = layout.frost > 0 ? `frosted ${frostClass}` : "";

  return (
    <div className={`relative inline-block ${frostWrap}`} style={{ width, height }}>
      <div className="frost-content">
        <svg
          width={width}
          height={height}
          viewBox={`0 0 ${width} ${height}`}
          role="img"
          aria-label={description}
          className={size === "hero" ? "gauge-hero" : undefined}
        >
          <defs>
            <linearGradient id={`${uid}-zone`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="var(--color-jade)" stopOpacity="0.28" />
              <stop offset="100%" stopColor="var(--color-jade)" stopOpacity="0.04" />
            </linearGradient>
            <linearGradient id={`${uid}-nacre`} x1="0" y1="0" x2="1" y2="0.25">
              <stop offset="0%" stopColor="#B7C9E8" stopOpacity="0.5" />
              <stop offset="45%" stopColor="#E8C9D8" stopOpacity="0.42" />
              <stop offset="100%" stopColor="var(--nacre-ink)" stopOpacity="0.5" />
            </linearGradient>
            <clipPath id={`${uid}-zone-clip`}>
              {layout.zone && (
                <rect x={0} y={layout.zone.y} width={width} height={layout.zone.height} />
              )}
            </clipPath>
          </defs>

          {/* Depth ticks — recessive price gridlines, hero only. */}
          {size === "hero" &&
            depthTicks(layout.domain).map((tick) => (
              <g key={tick}>
                <line
                  x1={0}
                  x2={shelfRight}
                  y1={layout.y(tick)}
                  y2={layout.y(tick)}
                  stroke="var(--color-glass-tint)"
                  strokeOpacity={0.07}
                />
                <text
                  x={4}
                  y={layout.y(tick) - 3}
                  fill="var(--color-sounding)"
                  fillOpacity={0.55}
                  fontSize={10}
                  className="font-data"
                >
                  {tick}
                </text>
              </g>
            ))}

          {/* The buy zone: submerged water below the waterline. */}
          {layout.zone && (
            <>
              <rect
                x={0}
                y={layout.zone.y}
                width={width}
                height={layout.zone.height}
                fill={`url(#${uid}-zone)`}
              />
              {inZone && (
                <g clipPath={`url(#${uid}-zone-clip)`}>
                  <rect
                    className="gauge-zone-nacre"
                    x={-width * 0.5}
                    y={layout.zone.y}
                    width={width * 2}
                    height={layout.zone.height}
                    fill={`url(#${uid}-nacre)`}
                    opacity={0.55}
                  />
                </g>
              )}
              <line
                x1={0}
                x2={shelfRight}
                y1={layout.zone.y}
                y2={layout.zone.y}
                stroke="var(--color-jade)"
                strokeWidth={1.5}
                strokeDasharray="5 4"
                strokeOpacity={0.85}
              />
              <text
                x={labelX}
                y={labelY("buy", layout.zone.y) + 4}
                fill="var(--color-jade)"
                fontSize={size === "hero" ? 11 : 10}
                className="font-data"
              >
                {size === "hero" && input.buyTarget !== null
                  ? `buy ${formatMoney(input.buyTarget)}`
                  : "buy"}
              </text>
            </>
          )}

          {/* Fair-value shelves. */}
          {layout.shelves.map((shelf, i) => {
            const stroke = shelf.emphasized ? "var(--color-moonlight)" : "var(--color-glass-tint)";
            const strokeOpacity = shelf.emphasized ? 0.95 : 0.55;
            const common = {
              onMouseEnter: () => setHover({ shelf, x: shelfRight / 2, y: shelf.y }),
              onMouseLeave: () => setHover(null),
              onFocus: () => setHover({ shelf, x: shelfRight / 2, y: shelf.y }),
              onBlur: () => setHover(null),
            };
            return (
              <g key={shelf.method} style={{ ["--shelf-i" as string]: i }}>
                {shelf.kind === "band" && shelf.yHigh !== undefined && shelf.yLow !== undefined && (
                  <rect
                    className="gauge-band"
                    x={0}
                    y={shelf.yHigh}
                    width={shelfRight}
                    height={Math.max(1, shelf.yLow - shelf.yHigh)}
                    fill="var(--color-glass-tint)"
                    fillOpacity={0.08}
                  />
                )}
                <line
                  className="gauge-shelf-line"
                  x1={0}
                  x2={shelfRight}
                  y1={shelf.y}
                  y2={shelf.y}
                  stroke={stroke}
                  strokeOpacity={strokeOpacity}
                  strokeWidth={shelf.emphasized ? 2 : 1.25}
                />
                <text
                  x={labelX}
                  y={labelY(shelf.method, shelf.y) + 4}
                  fill={shelf.emphasized ? "var(--color-moonlight)" : "var(--color-sounding)"}
                  fontSize={size === "hero" ? 11 : 10}
                  className="font-data"
                >
                  {shelf.label}
                  {size === "hero" ? ` ${formatMoney(shelf.value)}` : ""}
                </text>
                {size === "hero" && (
                  <rect
                    x={0}
                    y={shelf.y - 8}
                    width={shelfRight}
                    height={16}
                    fill="transparent"
                    tabIndex={0}
                    role="note"
                    aria-label={`${shelf.label} fair value ${formatMoney(shelf.value)}${
                      shelf.kind === "band" && shelf.low !== undefined && shelf.high !== undefined
                        ? `, band ${formatMoney(shelf.low)} to ${formatMoney(shelf.high)}`
                        : ""
                    }`}
                    {...common}
                  />
                )}
              </g>
            );
          })}

          {/* The floating price marker and its 30-day wake. */}
          {layout.priceY !== null && (
            <g className="gauge-price-group">
              {layout.wake.length >= 2 && (
                <polyline
                  points={layout.wake.map((p) => `${p.x},${p.y}`).join(" ")}
                  fill="none"
                  stroke={inZone ? "var(--nacre-ink)" : "var(--color-moonlight)"}
                  strokeOpacity={0.35}
                  strokeWidth={1}
                />
              )}
              <line
                x1={geometry.wakeRight}
                x2={shelfRight}
                y1={layout.priceY}
                y2={layout.priceY}
                stroke={inZone ? "var(--nacre-ink)" : "var(--color-moonlight)"}
                strokeWidth={1.5}
                strokeOpacity={0.9}
              />
              <circle
                className="gauge-ripple"
                cx={shelfRight - 10}
                cy={layout.priceY}
                r={7}
                fill="none"
                stroke={inZone ? "var(--nacre-ink)" : "var(--color-moonlight)"}
                strokeWidth={1}
              />
              <circle
                cx={shelfRight - 10}
                cy={layout.priceY}
                r={3.5}
                fill={inZone ? "var(--nacre-ink)" : "var(--color-moonlight)"}
              />
              <text
                x={labelX}
                y={labelY("price", layout.priceY) + 4}
                fill={inZone ? "var(--nacre-ink)" : "var(--color-moonlight)"}
                fontSize={size === "hero" ? 12 : 10}
                fontWeight={600}
                className="font-data"
              >
                {input.price !== null ? formatMoney(input.price) : ""}
              </text>
            </g>
          )}
        </svg>

        {/* Provenance tooltip (hero): hovering a shelf names its method. */}
        {size === "hero" && hover && (
          <div
            className="gauge-tooltip rounded-[2px] border border-glass-tint/20 bg-deepwater px-3 py-1.5 text-xs text-moonlight"
            style={{ left: hover.x, top: hover.y }}
          >
            <span className="font-medium">{hover.shelf.label}</span>{" "}
            <span className="font-data">{formatMoney(hover.shelf.value)}</span>
            {hover.shelf.kind === "band" &&
              hover.shelf.low !== undefined &&
              hover.shelf.high !== undefined && (
                <span className="text-sounding font-data">
                  {" "}
                  · {formatMoney(hover.shelf.low)}–{formatMoney(hover.shelf.high)}
                </span>
              )}
          </div>
        )}
      </div>
    </div>
  );
}
