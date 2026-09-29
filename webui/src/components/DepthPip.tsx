/**
 * Depth pip — the table-row reduction of the Depth Gauge.
 *
 * At row scale the full gauge's shelves and wake read as noise, so the pip
 * keeps only the depth story: a vertical water column with the buy zone at
 * the bottom, the anchor's fair-value tick above it, and the price as a
 * floating dot. Frost stays flat (opacity only — the long-table GPU rule).
 */

import { gaugeLayout, type GaugeGeometry, type GaugeInput } from "../lib/gauge";

const PIP_GEOMETRY: GaugeGeometry = {
  width: 14,
  height: 34,
  padTop: 4,
  padBottom: 4,
  wakeLeft: 0,
  wakeRight: 0,
};

const FROST_OPACITY = [1, 0.85, 0.62, 0.4] as const;

export function DepthPip({ input }: { input: GaugeInput }) {
  const layout = gaugeLayout(input, PIP_GEOMETRY);
  const { width, height } = PIP_GEOMETRY;
  const inZone = layout.zoneState === "in-zone";
  const description =
    `${input.ticker}: ` +
    (layout.zoneState === "in-zone"
      ? "price in the buy zone"
      : layout.zoneState === "above"
        ? "price above the buy zone"
        : "buy zone unknown") +
    (layout.distancePct !== null ? `, ${layout.distancePct.toFixed(1)}% from target` : "");

  return (
    <svg
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={description}
      style={{ opacity: FROST_OPACITY[layout.frost] }}
    >
      {/* The column. */}
      <line
        x1={width / 2}
        x2={width / 2}
        y1={2}
        y2={height - 2}
        stroke="var(--color-glass-tint)"
        strokeOpacity={0.22}
      />
      {/* The zone below the waterline. */}
      {layout.zone && (
        <>
          <rect
            x={1.5}
            y={layout.zone.y}
            width={width - 3}
            height={Math.max(0, height - 2 - layout.zone.y)}
            rx={1.5}
            fill="var(--color-jade)"
            fillOpacity={inZone ? 0.3 : 0.16}
          />
          <line
            x1={0}
            x2={width}
            y1={layout.zone.y}
            y2={layout.zone.y}
            stroke="var(--color-jade)"
            strokeOpacity={0.85}
          />
        </>
      )}
      {/* Anchor fair-value tick. */}
      {layout.shelves[0] && (
        <line
          x1={3}
          x2={width - 3}
          y1={layout.shelves[0].y}
          y2={layout.shelves[0].y}
          stroke="var(--color-moonlight)"
          strokeOpacity={0.5}
        />
      )}
      {/* Price. */}
      {layout.priceY !== null && (
        <circle
          cx={width / 2}
          cy={layout.priceY}
          r={2.6}
          fill={inZone ? "var(--nacre-ink)" : "var(--color-moonlight)"}
        />
      )}
    </svg>
  );
}
