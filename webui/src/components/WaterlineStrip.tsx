/**
 * The waterline — Today's hero. Not a card row: a place.
 *
 * A full-bleed cross-section of the water around the buy line. Names hang
 * from sounding lines at their signed distance to their buy targets:
 * above the line = still approaching, below = submerged in the zone.
 * Depth guides mark −20/−40/−60; the seabed footer counts the names too
 * deep for the strip. Markers descend to their depth on load ("taking a
 * sounding"); reduced motion renders them in place.
 *
 * Status is never color-alone: clear names (DEPLOY_READY/BUY_CONFIRMED)
 * are filled dots, held names are rings, and every marker's link text
 * carries the status. Jade appears only on in-zone + clear markers; a
 * held name in the zone wears the amber ring of its status instead.
 */

import { Link } from "@tanstack/react-router";

import type { WaterlineDeeper, WaterlineItem } from "../lib/api";
import {
  clearToAct,
  formatAge,
  formatMoneyCompact,
  formatPct,
  humanizeToken,
} from "../lib/floor";
import { frostLevel } from "../lib/gauge";

const STRIP_HEIGHT = 300;

interface Marker {
  item: WaterlineItem;
  xPct: number;
  yPct: number;
  submerged: boolean;
  clear: boolean;
  frosted: boolean;
}

interface StripModel {
  markers: Marker[];
  linePct: number;
  /** Depth guide lines: [pct-of-height, label]. */
  guides: [number, string][];
}

export function stripModel(items: WaterlineItem[]): StripModel | null {
  const placed = items.filter((i) => i.distance_from_buy_pct !== null);
  if (placed.length === 0) return null;
  const distances = placed.map((i) => i.distance_from_buy_pct!);
  const aboveMax = Math.max(7, ...distances.map((d) => d * 1.35));
  const belowMin = Math.min(-9, ...distances.map((d) => d * 1.3));
  const span = aboveMax - belowMin;
  // Map into a padded band so edge markers keep room for their labels.
  const yPct = (d: number) => 7 + ((aboveMax - d) / span) * 89;

  const markers = placed.map((item, i) => {
    const d = item.distance_from_buy_pct!;
    return {
      item,
      xPct: ((i + 0.5) / placed.length) * 100,
      yPct: yPct(d),
      submerged: d <= 0,
      clear: clearToAct(item.presented_status),
      frosted: frostLevel(item.price_age_hours, item.price_suspect) >= 2,
    };
  });

  const guides: [number, string][] = [];
  for (const depth of [-20, -40, -60]) {
    if (depth > belowMin + 4) guides.push([yPct(depth), `${depth}%`]);
  }
  return { markers, linePct: yPct(0), guides };
}

function MarkerView({ marker, index }: { marker: Marker; index: number }) {
  const { item, submerged, clear } = marker;
  // Labels sit below the dot when submerged — and also for above-line
  // markers riding too close to the frame's top edge.
  const labelBelow = submerged || marker.yPct < 15;
  const dotTone = submerged
    ? clear
      ? "border-jade bg-jade shadow-[0_0_10px_rgba(45,212,167,0.55)]"
      : "border-amber/80 bg-transparent"
    : clear
      ? "border-moonlight bg-moonlight"
      : "border-moonlight/70 bg-transparent";
  const status = humanizeToken(item.presented_status ?? "");
  // The accessible name is the visible text plus an sr-only suffix — never
  // a parallel aria-label that could drift from what sighted users read.
  const srSuffix =
    ` from buy target, ${status}` +
    (marker.frosted ? `, price ${formatAge(item.price_age_hours)} old` : "");

  return (
    <Link
      to="/company/$ticker"
      params={{ ticker: item.ticker }}
      title={`${item.ticker} · ${formatMoneyCompact(item.latest_price)} vs buy ${formatMoneyCompact(
        item.buy_price_target,
      )} · ${status}${marker.frosted ? ` · ${formatAge(item.price_age_hours)} old` : ""}`}
      className={`strip-marker group absolute ${marker.frosted ? "opacity-55" : ""}`}
      style={
        {
          left: `${marker.xPct}%`,
          top: `${marker.yPct}%`,
          "--descend-i": index,
        } as React.CSSProperties
      }
    >
      {/* Dot on the sounding line. */}
      <span
        aria-hidden="true"
        className={`strip-marker-dot absolute left-1/2 top-0 h-2.5 w-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 transition-transform group-hover:scale-125 ${dotTone}`}
      />
      <span
        className={`absolute left-1/2 flex -translate-x-1/2 flex-col items-center gap-0 whitespace-nowrap ${
          labelBelow ? "top-2" : "bottom-2"
        }`}
      >
        <span
          className={`strip-marker-ticker font-data text-[11px] leading-tight ${
            submerged && clear ? "text-jade font-semibold" : "text-moonlight"
          } group-hover:text-jade`}
        >
          {item.ticker}
        </span>
        <span className="strip-marker-detail font-data text-[9.5px] leading-tight text-sounding">
          {formatPct(item.distance_from_buy_pct)}
        </span>
      </span>
      <span className="sr-only">{srSuffix}</span>
    </Link>
  );
}

export function WaterlineStrip({
  items,
  deeper,
}: {
  items: WaterlineItem[];
  deeper: WaterlineDeeper;
}) {
  const model = stripModel(items);
  if (!model) {
    return (
      <div className="border-y border-glass-tint/10">
        <p className="voice px-6 py-10">
          No name is near its buy target right now.
        </p>
      </div>
    );
  }

  return (
    <div className="border-y border-glass-tint/10">
      <div className="strip relative overflow-hidden" style={{ height: STRIP_HEIGHT }}>
        {/* The water: light above the line, the zone below it. */}
        <div
          aria-hidden="true"
          className="strip-above absolute inset-x-0 top-0"
          style={{ height: `${model.linePct}%` }}
        />
        <div
          aria-hidden="true"
          className="strip-below absolute inset-x-0 bottom-0"
          style={{ height: `${100 - model.linePct}%` }}
        />

        {/* Depth guides. */}
        {model.guides.map(([pct, label]) => (
          <div
            key={label}
            aria-hidden="true"
            className="absolute inset-x-0"
            style={{ top: `${pct}%` }}
          >
            <div className="border-t border-glass-tint/6" />
            <span className="font-data absolute left-3 top-0.5 text-[9px] text-sounding/45">
              {label}
            </span>
          </div>
        ))}

        {/* The line itself. */}
        <div
          aria-hidden="true"
          className="strip-line absolute inset-x-0"
          style={{ top: `${model.linePct}%` }}
        >
          <span className="font-data absolute right-3 -top-4 text-[9px] tracking-[0.2em] text-jade/80">
            BUY LINE
          </span>
        </div>

        {/* Sounding lines + markers. */}
        {model.markers.map((marker, i) => (
          <div key={marker.item.watchlist_id} className="contents">
            <div
              aria-hidden="true"
              className="absolute w-px bg-sounding/25"
              style={{
                left: `${marker.xPct}%`,
                top: `${Math.min(model.linePct, marker.yPct)}%`,
                height: `${Math.abs(marker.yPct - model.linePct)}%`,
              }}
            />
            <MarkerView marker={marker} index={i} />
          </div>
        ))}
      </div>

      {/* Seabed footer. */}
      <div className="flex flex-wrap items-baseline justify-between gap-2 border-t border-glass-tint/10 px-6 py-3">
        <span className="text-xs text-sounding">
          <span aria-hidden="true">● </span>clear to act ·{" "}
          <span aria-hidden="true">○ </span>held by an event or gate
        </span>
        {deeper.count > 0 ? (
          <span className="font-data text-[11px] text-sounding">
            {deeper.count} more below the shown range —{" "}
            {deeper.names
              .map((n) => `${n.ticker} ${formatPct(n.distance_from_buy_pct, 0)}`)
              .join(" · ")}{" "}
            <Link to="/watchlist" className="text-jade underline-offset-4 hover:underline">
              full watchlist →
            </Link>
          </span>
        ) : (
          <Link
            to="/watchlist"
            className="text-xs text-jade underline-offset-4 hover:underline"
          >
            full watchlist →
          </Link>
        )}
      </div>
    </div>
  );
}
