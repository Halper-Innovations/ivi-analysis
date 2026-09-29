/**
 * Gauge Lab — the Phase 0 acceptance surface: the Depth Gauge at all three
 * sizes from fixtures, the frost mechanic, and the reduced-motion contract.
 */

import { GlassPanel } from "../components/GlassPanel";
import { DepthGauge } from "../components/DepthGauge";
import { DepthPip } from "../components/DepthPip";
import { APPROACHING, FROSTED, IN_ZONE, SUSPECT } from "../fixtures/gauges";
import { distanceToZonePct, formatMoney } from "../lib/gauge";

function Distance({ price, target }: { price: number | null; target: number | null }) {
  const pct = distanceToZonePct(price, target);
  if (pct === null) return null;
  const inZone = pct <= 0;
  return (
    <span className={`font-data text-xs ${inZone ? "text-jade font-semibold" : "text-sounding"}`}>
      {inZone
        ? `${Math.abs(pct).toFixed(1)}% below buy target`
        : `${pct.toFixed(1)}% above the buy zone`}
    </span>
  );
}

export function GaugeLab() {
  return (
    <div className="max-w-5xl">
      <p className="text-[11px] uppercase tracking-[0.25em] text-sounding">
        valuation gauge reference
      </p>
      <h1 className="font-display mt-1 text-3xl text-moonlight">Gauge Lab</h1>
      <p className="mt-3 max-w-2xl text-sm leading-relaxed text-sounding">
        Each horizontal line is a fair-value estimate (DCF, EPV, Graham). The shaded band below
        the buy target is the buy zone — the waterline. The dot is the latest price, trailed by
        its last 30 days. A price older than the 96-hour staleness limit renders dimmed (frosted)
        until you hover it. All cards on this page render sample data — the tickers are fictional.
      </p>

      <div className="mt-8 grid grid-cols-1 gap-6 lg:grid-cols-[auto_1fr]">
        {/* Hero */}
        <GlassPanel condenseIndex={0} className="p-6 lg:self-start">
          <div className="mb-2 flex items-baseline justify-between gap-6">
            <div className="flex items-baseline gap-2">
              <h2 className="font-display text-xl text-moonlight">{APPROACHING.ticker}</h2>
              <span className="text-xs text-sounding/80">sample data</span>
            </div>
            <Distance price={APPROACHING.price} target={APPROACHING.buyTarget} />
          </div>
          <p className="mb-3 text-xs text-sounding">
            hero · hover a fair-value line for its method
          </p>
          <DepthGauge input={APPROACHING} size="hero" />
        </GlassPanel>

        <div className="flex flex-col gap-6">
          {/* Cards: in-zone (nacre) + frosted */}
          <div className="flex flex-wrap gap-6">
            <GlassPanel condenseIndex={1} className="p-5">
              <div className="mb-2 flex items-baseline justify-between gap-4">
                <div className="flex items-baseline gap-2">
                  <h3 className="font-display text-xl text-moonlight">{IN_ZONE.ticker}</h3>
                  <span className="text-xs text-sounding/80">sample data</span>
                </div>
                <Distance price={IN_ZONE.price} target={IN_ZONE.buyTarget} />
              </div>
              <p className="mb-2 text-xs text-sounding">card · price inside the buy zone</p>
              <DepthGauge input={IN_ZONE} size="card" />
            </GlassPanel>

            <GlassPanel condenseIndex={2} className="p-5">
              <div className="mb-2 flex items-baseline justify-between gap-4">
                <div className="flex items-baseline gap-2">
                  <h3 className="font-display text-xl text-moonlight">{FROSTED.ticker}</h3>
                  <span className="text-xs text-sounding/80">sample data</span>
                </div>
                <span className="font-data text-xs text-sounding">price 8d old</span>
              </div>
              <p className="mb-2 text-xs text-sounding">
                card · price stale (older than 96h) — dimmed
              </p>
              <DepthGauge input={FROSTED} size="card" />
            </GlassPanel>
          </div>

          {/* Depth-pip row strip */}
          <GlassPanel condenseIndex={3} className="p-5">
            <div className="mb-3 flex items-baseline justify-between gap-3">
              <p className="text-xs text-sounding">
                row gauge · the compact form used in watchlist tables
              </p>
              <span className="text-xs text-sounding/80">sample data</span>
            </div>
            <ul className="divide-y divide-glass-tint/10">
              {[APPROACHING, IN_ZONE, FROSTED, SUSPECT].map((fixture) => (
                <li key={fixture.ticker} className="flex items-center gap-4 py-2">
                  <span className="font-data w-14 text-sm text-moonlight">{fixture.ticker}</span>
                  <DepthPip input={fixture} />
                  <span className="font-data w-20 text-right text-sm text-moonlight">
                    {fixture.price !== null ? formatMoney(fixture.price) : "—"}
                  </span>
                  <span className="font-data w-24 text-right text-xs text-sounding">
                    {fixture.buyTarget !== null ? `buy ${formatMoney(fixture.buyTarget)}` : ""}
                  </span>
                  <span className="flex-1 text-right">
                    {fixture.priceSuspect ? (
                      <span className="font-data text-xs text-rose">PRICE_DATA_SUSPECT</span>
                    ) : (
                      <Distance price={fixture.price} target={fixture.buyTarget} />
                    )}
                  </span>
                </li>
              ))}
            </ul>
          </GlassPanel>

          <GlassPanel condenseIndex={4} className="p-5">
            <p className="text-xs leading-relaxed text-sounding">
              <span className="text-moonlight">Reduced motion:</span> with{" "}
              <span className="font-data">prefers-reduced-motion</span> set, the background drift
              pauses, panels appear instantly, shelves render without the draw-in, the marker drops
              without the spring, and the ripple and nacre sweep are static. Nothing loops.
            </p>
          </GlassPanel>
        </div>
      </div>
    </div>
  );
}
