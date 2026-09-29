/**
 * Company — conviction depth. Masthead + hero Depth Gauge, then Valuation
 * (one card per method, estimate evolution) and Fundamentals (FY sparkline
 * grid; quarterly/TTM basis from the quarterly fact chain — computed, never faked).
 *
 * Chart discipline: the evolution chart's three method hues are a validated
 * categorical trio (CVD-safe on the abyss surface); identity is never
 * color-alone — every line ends in its own label, and pre-hardening points
 * render hollow with a labeled regime divider.
 */

import { useQuery } from "@tanstack/react-query";
import { Link, useNavigate, useParams, useSearch } from "@tanstack/react-router";
import { scaleLinear, scaleTime } from "d3-scale";
import { useEffect, useMemo, useState } from "react";

import { CopyChip, GradeWord, StatusChip } from "../components/chips";
import { DepthGauge } from "../components/DepthGauge";
import { Section } from "../components/GlassPanel";
import type {
  CompanyResponse,
  EvolutionPoint,
  FundamentalsFy,
  FundamentalsTtm,
  MethodCard,
} from "../lib/api";
import {
  HttpError,
  OfflineError,
  fetchCompany,
  fetchCompanyDecisions,
  fetchCompanyDossier,
  fetchCompanyResearch,
  fetchFundamentals,
} from "../lib/api";
import { DecisionsTab, DossierTab, ResearchTab } from "./CompanyDepth";
import {
  clearToAct,
  formatAge,
  formatMillions,
  formatMoneyCompact,
  formatPct,
  formatRawCompact,
  hoursSince,
  humanizeEventPending,
  humanizeSignalCode,
  humanizeToken,
  marginOfSafetyPct,
} from "../lib/floor";
import { formatMoney, type GaugeInput } from "../lib/gauge";

/** Validated categorical trio for method identity on the abyss surface
 *  (validate_palette.js: all five checks pass on #0B1626). */
const METHOD_HUES: Record<string, string> = {
  dcf: "#3987e5",
  epv: "#d55181",
  graham: "#c98500",
};

const HARDENING_CUTOFF = new Date("2026-07-16T00:00:00Z");

function heroGaugeInput(data: CompanyResponse, now: Date): GaugeInput {
  return {
    ticker: data.ticker,
    shelves: data.shelves.map((shelf) => ({
      method: shelf.method,
      label: shelf.label,
      value: shelf.value ?? undefined,
      low: shelf.low ?? undefined,
      base: shelf.base ?? undefined,
      high: shelf.high ?? undefined,
      emphasized: shelf.emphasized,
    })),
    buyTarget: data.watchlist?.buy_price_target ?? null,
    price: data.prices.latest ?? (data.as_of ? null : data.watchlist?.latest_price ?? null),
    wake: data.prices.wake,
    priceAgeHours:
      data.prices.age_hours ??
      (data.watchlist && !data.as_of
        ? hoursSince(data.watchlist.latest_price_checked_at, now)
        : null),
    priceSuspect: data.watchlist?.status === "PRICE_DATA_SUSPECT",
  };
}

/* ------------------------------------------------------------------ */
/* Estimate evolution chart                                            */
/* ------------------------------------------------------------------ */

interface EvolutionHover {
  x: number;
  date: string;
  values: { method: string; value: number; preHardening: boolean }[];
}

function EvolutionChart({ points }: { points: EvolutionPoint[] }) {
  const [hover, setHover] = useState<EvolutionHover | null>(null);
  const width = 640;
  const height = 230;
  const pad = { top: 14, right: 96, bottom: 24, left: 46 };

  const model = useMemo(() => {
    const parsed = points
      .map((p) => ({ ...p, date: new Date(`${p.as_of_date}T00:00:00Z`) }))
      .filter((p) => !Number.isNaN(p.date.getTime()));
    if (parsed.length === 0) return null;
    const dates = parsed.map((p) => p.date.getTime());
    const values = parsed.map((p) => p.value);
    const x = scaleTime()
      .domain([new Date(Math.min(...dates)), new Date(Math.max(...dates))])
      .range([pad.left, width - pad.right]);
    const y = scaleLinear()
      .domain([Math.min(...values), Math.max(...values)])
      .nice()
      .range([height - pad.bottom, pad.top]);
    const series = new Map<string, typeof parsed>();
    for (const p of parsed) {
      if (!series.has(p.method)) series.set(p.method, []);
      series.get(p.method)!.push(p);
    }
    for (const list of series.values()) list.sort((a, b) => a.date.getTime() - b.date.getTime());
    const uniqueDates = [...new Set(parsed.map((p) => p.as_of_date))].sort();
    return { x, y, series, uniqueDates, minDate: Math.min(...dates), maxDate: Math.max(...dates) };
  }, [points]);

  if (!model) {
    return <p className="text-sm text-sounding">No estimate history yet.</p>;
  }
  const { x, y, series, uniqueDates } = model;
  const hardeningX =
    HARDENING_CUTOFF.getTime() >= model.minDate && HARDENING_CUTOFF.getTime() <= model.maxDate
      ? x(HARDENING_CUTOFF)
      : null;

  const onMove = (event: React.MouseEvent<SVGSVGElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const px = ((event.clientX - rect.left) / rect.width) * width;
    let best: string | null = null;
    let bestDx = Infinity;
    for (const d of uniqueDates) {
      const dx = Math.abs(x(new Date(`${d}T00:00:00Z`)) - px);
      if (dx < bestDx) {
        bestDx = dx;
        best = d;
      }
    }
    if (best === null) return setHover(null);
    const values = [...series.values()]
      .map((pts) => pts.find((p) => p.as_of_date === best))
      .filter((p): p is NonNullable<typeof p> => Boolean(p))
      .map((p) => ({ method: p.method, value: p.value, preHardening: p.pre_hardening }));
    setHover({ x: x(new Date(`${best}T00:00:00Z`)), date: best, values });
  };

  return (
    <div className="relative">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="w-full"
        role="img"
        aria-label="Fair-value estimates over time by method"
        onMouseMove={onMove}
        onMouseLeave={() => setHover(null)}
      >
        {y.ticks(4).map((tick) => (
          <g key={tick}>
            <line
              x1={pad.left}
              x2={width - pad.right}
              y1={y(tick)}
              y2={y(tick)}
              stroke="var(--color-glass-tint)"
              strokeOpacity={0.07}
            />
            <text x={pad.left - 6} y={y(tick) + 3} textAnchor="end" fontSize={9} fill="var(--color-sounding)" className="font-data">
              {tick}
            </text>
          </g>
        ))}
        {hardeningX !== null && (
          <g>
            <line
              x1={hardeningX}
              x2={hardeningX}
              y1={pad.top}
              y2={height - pad.bottom}
              stroke="var(--color-sounding)"
              strokeOpacity={0.35}
              strokeDasharray="3 4"
            />
            <text x={hardeningX + 4} y={pad.top + 8} fontSize={8} fill="var(--color-sounding)" className="font-data">
              writer hardened 07-16
            </text>
          </g>
        )}
        {hover && (
          <line
            x1={hover.x}
            x2={hover.x}
            y1={pad.top}
            y2={height - pad.bottom}
            stroke="var(--color-moonlight)"
            strokeOpacity={0.25}
          />
        )}
        {[...series.entries()].map(([method, pts]) => {
          const hue = METHOD_HUES[method] ?? "var(--color-glass-tint)";
          const path = pts
            .map((p, i) => `${i === 0 ? "M" : "L"}${x(p.date).toFixed(1)},${y(p.value).toFixed(1)}`)
            .join(" ");
          const last = pts[pts.length - 1];
          return (
            <g key={method}>
              <path d={path} fill="none" stroke={hue} strokeWidth={2} strokeLinejoin="round" />
              {pts.map((p) => (
                <circle
                  key={p.as_of_date}
                  cx={x(p.date)}
                  cy={y(p.value)}
                  r={3}
                  fill={p.pre_hardening ? "var(--color-deepwater)" : hue}
                  stroke={hue}
                  strokeWidth={1.5}
                >
                  <title>
                    {method} {p.as_of_date}: {formatMoney(p.value)}
                    {p.pre_hardening ? " (pre-hardening)" : ""}
                  </title>
                </circle>
              ))}
              {/* Direct label at the line end — identity is never color-alone. */}
              <text
                x={x(last.date) + 7}
                y={y(last.value) + 3}
                fontSize={10}
                fill="var(--color-moonlight)"
                className="font-data"
              >
                {method} {formatMoney(last.value)}
              </text>
            </g>
          );
        })}
      </svg>
      {hover && (
        <div
          className="glass pointer-events-none absolute z-10 px-3 py-2 text-xs"
          style={{ left: `${(hover.x / width) * 100}%`, top: 0, transform: "translateX(-50%)" }}
        >
          <p className="font-data text-[10px] text-sounding">{hover.date}</p>
          {hover.values.map((v) => (
            <p key={v.method} className="font-data text-moonlight">
              <span
                className="mr-1.5 inline-block h-2 w-2 rounded-sm"
                style={{ background: METHOD_HUES[v.method] ?? "var(--color-glass-tint)" }}
              />
              {v.method} {formatMoney(v.value)}
              {v.preHardening && <span className="text-sounding"> · pre-hardening</span>}
            </p>
          ))}
        </div>
      )}
      <div className="mt-1 flex flex-wrap gap-4">
        {[...series.keys()].map((method) => (
          <span key={method} className="font-data flex items-center gap-1.5 text-[10px] text-sounding">
            <span className="inline-block h-2 w-2 rounded-sm" style={{ background: METHOD_HUES[method] ?? "var(--color-glass-tint)" }} />
            {method}
          </span>
        ))}
        <span className="font-data text-[10px] text-sounding">○ pre-hardening estimate</span>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Valuation method cards                                              */
/* ------------------------------------------------------------------ */

/** WACC values arrive as fractions; speak them as percents. */
function formatWacc(value: number | null): string {
  if (value === null) return "—";
  return `${(value * 100).toFixed(1)}%`;
}

function MethodCardView({ card }: { card: MethodCard }) {
  const fv = card.fair_value;
  return (
    <div className="glass rounded-xl p-4">
      <div className="flex items-baseline justify-between gap-2">
        <span className="min-w-0 truncate text-sm text-moonlight" title={card.method}>
          {humanizeToken(card.method)}
        </span>
        <StatusChip value={card.quality_gate_verdict} />
      </div>
      {(card.confidence_class || card.as_of_date) && (
        <div className="mt-0.5 flex items-baseline justify-between gap-2 text-[10px] text-sounding">
          <span className="lowercase">
            {card.confidence_class ? `${humanizeToken(card.confidence_class)} confidence` : ""}
          </span>
          <span className="font-data">{card.as_of_date}</span>
        </div>
      )}
      <div className="font-data mt-2.5 text-xl text-moonlight">
        {fv.kind === "band" && fv.base !== null ? (
          <>
            {formatMoney(fv.base)}{" "}
            <span className="text-xs text-sounding">
              {fv.low !== null && fv.high !== null
                ? `${formatMoney(fv.low)} – ${formatMoney(fv.high)}`
                : ""}
            </span>
          </>
        ) : fv.kind === "line" && fv.value !== null ? (
          formatMoney(fv.value)
        ) : fv.kind === "not_a_price" ? (
          <span className="voice text-sm" title={`${fv.value ?? ""}`}>
            not a price
            {fv.value !== null ? ` (${formatMoney(fv.value)})` : ""}
            {card.status ? ` · ${humanizeToken(card.status)}` : ""}
          </span>
        ) : (
          <span className="voice text-sm">
            {card.status === "METHOD_INSUFFICIENT_DATA" ? "insufficient data" : "no fair value"}
          </span>
        )}
      </div>
      {card.wacc && (
        <p
          className="font-data mt-1.5 text-[10px] text-sounding"
          title={card.wacc.adjustments.map((a) => `${a.code} ${a.delta ?? ""}: ${a.reason}`).join("\n")}
        >
          wacc {formatWacc(card.wacc.baseline_wacc)} → {formatWacc(card.wacc.adjusted_wacc)}
          {card.wacc.adjustments.length > 0 &&
            ` (${card.wacc.adjustments.length} adjustment${card.wacc.adjustments.length === 1 ? "" : "s"})`}
        </p>
      )}
      {card.headwinds.length > 0 && (
        <p className="mt-2 text-[11px] leading-relaxed text-rose/85" title={card.headwinds.join(", ")}>
          <span aria-hidden="true">▾ </span>
          {card.headwinds.map(humanizeSignalCode).join(" · ")}
        </p>
      )}
      {card.supports.length > 0 && (
        <p className="mt-1 text-[11px] leading-relaxed text-jade/85" title={card.supports.join(", ")}>
          <span aria-hidden="true">▴ </span>
          {card.supports.map(humanizeSignalCode).join(" · ")}
        </p>
      )}
      {card.flags.length > 0 && (
        <p className="mt-1.5 truncate text-[10px] text-sounding" title={card.flags.join(", ")}>
          {card.flags.map(humanizeToken).join(" · ")}
        </p>
      )}
      {Object.keys(card.extras).length > 0 && (
        <dl className="font-data mt-2.5 grid grid-cols-2 gap-x-3 gap-y-0.5 text-[10px]">
          {Object.entries(card.extras).map(([key, value]) => (
            <div key={key} className="contents">
              <dt className="text-sounding" title={key}>
                {humanizeToken(key)}
              </dt>
              <dd className="text-right text-moonlight">
                {typeof value === "number" ? formatRawCompact(value, false) : String(value)}
              </dd>
            </div>
          ))}
        </dl>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Fundamentals                                                        */
/* ------------------------------------------------------------------ */

function Sparkline({ values, hue = "var(--color-moonlight)" }: { values: (number | null)[]; hue?: string }) {
  const width = 96;
  const height = 26;
  const finite = values.filter((v): v is number => v !== null && Number.isFinite(v));
  if (finite.length < 2) {
    return <span className="font-data text-[10px] text-sounding">not enough data</span>;
  }
  const lo = Math.min(...finite);
  const hi = Math.max(...finite);
  const y = (v: number) => (hi === lo ? height / 2 : 3 + (height - 6) * (1 - (v - lo) / (hi - lo)));
  const step = width / (values.length - 1);
  let d = "";
  values.forEach((v, i) => {
    if (v === null) return;
    const cmd = d === "" || values[i - 1] === null ? "M" : "L";
    d += `${cmd}${(i * step).toFixed(1)},${y(v).toFixed(1)}`;
  });
  const zeroInRange = lo < 0 && hi > 0;
  return (
    <svg width={width} height={height} aria-hidden="true">
      {zeroInRange && (
        <line x1={0} x2={width} y1={y(0)} y2={y(0)} stroke="var(--color-sounding)" strokeOpacity={0.3} strokeDasharray="2 3" />
      )}
      <path d={d} fill="none" stroke={hue} strokeOpacity={0.75} strokeWidth={1.5} />
    </svg>
  );
}

interface FyMetric {
  key: string;
  label: string;
  source: "series" | "derived";
  format: (v: number | null) => string;
}

const FY_METRICS: FyMetric[] = [
  { key: "revenue", label: "revenue", source: "series", format: formatMillions },
  { key: "gross_margin", label: "gross margin", source: "derived", format: (v) => (v === null ? "—" : `${(v * 100).toFixed(1)}%`) },
  { key: "operating_margin", label: "op margin", source: "derived", format: (v) => (v === null ? "—" : `${(v * 100).toFixed(1)}%`) },
  { key: "net_margin", label: "net margin", source: "derived", format: (v) => (v === null ? "—" : `${(v * 100).toFixed(1)}%`) },
  { key: "cfo", label: "CFO", source: "series", format: formatMillions },
  { key: "capex", label: "capex", source: "series", format: formatMillions },
  { key: "fcf", label: "FCF (CFO−capex)", source: "derived", format: formatMillions },
  { key: "sbc", label: "SBC", source: "series", format: formatMillions },
  { key: "net_debt", label: "net debt", source: "derived", format: formatMillions },
  { key: "equity", label: "equity", source: "series", format: formatMillions },
  { key: "roe", label: "ROE", source: "derived", format: (v) => (v === null ? "—" : `${(v * 100).toFixed(1)}%`) },
  { key: "shares_outstanding", label: "shares (M)", source: "series", format: (v) => (v === null ? "—" : v.toFixed(1)) },
];

function FyGrid({ data }: { data: FundamentalsFy }) {
  if (data.fiscal_years.length === 0) {
    return <p className="text-sm text-sounding">No cached FY facts for this ticker.</p>;
  }
  const dilution = data.derived.shares_dilution_pct ?? [];
  return (
    <div>
      <p className="font-data mb-3 text-[10px] text-sounding">
        FY{data.fiscal_years[0]} – FY{data.fiscal_years[data.fiscal_years.length - 1]} · USD
        millions · from cached SEC companyfacts
      </p>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-3">
        {FY_METRICS.map((metric) => {
          const values =
            metric.source === "series" ? data.series[metric.key] : data.derived[metric.key];
          if (!values) return null;
          const latest = [...values].reverse().find((v) => v !== null) ?? null;
          return (
            <div key={metric.key} className="glass flex items-center justify-between gap-3 rounded-xl p-3">
              <div>
                <p className="eyebrow text-[10px]">{metric.label}</p>
                <p className="font-data mt-1 text-base text-moonlight">{metric.format(latest)}</p>
              </div>
              <Sparkline values={values} />
            </div>
          );
        })}
      </div>
      {dilution.some((v) => v !== null) && (
        <div className="mt-4">
          <p className="eyebrow text-[10px]">dilution — shares outstanding YoY</p>
          <div className="mt-1.5 flex flex-wrap gap-1.5">
            {data.fiscal_years.map((year, i) => {
              const v = dilution[i];
              if (v === null || v === undefined) return null;
              const tone =
                v > 2 ? "border-rose/30 text-rose" : v < 0 ? "border-jade/30 text-jade" : "border-glass-tint/20 text-sounding";
              return (
                <span key={year} className={`font-data rounded border px-1.5 py-0.5 text-[10px] ${tone}`}>
                  FY{year} {formatPct(v)}
                </span>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

const TTM_BASIS_LABEL: Record<string, string> = {
  ttm: "TTM (4 discrete quarters)",
  fy: "FY fallback",
  quarter_end: "newest quarter-end",
};

function TtmTable({ data }: { data: FundamentalsTtm }) {
  if (!data.available) {
    return (
      <p className="text-sm text-sounding">
        No quarterly facts for this ticker — the TTM basis only exists where the
        quarterly fact chain has been ingested.
      </p>
    );
  }
  const entries = Object.entries(data.values);
  return (
    <div>
      <p className="font-data mb-2 text-[10px] text-sounding">
        deciding basis per concept — TTM is computed from discrete quarters, never faked from FY
      </p>
      <table className="w-full border-collapse text-left">
        <thead>
          <tr className="text-[10px] uppercase tracking-wider text-sounding">
            <th className="py-1.5 pr-3 font-medium">concept</th>
            <th className="py-1.5 pr-3 text-right font-medium">value</th>
            <th className="py-1.5 pr-3 font-medium">basis</th>
            <th className="py-1.5 pr-3 font-medium">through</th>
            <th className="py-1.5 pr-0 font-medium">reason</th>
          </tr>
        </thead>
        <tbody>
          {entries.map(([concept, value]) => (
            <tr key={concept} className="border-t border-glass-tint/8 text-xs">
              <td className="font-data py-1.5 pr-3 text-moonlight">{concept}</td>
              <td className="font-data py-1.5 pr-3 text-right text-moonlight">
                {formatRawCompact(value.value, concept !== "shares_outstanding")}
              </td>
              <td className={`font-data py-1.5 pr-3 text-[10px] ${value.basis === "ttm" ? "text-moonlight" : "text-sounding"}`}>
                {TTM_BASIS_LABEL[value.basis] ?? value.basis}
              </td>
              <td className="font-data py-1.5 pr-3 text-[10px] text-sounding">{value.period_end}</td>
              <td className="font-data py-1.5 pr-0 text-[9px] text-sounding">{value.reason}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* The page                                                            */
/* ------------------------------------------------------------------ */

export type CompanyTab =
  | "valuation"
  | "fundamentals"
  | "research"
  | "dossier"
  | "decisions";

const TIME_MACHINE_DISABLED_TABS: readonly CompanyTab[] = [
  "research",
  "dossier",
  "decisions",
];

function isTimeMachineDisabledTab(tab: CompanyTab): boolean {
  return TIME_MACHINE_DISABLED_TABS.includes(tab);
}

function localIsoDate(date: Date): string {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

export function Company() {
  const { ticker } = useParams({ from: "/company/$ticker" });
  const search = useSearch({ from: "/company/$ticker" });
  const navigate = useNavigate();
  const tab: CompanyTab = search.tab ?? "valuation";
  const asOf = search.as_of;
  const basis: "fy" | "ttm" = asOf ? "fy" : search.basis ?? "fy";
  const setTab = (next: CompanyTab) =>
    void navigate({
      to: "/company/$ticker",
      params: { ticker },
      search: {
        tab: next === "valuation" ? undefined : next,
        basis: search.basis,
        as_of: asOf,
      },
      replace: true,
    });
  const setBasis = (next: "fy" | "ttm") =>
    void navigate({
      to: "/company/$ticker",
      params: { ticker },
      search: {
        tab: search.tab,
        basis: next === "fy" ? undefined : next,
        as_of: asOf,
      },
      replace: true,
    });
  const setAsOf = (next?: string) =>
    void navigate({
      to: "/company/$ticker",
      params: { ticker },
      search: {
        tab: next && isTimeMachineDisabledTab(tab) ? undefined : search.tab,
        basis: next ? undefined : search.basis,
        as_of: next || undefined,
      },
      replace: true,
    });
  const company = useQuery({
    queryKey: ["company", ticker, asOf ?? null],
    queryFn: () => fetchCompany(ticker, asOf),
    retry: 1,
  });
  const fundamentals = useQuery({
    queryKey: ["fundamentals", ticker, basis, asOf ?? null],
    queryFn: () => fetchFundamentals(ticker, basis, asOf),
    enabled: tab === "fundamentals",
  });
  const research = useQuery({
    queryKey: ["company-research", ticker],
    queryFn: () => fetchCompanyResearch(ticker),
    enabled: tab === "research" && !asOf,
  });
  const dossier = useQuery({
    queryKey: ["company-dossier", ticker],
    queryFn: () => fetchCompanyDossier(ticker),
    enabled: tab === "dossier" && !asOf,
  });
  const decisions = useQuery({
    queryKey: ["company-decisions", ticker],
    queryFn: () => fetchCompanyDecisions(ticker),
    enabled: tab === "decisions" && !asOf,
  });
  const now = new Date();

  useEffect(() => {
    if (!asOf || (!isTimeMachineDisabledTab(tab) && search.basis !== "ttm")) return;
    void navigate({
      to: "/company/$ticker",
      params: { ticker },
      search: {
        tab: isTimeMachineDisabledTab(tab) ? undefined : search.tab,
        basis: undefined,
        as_of: asOf,
      },
      replace: true,
    });
  }, [asOf, navigate, search.basis, search.tab, tab, ticker]);

  if (company.error instanceof OfflineError) {
    return (
      <Section className="max-w-xl p-8">
        <h1 className="font-display text-2xl text-moonlight">IVI offline</h1>
        <p className="font-data mt-3 text-sm text-rose">{company.error.precondition}</p>
      </Section>
    );
  }
  if (company.isError) {
    const notFound = company.error instanceof HttpError && company.error.status === 404;
    return (
      <Section className="max-w-xl p-8">
        <h1 className="font-display text-2xl text-moonlight">{ticker.toUpperCase()}</h1>
        <p className="mt-3 text-sm text-sounding">
          {notFound
            ? "Not known to the platform — not on the watchlist, never valued, no cached facts."
            : `Couldn't load ${ticker.toUpperCase()} — the books may be briefly offline. Retry.`}
        </p>
        <Link to="/watchlist" className="mt-4 inline-block text-xs text-jade underline-offset-4 hover:underline">
          back to the floor →
        </Link>
      </Section>
    );
  }
  const data = company.data;
  if (!data) {
    return <p className="text-sm text-sounding">loading…</p>;
  }

  const row = data.watchlist;
  const price = data.prices.latest ?? (asOf ? null : row?.latest_price ?? null);
  const mos = marginOfSafetyPct(price, data.profile?.valuation_anchor_value ?? null);
  const gauge = heroGaugeInput(data, now);
  const submerged =
    row?.distance_from_buy_pct !== null &&
    row?.distance_from_buy_pct !== undefined &&
    row.distance_from_buy_pct <= 0;
  // Jade marks in-zone AND clear-to-act — a held name stays neutral.
  const inZoneClear = submerged && clearToAct(row?.presented_status);
  const earliestEvolutionDate = data.evolution.reduce<string | undefined>(
    (earliest, point) =>
      earliest === undefined || point.as_of_date < earliest ? point.as_of_date : earliest,
    undefined,
  );
  const today = localIsoDate(now);

  return (
    <div className="max-w-6xl">
      {/* Masthead: identity on the left, the price as the one display
          number on the right. */}
      <div className={`relative rounded-xl ${asOf ? "frosted frost-1" : ""}`}>
        <div className={asOf ? "frost-content" : ""}>
          <div className="flex flex-wrap items-end justify-between gap-x-8 gap-y-3">
            <div>
              <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1.5">
                <h1 className="font-display text-5xl text-moonlight">{data.ticker}</h1>
                <StatusChip
                  value={row?.presented_status ?? null}
                  title={row?.status_reason ?? undefined}
                />
                <GradeWord value={row?.conviction_grade ?? null} />
              </div>
              <p className="mt-2 text-xs text-sounding">
                {row?.source_sector ? humanizeToken(row.source_sector) : "—"}
                {row?.cap_band_label && row.cap_band_label !== "UNKNOWN_CAP"
                  ? ` · ${humanizeToken(row.cap_band_label)} cap`
                  : ""}
                {row?.market_cap_mm != null ? ` · ${formatMillions(row.market_cap_mm)}` : ""}
                {row?.confidence ? ` · ${row.confidence.toLowerCase()} confidence` : ""}
                {row?.cheapness && row.cheapness !== "n/a" && (
                  <span className="text-amber/90" title="known reasons it may be cheap">
                    {" "}
                    · {row.cheapness}
                  </span>
                )}
              </p>
            </div>
            <div className="text-right">
              <div
                className={`display-num text-5xl leading-none ${
                  inZoneClear ? "text-jade" : "text-moonlight"
                }`}
              >
                {formatMoneyCompact(price)}
              </div>
              <p className="font-data mt-1.5 text-[11px] text-sounding">
                {formatAge(gauge.priceAgeHours)} old ·{" "}
                <span className={inZoneClear ? "text-jade" : "text-sounding"}>
                  {formatPct(row?.distance_from_buy_pct ?? null)} from buy
                </span>
              </p>
            </div>
          </div>

          <div className="font-data mt-3 flex flex-wrap gap-x-5 gap-y-1 border-t border-glass-tint/10 pt-3 text-xs text-sounding">
            <span>
              target{" "}
              <span className="text-moonlight">
                {formatMoneyCompact(row?.buy_price_target ?? null)}
              </span>
            </span>
            <span>
              anchor{" "}
              <span className="text-moonlight">
                {data.profile?.valuation_anchor_method ?? "—"}{" "}
                {data.profile?.valuation_anchor_value !== null &&
                data.profile?.valuation_anchor_value !== undefined
                  ? formatMoneyCompact(data.profile.valuation_anchor_value)
                  : ""}
              </span>
            </span>
            <span>
              MoS @ price <span className="text-moonlight">{formatPct(mos)}</span>
            </span>
            {row?.event_pending && (
              <span className="text-amber" title={row.event_pending}>
                held: {humanizeEventPending(row.event_pending)}
              </span>
            )}
          </div>
        </div>
      </div>

      <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-[auto_1fr]">
        {/* Hero gauge */}
        <Section condenseIndex={0} className="p-6 lg:self-start">
          <p className="voice mb-3 text-sm">
            valuation gauge — hover a fair-value line for its method
          </p>
          <DepthGauge input={gauge} size="hero" />
        </Section>

        <div className="flex min-w-0 flex-col gap-5">
          {/* Thesis & falsifiers */}
          {(data.profile?.thesis_text || (row?.falsifiers.length ?? 0) > 0) && (
            <Section condenseIndex={1} className="p-6">
              {data.profile?.thesis_text && (
                <>
                  <p className="eyebrow">thesis</p>
                  <p className="mt-2 max-w-[62ch] text-[15px] leading-relaxed text-moonlight/95">
                    {data.profile.thesis_text}
                  </p>
                </>
              )}
              <div className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2">
                {(row?.falsifiers.length ?? 0) > 0 && (
                  <div>
                    <p className="eyebrow text-amber/90">falsifiers</p>
                    <ul className="mt-1.5 space-y-1 pl-4 text-xs leading-relaxed text-sounding">
                      {row!.falsifiers.map((f) => (
                        <li key={f} className="list-disc">{f}</li>
                      ))}
                    </ul>
                  </div>
                )}
                {(data.profile?.key_risks.length ?? 0) > 0 && (
                  <div>
                    <p className="eyebrow text-rose/90">key risks</p>
                    <ul className="mt-1.5 space-y-1 pl-4 text-xs leading-relaxed text-sounding">
                      {data.profile!.key_risks.map((r) => (
                        <li key={r} className="list-disc">{r}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </Section>
          )}

          {/* Tabs */}
          <Section condenseIndex={2} className="p-5">
            <div className="mb-3 flex flex-wrap items-end justify-between gap-3">
              <div>
                <label htmlFor="company-as-of" className="eyebrow block text-[10px]">
                  time machine
                </label>
                <input
                  id="company-as-of"
                  type="date"
                  value={asOf ?? ""}
                  min={earliestEvolutionDate}
                  max={today}
                  onChange={(event) => setAsOf(event.currentTarget.value || undefined)}
                  className="font-data mt-1 rounded-md border border-glass-tint/20 bg-deepwater/60 px-2 py-1 text-xs text-moonlight"
                />
              </div>
              {asOf && (
                <button
                  type="button"
                  onClick={() => setAsOf()}
                  className="rounded-md border border-glass-tint/20 px-3 py-1 text-xs text-moonlight hover:bg-glass-tint/10"
                >
                  Return to present
                </button>
              )}
            </div>
            {asOf && (
              <div
                role="status"
                className="mb-3 rounded-lg border border-glass-tint/20 bg-glass-tint/10 px-3 py-2 text-xs text-moonlight"
              >
                Record as of {asOf} — filings and valuations visible on that date only.
              </div>
            )}
            <div className="flex flex-wrap items-center gap-1 border-b border-glass-tint/10 pb-3">
              {(
                ["valuation", "fundamentals", "research", "dossier", "decisions"] as const
              ).map((t) => {
                const timeMachineDisabled = Boolean(asOf && isTimeMachineDisabledTab(t));
                return (
                  <button
                    key={t}
                    type="button"
                    onClick={() => setTab(t)}
                    disabled={timeMachineDisabled}
                    aria-disabled={timeMachineDisabled}
                    title={timeMachineDisabled ? "Not yet time-machine aware" : undefined}
                    className={`rounded-lg px-3 py-1 text-xs capitalize transition-colors disabled:cursor-not-allowed ${
                      tab === t
                        ? "bg-glass-tint/15 text-moonlight"
                        : "text-sounding hover:text-moonlight disabled:hover:text-sounding"
                    }`}
                  >
                    {t}
                  </button>
                );
              })}
              {tab === "fundamentals" && (
                <div className="ml-auto flex gap-1">
                  {(["fy", "ttm"] as const).map((b) => (
                    <button
                      key={b}
                      type="button"
                      onClick={() => setBasis(b)}
                      disabled={Boolean(asOf && b === "ttm")}
                      aria-disabled={Boolean(asOf && b === "ttm")}
                      title={asOf && b === "ttm" ? "Time machine supports FY only" : undefined}
                      className={`font-data rounded-md px-2 py-0.5 text-[10px] uppercase ${
                        basis === b
                          ? "bg-glass-tint/15 text-moonlight"
                          : "text-sounding hover:text-moonlight disabled:cursor-not-allowed disabled:hover:text-sounding"
                      }`}
                    >
                      {b}
                    </button>
                  ))}
                </div>
              )}
            </div>

            {tab === "valuation" && (
              <div className="mt-4">
                {data.audit && data.audit.state !== "AUDITED" && (
                  <div
                    role="note"
                    className="mb-4 rounded-lg border border-amber/30 bg-amber/5 px-3 py-2 text-xs text-moonlight"
                  >
                    <p className="eyebrow text-[10px] text-amber/90">
                      {data.audit.state === "NO_VALUATION" ? "no valuation yet" : "not audited"}
                    </p>
                    <p className="mt-1 leading-relaxed">{data.audit.message}</p>
                    {data.audit.command && (
                      <p className="font-data mt-1.5 text-sounding">{data.audit.command}</p>
                    )}
                  </div>
                )}
                {data.valuations.length === 0 ? (
                  (data.research_valuations ?? []).length > 0 ? (
                    <>
                      <p className="eyebrow mb-2 text-[10px]">research output — not audited</p>
                      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                        {(data.research_valuations ?? []).map((card) => (
                          <MethodCardView key={card.method} card={card} />
                        ))}
                      </div>
                    </>
                  ) : (
                    data.audit?.state !== "NO_VALUATION" && (
                      <p className="text-sm text-sounding">No valuation rows for this ticker.</p>
                    )
                  )
                ) : (
                  <>
                    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                      {data.valuations.map((card) => (
                        <MethodCardView key={card.method} card={card} />
                      ))}
                    </div>
                    {data.evolution.length > 0 && (
                      <div className="mt-6">
                        <p className="eyebrow text-[10px]">
                          estimate evolution — how fair value moved
                        </p>
                        <div className="mt-2">
                          <EvolutionChart points={data.evolution} />
                        </div>
                        {((data.evolution_rebased_points ?? 0) > 0 ||
                          (data.evolution_dropped_pre_split ?? 0) > 0) && (
                          <p className="mt-1.5 text-[10px] text-sounding">
                            per-share values shown on today&apos;s share count
                            {(data.evolution_rebased_points ?? 0) > 0
                              ? ` · ${data.evolution_rebased_points} earlier point(s) adjusted for a stock split`
                              : ""}
                            {(data.evolution_dropped_pre_split ?? 0) > 0
                              ? ` · ${data.evolution_dropped_pre_split} earlier point(s) left off (split factor unclear)`
                              : ""}
                          </p>
                        )}
                      </div>
                    )}
                  </>
                )}
              </div>
            )}
            {tab === "fundamentals" && (
              <div className="mt-4">
                {fundamentals.isLoading ? (
                  <p className="text-sm text-sounding">loading…</p>
                ) : fundamentals.data?.basis === "fy" ? (
                  <FyGrid data={fundamentals.data as FundamentalsFy} />
                ) : fundamentals.data?.basis === "ttm" ? (
                  <TtmTable data={fundamentals.data as FundamentalsTtm} />
                ) : (
                  <p className="text-sm text-sounding">Could not load fundamentals.</p>
                )}
              </div>
            )}
            {tab === "research" && (
              <div className="mt-4">
                {research.isLoading ? (
                  <p className="text-sm text-sounding">loading…</p>
                ) : research.data ? (
                  <ResearchTab data={research.data} />
                ) : (
                  <p className="text-sm text-sounding">Could not load research.</p>
                )}
              </div>
            )}
            {tab === "dossier" && (
              <div className="mt-4">
                {dossier.isLoading ? (
                  <p className="text-sm text-sounding">loading…</p>
                ) : dossier.data ? (
                  <DossierTab data={dossier.data} />
                ) : (
                  <p className="text-sm text-sounding">Could not load the dossier.</p>
                )}
              </div>
            )}
            {tab === "decisions" && (
              <div className="mt-4">
                {decisions.isLoading ? (
                  <p className="text-sm text-sounding">loading…</p>
                ) : decisions.data ? (
                  <DecisionsTab data={decisions.data} falsifiers={row?.falsifiers ?? []} />
                ) : (
                  <p className="text-sm text-sounding">Could not load decisions.</p>
                )}
              </div>
            )}
          </Section>

          <div className="flex flex-wrap gap-2">
            <CopyChip
              command={`ivi watchlist check-triggers ${data.ticker}`}
              label={`ivi watchlist check-triggers ${data.ticker}`}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
