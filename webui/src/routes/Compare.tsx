/**
 * Compare — 2–4 tickers side by side on a COMMON scale.
 *
 * The ladder aligns every name on "% vs its own buy target" (0 = the
 * waterline), so submersion depth is comparable even when prices differ by
 * orders of magnitude; marks beyond the clamp are pinned to the edge with
 * their true value printed (no silent truncation). Fundamentals overlay on
 * a shared year axis — revenue indexed to 100 at each name's first visible
 * year (never a dual axis), FCF margin in comparable percent.
 *
 * Ticker identity wears the validated 4-slot categorical set (all six
 * checks pass on #0B1626); identity is never color-alone — every line ends
 * in its ticker label and every column is headed by it.
 */

import { useQueries } from "@tanstack/react-query";
import { Link, useNavigate, useSearch } from "@tanstack/react-router";
import { useState } from "react";

import { GradeWord, StatusChip } from "../components/chips";
import { Section } from "../components/GlassPanel";
import type { CompanyResponse, FundamentalsFy } from "../lib/api";
import { HttpError, fetchCompany, fetchFundamentals } from "../lib/api";
import {
  formatMillions,
  formatMoneyCompact,
  formatPct,
  humanizeToken,
  marginOfSafetyPct,
} from "../lib/floor";

/** Validated 4-slot categorical set for ticker identity (dark surface). */
const TICKER_HUES = ["#3987e5", "#d55181", "#c98500", "#8f77e8"];

export interface CompareSearch {
  tickers?: string;
}

const MAX_TICKERS = 4;
/** Ladder clamp: marks beyond ±this are pinned to the edge, value printed. */
const CLAMP_PCT = 150;

interface Column {
  ticker: string;
  company: CompanyResponse | undefined;
  error: Error | null;
  loading: boolean;
}

/** % vs buy target; falls back to the anchor value when no target exists. */
function referencePrice(company: CompanyResponse): {
  reference: number | null;
  basis: "target" | "anchor" | null;
} {
  const target = company.watchlist?.buy_price_target ?? null;
  if (target && target > 0) return { reference: target, basis: "target" };
  const anchor = company.profile?.valuation_anchor_value ?? null;
  if (anchor && anchor > 0) return { reference: anchor, basis: "anchor" };
  return { reference: null, basis: null };
}

function pctVs(value: number | null, reference: number | null): number | null {
  if (value === null || reference === null || reference <= 0) return null;
  return ((value - reference) / reference) * 100;
}

/* ------------------------------------------------------------------ */
/* The aligned ladder                                                  */
/* ------------------------------------------------------------------ */

interface LadderMark {
  kind: "price" | "shelf";
  label: string;
  pct: number;
  clamped: boolean;
  detail: string;
}

function ladderMarks(company: CompanyResponse): { marks: LadderMark[]; basis: string | null } {
  const { reference, basis } = referencePrice(company);
  if (reference === null) return { marks: [], basis: null };
  const marks: LadderMark[] = [];
  const price = company.prices.latest ?? company.watchlist?.latest_price ?? null;
  const pricePct = pctVs(price, reference);
  if (pricePct !== null && price !== null) {
    marks.push({
      kind: "price",
      label: formatMoneyCompact(price),
      pct: Math.max(Math.min(pricePct, CLAMP_PCT), -CLAMP_PCT),
      clamped: Math.abs(pricePct) > CLAMP_PCT,
      detail: `price ${formatMoneyCompact(price)} · ${formatPct(pricePct)} vs ${basis}`,
    });
  }
  for (const shelf of company.shelves) {
    const value = shelf.base ?? shelf.value ?? null;
    const pct = pctVs(value, reference);
    if (pct === null || value === null) continue;
    marks.push({
      kind: "shelf",
      label: shelf.label,
      pct: Math.max(Math.min(pct, CLAMP_PCT), -CLAMP_PCT),
      clamped: Math.abs(pct) > CLAMP_PCT,
      detail: `${shelf.label} ${formatMoneyCompact(value)} · ${formatPct(pct)} vs ${basis}`,
    });
  }
  return { marks, basis };
}

function AlignedLadder({ columns }: { columns: Column[] }) {
  const present = columns.filter((c) => c.company);
  const perColumn = present.map((c) => ladderMarks(c.company!));
  const allPcts = perColumn.flatMap(({ marks }) => marks.map((m) => m.pct));
  if (allPcts.length === 0) {
    return <p className="voice text-sm">no name here carries a buy target or anchor to align on.</p>;
  }
  const width = 640;
  const height = 340;
  const pad = { top: 18, right: 12, bottom: 26, left: 44 };
  const lo = Math.min(-25, ...allPcts) - 8;
  const hi = Math.max(25, ...allPcts) + 8;
  const y = (pct: number) =>
    pad.top + (hi - pct) * ((height - pad.top - pad.bottom) / (hi - lo));
  const colW = (width - pad.left - pad.right) / present.length;
  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      className="w-full"
      role="img"
      aria-label="Tickers aligned on percent versus their buy targets"
    >
      {/* The waterline: 0 = at target; the zone below is submerged. */}
      <rect
        x={pad.left}
        width={width - pad.left - pad.right}
        y={y(0)}
        height={height - pad.bottom - y(0)}
        fill="var(--color-jade)"
        fillOpacity={0.07}
      />
      <line
        x1={pad.left}
        x2={width - pad.right}
        y1={y(0)}
        y2={y(0)}
        stroke="var(--color-jade)"
        strokeOpacity={0.55}
        strokeDasharray="5 4"
      />
      {[...new Set([Math.round(lo / 50) * 50, -100, -50, 50, 100, 150, Math.round(hi / 50) * 50])]
        .filter((tick) => tick > lo && tick < hi && tick !== 0)
        .map((tick) => (
          <g key={tick}>
            <line
              x1={pad.left}
              x2={width - pad.right}
              y1={y(tick)}
              y2={y(tick)}
              stroke="var(--color-glass-tint)"
              strokeOpacity={0.06}
            />
            <text
              x={pad.left - 6}
              y={y(tick) + 3}
              textAnchor="end"
              fontSize={9}
              fill="var(--color-sounding)"
              className="font-data"
            >
              {tick > 0 ? `+${tick}%` : `${tick}%`}
            </text>
          </g>
        ))}
      <text
        x={pad.left - 6}
        y={y(0) + 3}
        textAnchor="end"
        fontSize={9}
        fill="var(--color-jade)"
        className="font-data"
      >
        target
      </text>
      {present.map((column, index) => {
        const { marks } = perColumn[index];
        const hue = TICKER_HUES[index % TICKER_HUES.length];
        const cx = pad.left + colW * index + colW / 2;
        return (
          <g key={column.ticker}>
            <text
              x={cx}
              y={height - pad.bottom + 14}
              textAnchor="middle"
              fontSize={11}
              fill={hue}
              className="font-data"
            >
              {column.ticker}
            </text>
            <line
              x1={cx}
              x2={cx}
              y1={pad.top}
              y2={height - pad.bottom}
              stroke="var(--color-glass-tint)"
              strokeOpacity={0.08}
            />
            {(() => {
              // Clamped marks pile up at the edge — fan their labels apart
              // so every value stays legible.
              const ordered = [...marks].sort((a, b) => y(a.pct) - y(b.pct));
              let lastLabelY = -Infinity;
              return ordered.map((mark, markIndex) => {
                const markY = y(mark.pct);
                const labelY = Math.max(markY, lastLabelY + 10);
                lastLabelY = labelY;
                return (
                  <g key={markIndex}>
                    {mark.kind === "price" ? (
                      <circle cx={cx} cy={markY} r={5} fill={hue}>
                        <title>{mark.detail}</title>
                      </circle>
                    ) : (
                      <line
                        x1={cx - 16}
                        x2={cx + 16}
                        y1={markY}
                        y2={markY}
                        stroke="var(--color-moonlight)"
                        strokeOpacity={0.55}
                        strokeWidth={2}
                      >
                        <title>{mark.detail}</title>
                      </line>
                    )}
                    <text
                      x={cx + (mark.kind === "price" ? 9 : 20)}
                      y={labelY + 3}
                      fontSize={8.5}
                      fill={mark.kind === "price" ? "var(--color-moonlight)" : "var(--color-sounding)"}
                      className="font-data"
                    >
                      {mark.label}
                      {mark.clamped ? " ⤒" : ""}
                    </text>
                  </g>
                );
              });
            })()}
          </g>
        );
      })}
      {perColumn.some(({ marks }) => marks.some((m) => m.clamped)) && (
        <text x={width - pad.right} y={12} textAnchor="end" fontSize={8} fill="var(--color-sounding)" className="font-data">
          ⤒ beyond ±{CLAMP_PCT}% — pinned to edge, true value in tooltip
        </text>
      )}
    </svg>
  );
}

/* ------------------------------------------------------------------ */
/* Fundamentals overlay                                                */
/* ------------------------------------------------------------------ */

interface OverlaySeries {
  ticker: string;
  hue: string;
  points: { year: number; value: number }[];
}

function OverlayChart({
  series,
  label,
  format,
}: {
  series: OverlaySeries[];
  label: string;
  format: (value: number) => string;
}) {
  const populated = series.filter((s) => s.points.length >= 2);
  if (populated.length === 0) {
    return null;
  }
  const width = 640;
  const height = 200;
  const pad = { top: 12, right: 74, bottom: 20, left: 44 };
  const years = populated.flatMap((s) => s.points.map((p) => p.year));
  const values = populated.flatMap((s) => s.points.map((p) => p.value));
  const yearLo = Math.min(...years);
  const yearHi = Math.max(...years);
  const valueLo = Math.min(...values);
  const valueHi = Math.max(...values);
  const x = (year: number) =>
    pad.left + (yearHi === yearLo ? 0 : (year - yearLo) * ((width - pad.left - pad.right) / (yearHi - yearLo)));
  const y = (value: number) =>
    pad.top +
    (valueHi === valueLo
      ? (height - pad.top - pad.bottom) / 2
      : (valueHi - value) * ((height - pad.top - pad.bottom) / (valueHi - valueLo)));
  return (
    <div>
      <p className="eyebrow text-[10px]">{label}</p>
      <svg viewBox={`0 0 ${width} ${height}`} className="mt-1 w-full" role="img" aria-label={label}>
        {valueLo < 0 && valueHi > 0 && (
          <line x1={pad.left} x2={width - pad.right} y1={y(0)} y2={y(0)} stroke="var(--color-sounding)" strokeOpacity={0.3} strokeDasharray="2 3" />
        )}
        {[valueLo, valueHi].map((tick) => (
          <text
            key={tick}
            x={pad.left - 6}
            y={y(tick) + 3}
            textAnchor="end"
            fontSize={9}
            fill="var(--color-sounding)"
            className="font-data"
          >
            {format(tick)}
          </text>
        ))}
        {[yearLo, yearHi].map((year) => (
          <text
            key={year}
            x={x(year)}
            y={height - 4}
            textAnchor="middle"
            fontSize={9}
            fill="var(--color-sounding)"
            className="font-data"
          >
            {year}
          </text>
        ))}
        {populated.map((s) => {
          const path = s.points
            .map((p, i) => `${i === 0 ? "M" : "L"}${x(p.year).toFixed(1)},${y(p.value).toFixed(1)}`)
            .join(" ");
          const last = s.points[s.points.length - 1];
          return (
            <g key={s.ticker}>
              <path d={path} fill="none" stroke={s.hue} strokeWidth={2} strokeLinejoin="round" />
              <text
                x={x(last.year) + 7}
                y={y(last.value) + 3}
                fontSize={10}
                fill="var(--color-moonlight)"
                className="font-data"
              >
                {s.ticker} {format(last.value)}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* The page                                                            */
/* ------------------------------------------------------------------ */

function parseTickers(raw: string | undefined): string[] {
  if (!raw) return [];
  return [...new Set(raw.split(",").map((t) => t.trim().toUpperCase()).filter(Boolean))].slice(
    0,
    MAX_TICKERS,
  );
}

const STAT_ROWS: {
  label: string;
  value: (c: CompanyResponse) => string;
}[] = [
  { label: "sector", value: (c) => humanizeToken(c.watchlist?.source_sector ?? null) || "—" },
  {
    label: "cap band",
    value: (c) =>
      c.watchlist?.cap_band_label && c.watchlist.cap_band_label !== "UNKNOWN_CAP"
        ? humanizeToken(c.watchlist.cap_band_label)
        : "—",
  },
  { label: "market cap", value: (c) => formatMillions(c.watchlist?.market_cap_mm ?? null) },
  {
    label: "price",
    value: (c) => formatMoneyCompact(c.prices.latest ?? c.watchlist?.latest_price ?? null),
  },
  { label: "buy target", value: (c) => formatMoneyCompact(c.watchlist?.buy_price_target ?? null) },
  {
    label: "from buy",
    value: (c) => formatPct(c.watchlist?.distance_from_buy_pct ?? null),
  },
  {
    label: "anchor",
    value: (c) =>
      c.profile?.valuation_anchor_method
        ? `${c.profile.valuation_anchor_method} ${formatMoneyCompact(c.profile.valuation_anchor_value)}`
        : "—",
  },
  {
    label: "MoS @ price",
    value: (c) =>
      formatPct(
        marginOfSafetyPct(
          c.prices.latest ?? c.watchlist?.latest_price ?? null,
          c.profile?.valuation_anchor_value ?? null,
        ),
      ),
  },
];

export function Compare() {
  const search = useSearch({ from: "/compare" }) as CompareSearch;
  const navigate = useNavigate({ from: "/compare" });
  const tickers = parseTickers(search.tickers);
  const [draft, setDraft] = useState("");

  const setTickers = (next: string[]) =>
    void navigate({
      search: { tickers: next.length > 0 ? next.join(",") : undefined },
      replace: true,
    });

  const companies = useQueries({
    queries: tickers.map((ticker) => ({
      queryKey: ["company", ticker],
      queryFn: () => fetchCompany(ticker),
      retry: 1,
    })),
  });
  const fundamentals = useQueries({
    queries: tickers.map((ticker) => ({
      queryKey: ["fundamentals", ticker, "fy"],
      queryFn: () => fetchFundamentals(ticker, "fy"),
      retry: 1,
    })),
  });

  const columns: Column[] = tickers.map((ticker, index) => ({
    ticker,
    company: companies[index]?.data,
    error: companies[index]?.error ?? null,
    loading: Boolean(companies[index]?.isLoading),
  }));
  const ready = columns.filter((c) => c.company);
  const notFound = columns.filter(
    (column) => column.error instanceof HttpError && column.error.status === 404,
  );
  const transientErrors = columns.filter(
    (column) =>
      column.error !== null &&
      !(column.error instanceof HttpError && column.error.status === 404),
  );

  const addTicker = () => {
    const next = parseTickers([...tickers, draft].join(","));
    setDraft("");
    setTickers(next);
  };

  const revenueSeries: OverlaySeries[] = [];
  const fcfMarginSeries: OverlaySeries[] = [];
  tickers.forEach((ticker, index) => {
    const data = fundamentals[index]?.data;
    if (!data || data.basis !== "fy") return;
    const fy = data as FundamentalsFy;
    const hue = TICKER_HUES[index % TICKER_HUES.length];
    const revenue = fy.series.revenue ?? [];
    const fcf = fy.derived.fcf ?? [];
    let base: number | null = null;
    const indexed: { year: number; value: number }[] = [];
    const margins: { year: number; value: number }[] = [];
    fy.fiscal_years.forEach((year, i) => {
      const rev = revenue[i];
      if (rev !== null && rev !== undefined && rev > 0) {
        if (base === null) base = rev;
        indexed.push({ year, value: (rev / base) * 100 });
        const f = fcf[i];
        if (f !== null && f !== undefined) {
          margins.push({ year, value: (f / rev) * 100 });
        }
      }
    });
    revenueSeries.push({ ticker, hue, points: indexed });
    fcfMarginSeries.push({ ticker, hue, points: margins });
  });

  return (
    <div className="max-w-6xl">
      <h1 className="font-display text-4xl text-moonlight">Compare</h1>
      <p className="voice mt-2">
        side by side on the waterline — % distance to each buy target
      </p>

      <div className="mt-5 flex flex-wrap items-center gap-2">
        {tickers.map((ticker, index) => (
          <span
            key={ticker}
            className="glass font-data flex items-center gap-2 rounded-lg px-2.5 py-1 text-xs"
            style={{ color: TICKER_HUES[index % TICKER_HUES.length] }}
          >
            {ticker}
            <button
              type="button"
              aria-label={`Remove ${ticker}`}
              onClick={() => setTickers(tickers.filter((t) => t !== ticker))}
              className="text-sounding hover:text-rose"
            >
              ✕
            </button>
          </span>
        ))}
        {tickers.length < MAX_TICKERS && (
          <form
            onSubmit={(event) => {
              event.preventDefault();
              addTicker();
            }}
          >
            <input
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              placeholder="add ticker…"
              aria-label="Add ticker to comparison"
              className="glass font-data w-28 rounded-lg border-none px-2.5 py-1 text-xs uppercase text-moonlight placeholder:normal-case placeholder:text-sounding/50"
            />
          </form>
        )}
      </div>

      {tickers.length < 2 ? (
        <Section className="mt-6 max-w-xl p-8">
          <p className="voice text-sm">
            add two to four tickers to lay their depths side by side.
          </p>
        </Section>
      ) : (
        <>
          {notFound.length > 0 && (
            <p className="font-data mt-3 text-xs text-rose">
              unknown to the platform:{" "}
              {notFound.map((column) => column.ticker).join(", ")}
            </p>
          )}
          {transientErrors.length > 0 && (
            <p className="font-data mt-3 text-xs text-rose">
              Couldn't load {transientErrors.map((column) => column.ticker).join(", ")} — the books
              may be briefly offline. Retry.
            </p>
          )}
          {ready.length >= 2 && (
            <>
              <Section condenseIndex={0} className="mt-6 p-6">
                <p className="eyebrow">common % scale — distance to each buy target</p>
                <div className="mt-3">
                  <AlignedLadder columns={ready} />
                </div>
              </Section>

              <Section condenseIndex={1} className="mt-5 overflow-x-auto p-6">
                <table className="w-full border-collapse text-left">
                  <thead>
                    <tr className="text-[10px] uppercase tracking-wider text-sounding">
                      <th className="py-1.5 pr-4 font-medium">·</th>
                      {ready.map((column, index) => (
                        <th key={column.ticker} className="py-1.5 pr-4 font-medium">
                          <Link
                            to="/company/$ticker"
                            params={{ ticker: column.ticker }}
                            search={{ tab: undefined, basis: undefined }}
                            className="font-data text-xs underline-offset-4 hover:underline"
                            style={{ color: TICKER_HUES[index % TICKER_HUES.length] }}
                          >
                            {column.ticker}
                          </Link>
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    <tr className="border-t border-glass-tint/8">
                      <td className="py-1.5 pr-4 text-[10px] uppercase tracking-wider text-sounding">
                        chips
                      </td>
                      {ready.map((column) => (
                        <td key={column.ticker} className="py-1.5 pr-4">
                          <span className="flex flex-wrap items-baseline gap-1.5">
                            <StatusChip value={column.company!.watchlist?.presented_status ?? null} />
                            <GradeWord value={column.company!.watchlist?.conviction_grade ?? null} />
                          </span>
                        </td>
                      ))}
                    </tr>
                    {STAT_ROWS.map((row) => (
                      <tr key={row.label} className="border-t border-glass-tint/8 text-xs">
                        <td className="py-1.5 pr-4 text-[10px] uppercase tracking-wider text-sounding">
                          {row.label}
                        </td>
                        {ready.map((column) => (
                          <td key={column.ticker} className="font-data py-1.5 pr-4 text-moonlight">
                            {row.value(column.company!)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </Section>

              {(revenueSeries.some((s) => s.points.length >= 2) ||
                fcfMarginSeries.some((s) => s.points.length >= 2)) && (
                <Section condenseIndex={2} className="mt-5 space-y-6 p-6">
                  <OverlayChart
                    series={revenueSeries}
                    label="revenue — indexed to 100 at each name's first year"
                    format={(value) => value.toFixed(0)}
                  />
                  <OverlayChart
                    series={fcfMarginSeries}
                    label="FCF margin — %"
                    format={(value) => `${value.toFixed(0)}%`}
                  />
                </Section>
              )}
            </>
          )}
          {ready.length < 2 && columns.some((c) => c.loading) && (
            <p className="mt-6 text-sm text-sounding">loading…</p>
          )}
        </>
      )}
    </div>
  );
}
