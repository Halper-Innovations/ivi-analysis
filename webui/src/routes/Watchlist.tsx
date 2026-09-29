/**
 * Watchlist — the watchlist floor. Three projections of the same queue:
 * the ranked queue table, the status board (a kanban you can't drag —
 * statuses move only via the platform), and the distance-to-zone ladder.
 *
 * Frost is semantic: row tint tracks price-snapshot age (flat tint, never
 * per-row backdrop blur — the long-table GPU rule), and each row carries a
 * depth pip. Raw platform codes render humanized; the raw value stays in
 * the tooltip or URL.
 */

import { useQuery } from "@tanstack/react-query";
import { Link, useNavigate, useSearch } from "@tanstack/react-router";

import { CopyChip, GradeWord, StatusChip } from "../components/chips";
import { DepthPip } from "../components/DepthPip";
import { Section } from "../components/GlassPanel";
import type { WatchlistRow } from "../lib/api";
import { OfflineError, fetchWatchlist } from "../lib/api";
import {
  clearToAct,
  distinctValues,
  filterRows,
  formatAge,
  formatMillions,
  formatMoneyCompact,
  formatPct,
  formatRawCompact,
  groupForBoard,
  hoursSince,
  humanizeToken,
  leaderboardRows,
  marginOfSafetyPct,
  rowGaugeInput,
  type FloorFilters,
} from "../lib/floor";
import { frostLevel } from "../lib/gauge";

export interface FloorSearch extends FloorFilters {
  view?: "queue" | "board" | "ladder";
}

function rowFrostClass(row: WatchlistRow, now: Date): string {
  const level = frostLevel(
    hoursSince(row.latest_price_checked_at, now),
    row.status === "PRICE_DATA_SUSPECT",
  );
  return `row-frost-${level}`;
}

function TickerCell({ row }: { row: WatchlistRow }) {
  return (
    <Link
      to="/company/$ticker"
      params={{ ticker: row.ticker }}
      className="font-data text-sm font-medium text-moonlight hover:text-jade"
    >
      {row.ticker}
    </Link>
  );
}

/** Cap band cell: UNKNOWN_CAP is an absence, not a category. */
function bandLabel(band: string | null): string {
  if (!band || band === "UNKNOWN_CAP") return "—";
  return humanizeToken(band);
}

function FilterBar({
  rows,
  search,
  navigate,
}: {
  rows: WatchlistRow[];
  search: FloorSearch;
  navigate: ReturnType<typeof useNavigate>;
}) {
  const patch = (update: Partial<FloorSearch>) =>
    void navigate({
      to: "/watchlist",
      search: (prev: FloorSearch) => {
        const merged = { ...prev, ...update };
        // Empty facets leave the URL entirely.
        for (const key of Object.keys(merged) as (keyof FloorSearch)[]) {
          if (!merged[key]) delete merged[key];
        }
        return merged;
      },
      replace: true,
    });

  const facet = (
    label: string,
    key: "band" | "grade" | "family" | "status",
    values: string[],
  ) => (
    <label className="flex items-center gap-1.5 text-[11px] text-sounding">
      {label}
      <select
        value={search[key] ?? ""}
        onChange={(e) => patch({ [key]: e.target.value || undefined })}
        className="font-data rounded-md border border-glass-tint/20 bg-deepwater px-1.5 py-1 text-[11px] text-moonlight"
      >
        <option value="">all</option>
        {values.map((v) => (
          <option key={v} value={v}>
            {humanizeToken(v)}
          </option>
        ))}
      </select>
    </label>
  );

  return (
    <div className="flex flex-wrap items-center gap-3">
      <input
        value={search.q ?? ""}
        onChange={(e) => patch({ q: e.target.value || undefined })}
        placeholder="ticker…"
        className="font-data w-28 rounded-md border border-glass-tint/20 bg-deepwater px-2 py-1 text-xs text-moonlight placeholder:text-sounding/50"
      />
      {facet("band", "band", distinctValues(rows, "cap_band_label"))}
      {facet("grade", "grade", distinctValues(rows, "conviction_grade"))}
      {facet("family", "family", distinctValues(rows, "scan_family"))}
      {facet("status", "status", distinctValues(rows, "presented_status"))}
    </div>
  );
}

function QueueTable({ rows, now }: { rows: WatchlistRow[]; now: Date }) {
  return (
    <div className="overflow-x-auto">
      <table className="floor-table w-full border-collapse text-left">
        <thead>
          <tr className="text-[10px] uppercase tracking-wider text-sounding">
            <th className="py-2 pr-3 font-medium">ticker</th>
            <th className="py-2 pr-3 font-medium">depth</th>
            <th className="py-2 pr-3 font-medium">status</th>
            <th className="py-2 pr-3 font-medium">conviction</th>
            <th className="py-2 pr-3 text-right font-medium">price</th>
            <th className="py-2 pr-3 text-right font-medium">target</th>
            <th className="py-2 pr-3 text-right font-medium">distance</th>
            <th className="py-2 pr-3 text-right font-medium">age</th>
            <th className="py-2 pr-3 text-right font-medium">cap</th>
            <th className="py-2 pr-3 text-right font-medium">adv 20d</th>
            <th className="py-2 pr-3 font-medium">band</th>
            <th className="py-2 pr-0 font-medium">sector</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => {
            const age = hoursSince(row.latest_price_checked_at, now);
            const submerged =
              row.distance_from_buy_pct !== null && row.distance_from_buy_pct <= 0;
            const clear = clearToAct(row.presented_status);
            return (
              <tr
                key={row.id}
                className={`border-t border-glass-tint/8 text-xs transition-opacity ${rowFrostClass(row, now)}`}
              >
                <td className="py-2 pr-3"><TickerCell row={row} /></td>
                <td className="py-2 pr-3">
                  <DepthPip input={rowGaugeInput(row, now)} />
                </td>
                <td className="py-2 pr-3"><StatusChip value={row.presented_status} title={row.status_reason ?? undefined} /></td>
                <td className="py-2 pr-3"><GradeWord value={row.conviction_grade} /></td>
                <td className="font-data py-2 pr-3 text-right text-moonlight">
                  {formatMoneyCompact(row.latest_price)}
                </td>
                <td className="font-data py-2 pr-3 text-right text-sounding">
                  {formatMoneyCompact(row.buy_price_target)}
                </td>
                <td className={`font-data py-2 pr-3 text-right ${submerged && clear ? "text-jade" : "text-moonlight"}`}>
                  {formatPct(row.distance_from_buy_pct)}
                </td>
                <td className="font-data py-2 pr-3 text-right text-sounding" title={row.latest_price_checked_at ?? undefined}>
                  {formatAge(age)}
                </td>
                <td className="font-data py-2 pr-3 text-right text-sounding">
                  {formatMillions(row.market_cap_mm)}
                </td>
                <td className="font-data py-2 pr-3 text-right text-sounding" title={row.capacity_class ?? undefined}>
                  {formatRawCompact(row.adv_dollar_20d)}
                </td>
                <td className="py-2 pr-3 text-sounding" title={row.cap_band_label ?? undefined}>
                  {bandLabel(row.cap_band_label)}
                </td>
                <td className="py-2 pr-0 text-sounding" title={row.source_sector ?? undefined}>
                  {row.source_sector ? humanizeToken(row.source_sector) : "—"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function Board({ rows, now }: { rows: WatchlistRow[]; now: Date }) {
  const groups = groupForBoard(rows);
  return (
    <div className="board-scroll flex gap-4 overflow-x-auto pb-2">
      {[...groups.entries()].map(([status, group]) => (
        <div key={status} className="w-60 shrink-0">
          <div className="mb-2 flex items-baseline justify-between px-1">
            <StatusChip value={status} />
            <span className="font-data text-[10px] text-sounding">{group.length}</span>
          </div>
          <div className="flex max-h-[60vh] flex-col gap-2 overflow-y-auto pr-1">
            {group.map((row) => {
              const level = frostLevel(
                hoursSince(row.latest_price_checked_at, now),
                row.status === "PRICE_DATA_SUSPECT",
              );
              return (
                <Link
                  key={row.id}
                  to="/company/$ticker"
                  params={{ ticker: row.ticker }}
                  className={`glass block rounded-xl p-3 hover:-translate-y-0.5 ${level > 0 ? `frosted frost-${level}` : ""}`}
                  title={row.status_reason ?? undefined}
                >
                  <div className="frost-content">
                    <div className="flex items-baseline justify-between">
                      <span className="font-data text-sm font-medium text-moonlight">{row.ticker}</span>
                      <span className="font-data text-[10px] text-sounding">
                        {formatPct(row.distance_from_buy_pct)}
                      </span>
                    </div>
                    <div className="mt-1 flex items-center justify-between gap-2">
                      <GradeWord value={row.conviction_grade} />
                      <span className="font-data text-[10px] text-sounding">
                        {formatMoneyCompact(row.latest_price)}
                      </span>
                    </div>
                    {status === "QUARANTINE" && row.status_reason && (
                      <p className="mt-1.5 truncate text-[10px] text-slate-mist">
                        {humanizeToken(row.status_reason)}
                      </p>
                    )}
                  </div>
                </Link>
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
}

function Ladder({ rows, now }: { rows: WatchlistRow[]; now: Date }) {
  const ladder = leaderboardRows(rows);
  return (
    <div className="overflow-x-auto">
      <table className="floor-table w-full border-collapse text-left">
        <thead>
          <tr className="text-[10px] uppercase tracking-wider text-sounding">
            <th className="py-2 pr-3 font-medium">#</th>
            <th className="py-2 pr-3 font-medium">ticker</th>
            <th className="py-2 pr-3 text-right font-medium">distance</th>
            <th className="py-2 pr-3 text-right font-medium">price</th>
            <th className="py-2 pr-3 text-right font-medium">target</th>
            <th className="py-2 pr-3 text-right font-medium">MoS @ price</th>
            <th className="py-2 pr-3 text-right font-medium">age</th>
            <th className="py-2 pr-0 font-medium">status</th>
          </tr>
        </thead>
        <tbody>
          {ladder.map((row, i) => {
            const submerged = row.distance_from_buy_pct! <= 0;
            // Jade marks in-zone AND clear-to-act. A name held in the zone
            // stays neutral — its status chip carries the hold.
            const clear = clearToAct(row.presented_status);
            return (
              <tr
                key={row.id}
                className={`border-t border-glass-tint/8 text-xs ${rowFrostClass(row, now)}`}
              >
                <td className="font-data py-2 pr-3 text-sounding">{i + 1}</td>
                <td className="py-2 pr-3"><TickerCell row={row} /></td>
                <td
                  className={`font-data py-2 pr-3 text-right ${
                    submerged && clear ? "text-jade font-semibold" : "text-moonlight"
                  }`}
                >
                  {formatPct(row.distance_from_buy_pct)}
                </td>
                <td className="font-data py-2 pr-3 text-right text-moonlight">
                  {formatMoneyCompact(row.latest_price)}
                </td>
                <td className="font-data py-2 pr-3 text-right text-sounding">
                  {formatMoneyCompact(row.buy_price_target)}
                </td>
                <td className="font-data py-2 pr-3 text-right text-sounding">
                  {formatPct(marginOfSafetyPct(row.latest_price, row.valuation_anchor_value))}
                </td>
                <td className="font-data py-2 pr-3 text-right text-sounding">
                  {formatAge(hoursSince(row.latest_price_checked_at, now))}
                </td>
                <td className="py-2 pr-0"><StatusChip value={row.presented_status} /></td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export function Watchlist() {
  const search = useSearch({ from: "/watchlist" }) as FloorSearch;
  const navigate = useNavigate();
  const query = useQuery({ queryKey: ["watchlist"], queryFn: fetchWatchlist, retry: 1 });
  const now = new Date();

  if (query.error instanceof OfflineError) {
    return (
      <Section className="max-w-xl p-8">
        <h1 className="font-display text-2xl text-moonlight">IVI offline</h1>
        <p className="font-data mt-3 text-sm text-rose">{query.error.precondition}</p>
        <p className="font-data mt-1 break-all text-xs text-sounding">{query.error.detail}</p>
      </Section>
    );
  }

  const allRows = query.data?.rows ?? [];
  const rows = filterRows(allRows, search);
  const view = search.view ?? "queue";
  const viewButton = (v: FloorSearch["view"], label: string) => (
    <button
      type="button"
      onClick={() =>
        void navigate({
          to: "/watchlist",
          search: (prev: FloorSearch) => ({ ...prev, view: v === "queue" ? undefined : v }),
          replace: true,
        })
      }
      className={`rounded-lg px-3 py-1 text-xs transition-colors ${
        view === v ? "bg-glass-tint/15 text-moonlight" : "text-sounding hover:text-moonlight"
      }`}
    >
      {label}
    </button>
  );

  const quarantined = allRows.filter((r) => r.presented_status === "QUARANTINE").length;

  return (
    <div className="max-w-[88rem]">
      <div className="flex flex-wrap items-baseline justify-between gap-3">
        <div>
          <p className="eyebrow">every tracked name</p>
          <h1 className="font-display mt-1 text-4xl text-moonlight">Watchlist</h1>
        </div>
        <span className="font-data text-xs text-sounding">
          {rows.length} of {allRows.length} rows · {quarantined} quarantined
        </span>
      </div>

      <Section condenseIndex={0} className="mt-7 p-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex gap-1 rounded-xl border border-glass-tint/10 p-1">
            {viewButton("queue", "Queue")}
            {viewButton("board", "Board")}
            {viewButton("ladder", "Ladder")}
          </div>
          <FilterBar rows={allRows} search={search} navigate={navigate} />
        </div>
      </Section>

      <Section condenseIndex={1} className="mt-4 p-4">
        {query.isLoading ? (
          <p className="voice p-4">loading…</p>
        ) : rows.length === 0 ? (
          <p className="voice p-4">No rows match these filters.</p>
        ) : view === "board" ? (
          <Board rows={rows} now={now} />
        ) : view === "ladder" ? (
          <Ladder rows={rows} now={now} />
        ) : (
          <QueueTable rows={rows} now={now} />
        )}
      </Section>

      <div className="mt-4 flex items-center gap-3">
        <CopyChip command="ivi watchlist check-triggers" label="ivi watchlist check-triggers" />
        <span className="text-[10px] text-sounding/85">
          statuses move only via the platform — the board cannot be dragged
        </span>
      </div>
    </div>
  );
}
