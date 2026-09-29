/**
 * Scans — the run gallery. Every sector run on disk as a glass card, with
 * the recent sweeps rolled up above (band × week: sectors covered, spend,
 * watchlist rows produced). Quarantined artifacts stay visible — a run the
 * indexer couldn't parse is a fact about the record, not a gap in it.
 */

import { useQuery } from "@tanstack/react-query";
import { Link, useNavigate, useSearch } from "@tanstack/react-router";

import { Section } from "../components/GlassPanel";
import type { RunSummary, SweepGroup } from "../lib/api";
import { fetchRuns, fetchSweeps } from "../lib/api";
import { humanizeToken } from "../lib/floor";
import {
  distinctRunValues,
  filterRuns,
  formatUsd,
  runHistoryLabel,
  verdictTone,
  type RunFilters,
} from "../lib/scans";

export interface ScansSearch extends RunFilters {}

const TONE_TEXT: Record<string, string> = {
  jade: "text-jade",
  amber: "text-amber",
  rose: "text-rose",
  slate: "text-slate-mist",
};

export function VerdictWord({ verdict }: { verdict: string | null }) {
  const tone = verdictTone(verdict);
  return (
    <span className={`text-xs lowercase ${TONE_TEXT[tone]}`} title={verdict ?? undefined}>
      {verdict ? humanizeToken(verdict) : "unknown"}
    </span>
  );
}

function runDate(run: RunSummary): string {
  const stamp = run.created_at ?? run.as_of_date;
  return stamp ? stamp.slice(0, 10) : "undated";
}

function SweepStrip({ groups }: { groups: SweepGroup[] }) {
  if (groups.length === 0) return null;
  return (
    <div className="board-scroll flex gap-3 overflow-x-auto pb-1">
      {groups.slice(0, 8).map((group) => (
        <div key={`${group.week}-${group.band}`} className="glass w-44 shrink-0 rounded-xl p-3">
          <div className="flex items-baseline justify-between">
            <span className="font-data text-[10px] text-sounding">{group.week}</span>
            <span className="text-[10px] text-sounding">{humanizeToken(group.band)}</span>
          </div>
          <p className="font-data mt-1.5 text-lg text-moonlight">
            {group.runs}
            <span className="ml-1 text-[10px] text-sounding">runs</span>
          </p>
          <p className="mt-0.5 text-[10px] text-sounding">
            {group.sectors.length} sectors · {formatUsd(group.cost_usd)}
          </p>
          <p className="text-[10px] text-sounding">
            {group.watchlist_rows > 0 ? (
              <span className="text-jade">+{group.watchlist_rows} watchlist rows</span>
            ) : (
              "no rows produced"
            )}
          </p>
        </div>
      ))}
    </div>
  );
}

function RunCard({ run }: { run: RunSummary }) {
  const slug = run.slug ?? run.run_id;
  const historyLabel = runHistoryLabel(run);
  if (historyLabel) {
    return (
      <Link
        to="/scans/run/$"
        params={{ _splat: slug }}
        className={`glass block rounded-xl border p-4 hover:-translate-y-0.5 ${
          run.parse_error ? "border-rose/30" : "border-amber/30"
        }`}
      >
        <p className="truncate text-sm text-moonlight">
          {run.sector ? humanizeToken(run.sector) : run.run_id}
        </p>
        <p className={`mt-1 text-xs ${run.parse_error ? "text-rose" : "text-amber"}`}>
          {historyLabel}
        </p>
        <p className="font-data mt-1 text-[10px] text-sounding">
          {runDate(run)} ·{" "}
          {run.market_cap_focus ? humanizeToken(run.market_cap_focus) : "band unavailable"}
        </p>
        {run.parse_error && (
          <p className="font-data mt-1 truncate text-[10px] text-sounding" title={run.parse_error}>
            {run.parse_error}
          </p>
        )}
      </Link>
    );
  }
  return (
    <Link
      to="/scans/run/$"
      params={{ _splat: slug }}
      className="glass block rounded-xl p-4 hover:-translate-y-0.5"
    >
      <div className="flex items-baseline justify-between gap-2">
        <p className="truncate text-sm text-moonlight">
          {run.sector ? humanizeToken(run.sector) : run.run_id}
        </p>
        <VerdictWord verdict={run.final_verdict} />
      </div>
      <p className="font-data mt-1 text-[10px] text-sounding">
        {runDate(run)} · {run.market_cap_focus ? humanizeToken(run.market_cap_focus) : "—"}
        {run.pipeline_version === "v2" ? " · v2" : ""}
        {run.scan_family && run.scan_family !== "normal" ? ` · ${run.scan_family}` : ""}
      </p>
      <div className="mt-2.5 flex items-baseline justify-between">
        <span className="font-data text-xs text-sounding">
          {run.examined_count !== null ? `${run.examined_count} examined` : "—"}
        </span>
        {run.selected_ticker ? (
          <span className="font-data text-xs font-medium text-moonlight">
            {run.selected_ticker}
          </span>
        ) : (
          <span className="font-data text-xs text-sounding">{formatUsd(run.cost_usd)}</span>
        )}
      </div>
    </Link>
  );
}

export function Scans() {
  const search = useSearch({ from: "/scans" }) as ScansSearch;
  const navigate = useNavigate();
  const includeHistory = search.history === "all";
  const runsQuery = useQuery({
    queryKey: ["runs", { includeHistory }],
    queryFn: () => fetchRuns(includeHistory),
    retry: 1,
  });
  const sweepsQuery = useQuery({ queryKey: ["sweeps"], queryFn: fetchSweeps, retry: 1 });

  const allRuns = runsQuery.data?.runs ?? [];
  const runs = filterRuns(allRuns, search);

  const patch = (update: Partial<ScansSearch>) =>
    void navigate({
      to: "/scans",
      search: (prev: ScansSearch) => {
        const merged = { ...prev, ...update };
        for (const key of Object.keys(merged) as (keyof ScansSearch)[]) {
          if (!merged[key]) delete merged[key];
        }
        return merged;
      },
      replace: true,
    });

  const facet = (
    label: string,
    key: keyof ScansSearch,
    values: string[],
    humanize = true,
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
            {humanize ? humanizeToken(v) : v}
          </option>
        ))}
      </select>
    </label>
  );

  return (
    <div className="max-w-[88rem]">
      <div className="flex flex-wrap items-baseline justify-between gap-3">
        <div>
          <p className="eyebrow">
            {includeHistory ? "sector scan audit history" : "decision-eligible sector research"}
          </p>
          <h1 className="font-display mt-1 text-4xl text-moonlight">Scans</h1>
        </div>
        <span className="font-data text-xs text-sounding">
          {runs.length} {includeHistory ? "current + excluded runs" : "audited current runs"}
        </span>
      </div>

      <Section condenseIndex={0} className="mt-7 p-4">
        <p className="eyebrow mb-3">recent sweeps</p>
        {sweepsQuery.isLoading ? (
          <p className="voice">loading…</p>
        ) : (
          <SweepStrip groups={sweepsQuery.data?.groups ?? []} />
        )}
      </Section>

      <Section condenseIndex={1} className="mt-4 p-4">
        <div className="flex flex-wrap items-center gap-3">
          <input
            value={search.q ?? ""}
            onChange={(e) => patch({ q: e.target.value || undefined })}
            placeholder="run, sector, ticker…"
            className="font-data w-40 rounded-md border border-glass-tint/20 bg-deepwater px-2 py-1 text-xs text-moonlight placeholder:text-sounding/50"
          />
          {facet("sector", "sector", distinctRunValues(allRuns, "sector"))}
          {facet("band", "band", distinctRunValues(allRuns, "market_cap_focus"))}
          {facet("verdict", "verdict", distinctRunValues(allRuns, "final_verdict"))}
          {facet("pipeline", "pipeline", distinctRunValues(allRuns, "pipeline_version"), false)}
          <button
            type="button"
            onClick={() => patch({ history: includeHistory ? undefined : "all" })}
            className={`ml-auto rounded-md border px-2 py-1 text-[11px] ${
              includeHistory
                ? "border-amber/35 text-amber"
                : "border-glass-tint/20 text-sounding hover:text-moonlight"
            }`}
          >
            {includeHistory ? "show current only" : "show excluded history"}
          </button>
        </div>
      </Section>

      {runsQuery.isLoading ? (
        <Section condenseIndex={2} className="mt-4 p-8">
          <p className="voice">loading…</p>
        </Section>
      ) : runs.length === 0 ? (
        <Section condenseIndex={2} className="mt-4 p-8">
          <p className="voice">No runs match these filters.</p>
        </Section>
      ) : (
        <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {runs.map((run) => (
            <RunCard key={run.path} run={run} />
          ))}
        </div>
      )}
    </div>
  );
}
