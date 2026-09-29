/**
 * Ops — operational status. The morning check on one screen with zero
 * terminal use: the deadman wall (compute_data_health verbatim), the
 * heartbeat ledger with inline cron-log tails, backup freshness, the
 * cost ledger, and the legacy consoles carried into the shell.
 */

import { useQuery } from "@tanstack/react-query";
import { useState } from "react";

import { StatusChip } from "../components/chips";
import { Section } from "../components/GlassPanel";
import type { HeartbeatDay } from "../lib/api";
import {
  fetchBackups,
  fetchConsoles,
  fetchCosts,
  fetchHeartbeats,
  fetchLogTail,
  fetchOpsHealth,
} from "../lib/api";
import { humanizeToken } from "../lib/floor";
import { costChartRows, formatBytes, heartbeatGlyph, shortWeek } from "../lib/ops";
import { formatUsd } from "../lib/scans";

const TONE_TEXT = {
  jade: "text-jade",
  amber: "text-amber",
  rose: "text-rose",
  slate: "text-sounding/40",
} as const;

/** The validated categorical pair on the abyss surface (dataviz-checked):
 *  gold = scan spend, blue = research spend. */
const SCAN_HUE = "#c98500";
const RESEARCH_HUE = "#3987e5";

function LogViewer({ name, onClose }: { name: string; onClose: () => void }) {
  const query = useQuery({
    queryKey: ["ops-log", name],
    queryFn: () => fetchLogTail(name),
    staleTime: 30_000,
    retry: 1,
  });
  return (
    <Section className="mt-3 p-4">
      <div className="flex items-baseline justify-between gap-3">
        <p className="font-data text-xs text-moonlight">{name}</p>
        <div className="flex items-baseline gap-3">
          {query.data && (
            <span className="font-data text-[10px] text-sounding">
              {formatBytes(query.data.size)}
              {query.data.truncated ? " · tail" : ""}
            </span>
          )}
          <button
            type="button"
            onClick={onClose}
            className="font-data text-xs text-sounding hover:text-moonlight"
            aria-label="Close log"
          >
            ✕
          </button>
        </div>
      </div>
      {query.isLoading ? (
        <p className="voice mt-3">loading…</p>
      ) : query.error ? (
        <p className="mt-3 text-xs text-rose">The log could not be read.</p>
      ) : (
        <pre className="font-data mt-3 max-h-72 overflow-auto whitespace-pre-wrap break-all rounded-lg bg-abyss/60 p-3 text-[11px] leading-relaxed text-sounding">
          {query.data?.lines.join("\n") || "(empty)"}
        </pre>
      )}
    </Section>
  );
}

function HeartbeatCellDetail({
  heartbeat,
  day,
  onClose,
}: {
  heartbeat: string;
  day: HeartbeatDay;
  onClose: () => void;
}) {
  return (
    <Section className="mt-3 p-4">
      <div className="flex items-baseline justify-between gap-3">
        <p className="text-sm text-moonlight">
          <span className="font-data">{heartbeat}</span>
          <span className="text-sounding"> · {day.run_date} · </span>
          <span className={TONE_TEXT[heartbeatGlyph(day.status).tone]}>
            {heartbeatGlyph(day.status).label}
          </span>
        </p>
        <button
          type="button"
          onClick={onClose}
          className="font-data text-xs text-sounding hover:text-moonlight"
          aria-label="Close heartbeat detail"
        >
          ✕
        </button>
      </div>
      {day.steps.length > 0 ? (
        <table className="mt-3 w-full text-left text-[11px]">
          <thead>
            <tr className="text-[10px] uppercase tracking-wider text-sounding">
              <th className="pb-1 pr-4 font-medium">step</th>
              <th className="pb-1 pr-4 font-medium">status</th>
              <th className="pb-1 pr-4 font-medium">exit</th>
              <th className="pb-1 font-medium">recorded</th>
            </tr>
          </thead>
          <tbody>
            {day.steps.map((step, index) => (
              <tr key={`${step.step}-${index}`} className="border-t border-glass-tint/8">
                <td className="font-data py-1 pr-4 text-moonlight">{step.step}</td>
                <td className={`py-1 pr-4 ${step.status === "OK" ? "text-jade" : "text-rose"}`}>
                  {step.status === "OK" ? "● ok" : "✕ failed"}
                </td>
                <td className="font-data py-1 pr-4 text-sounding">{step.exit_code ?? "—"}</td>
                <td className="font-data py-1 text-sounding">
                  {step.recorded_at?.slice(11, 19) ?? "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <p className="mt-3 text-xs text-sounding">No steps recorded for this day.</p>
      )}
    </Section>
  );
}

export function Ops() {
  const health = useQuery({ queryKey: ["ops-health"], queryFn: fetchOpsHealth, retry: 1 });
  const heartbeats = useQuery({
    queryKey: ["ops-heartbeats"],
    queryFn: () => fetchHeartbeats(14),
    retry: 1,
  });
  const backups = useQuery({ queryKey: ["ops-backups"], queryFn: fetchBackups, retry: 1 });
  const costs = useQuery({ queryKey: ["ops-costs"], queryFn: fetchCosts, retry: 1 });
  const consoles = useQuery({ queryKey: ["ops-consoles"], queryFn: fetchConsoles, retry: 1 });

  const [openLog, setOpenLog] = useState<string | null>(null);
  const [openCell, setOpenCell] = useState<{ heartbeat: string; day: HeartbeatDay } | null>(
    null,
  );

  const wall = health.data?.health;
  const chartRows = costs.data ? costChartRows(costs.data.weeks) : [];

  return (
    <div className="max-w-[88rem]">
      <div className="flex flex-wrap items-baseline justify-between gap-3">
        <div>
          <p className="eyebrow">operational status</p>
          <h1 className="font-display mt-1 text-4xl text-moonlight">Ops</h1>
        </div>
        {wall && (
          <StatusChip
            value={wall.state}
            title={wall.blocking ? "database integrity failed — blocking" : "data health"}
          />
        )}
      </div>

      {/* Deadman wall */}
      <Section condenseIndex={0} className="mt-7 p-5">
        <h2 className="font-display text-xl text-moonlight">Deadman wall</h2>
        <p className="voice mt-0.5">
          the fail-loud assertions, exactly as `ivi ops deadman` computes them
        </p>
        {health.isLoading ? (
          <p className="voice mt-4">running the checks…</p>
        ) : !wall ? (
          <p className="mt-4 text-xs text-rose">The health block could not be read.</p>
        ) : (
          <>
            {wall.blocking && (
              <p className="mt-3 rounded-lg border border-rose/30 bg-rose/10 px-3 py-2 text-xs text-rose">
                BLOCKING — database failed basic integrity; at-target surfaces are
                suppressed until this is fixed.
              </p>
            )}
            <div className="mt-4 grid grid-cols-1 gap-2.5 md:grid-cols-2 xl:grid-cols-3">
              {wall.checks.map((check) => (
                <div
                  key={check.name}
                  className={`rounded-xl border p-3 ${
                    check.ok
                      ? "border-glass-tint/10 bg-deepwater/40"
                      : "border-rose/30 bg-rose/10"
                  }`}
                >
                  <p className="flex items-baseline justify-between gap-2">
                    <span className="text-xs text-moonlight">{humanizeToken(check.name)}</span>
                    <span className={check.ok ? "text-jade" : "text-rose"}>
                      {check.ok ? "● ok" : "✕ failing"}
                    </span>
                  </p>
                  <p className="font-data mt-1 break-all text-[11px] text-sounding">
                    {check.detail}
                  </p>
                </div>
              ))}
            </div>
          </>
        )}
      </Section>

      {/* Heartbeat ledger */}
      <Section condenseIndex={1} className="mt-5 p-5">
        <h2 className="font-display text-xl text-moonlight">Heartbeat ledger</h2>
        <p className="voice mt-0.5">
          each cron heartbeat's step record, last 14 days — click a cell for steps
          and the day's log
        </p>
        {heartbeats.isLoading ? (
          <p className="voice mt-4">loading…</p>
        ) : !heartbeats.data || heartbeats.data.heartbeats.length === 0 ? (
          <p className="mt-4 text-xs text-sounding">No heartbeat rows in the window.</p>
        ) : (
          <div className="mt-4 overflow-x-auto">
            <table className="border-separate" style={{ borderSpacing: "2px" }}>
              <thead>
                <tr>
                  <th className="pr-3 text-left text-[10px] font-medium uppercase tracking-wider text-sounding">
                    heartbeat
                  </th>
                  {heartbeats.data.dates
                    .slice()
                    .reverse()
                    .map((date) => (
                      <th
                        key={date}
                        className="font-data px-0.5 pb-1 text-center text-[9px] font-normal text-sounding/85"
                        title={date}
                      >
                        {date.slice(8)}
                      </th>
                    ))}
                </tr>
              </thead>
              <tbody>
                {heartbeats.data.heartbeats.map((row) => (
                  <tr key={row.heartbeat}>
                    <td className="font-data whitespace-nowrap pr-3 text-xs text-moonlight">
                      {row.heartbeat}
                    </td>
                    {row.days
                      .slice()
                      .reverse()
                      .map((day) => {
                        const glyph = heartbeatGlyph(day.status);
                        const selected =
                          openCell?.heartbeat === row.heartbeat &&
                          openCell.day.run_date === day.run_date;
                        return (
                          <td key={day.run_date} className="p-0">
                            <button
                              type="button"
                              onClick={() => {
                                setOpenCell(
                                  selected ? null : { heartbeat: row.heartbeat, day },
                                );
                                setOpenLog(selected ? null : day.log_name);
                              }}
                              className={`h-7 w-7 rounded-md border text-sm ${TONE_TEXT[glyph.tone]} ${
                                selected
                                  ? "border-glass-tint/70"
                                  : "border-glass-tint/10 hover:border-glass-tint/40"
                              }`}
                              title={`${row.heartbeat} · ${day.run_date} — ${glyph.label}${
                                day.failed_steps.length
                                  ? ` (failed: ${day.failed_steps.join(", ")})`
                                  : ""
                              }`}
                            >
                              {glyph.glyph}
                            </button>
                          </td>
                        );
                      })}
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="mt-2 text-[10px] text-sounding/85">
              ● complete · ◐ started, never completed · ✕ failed · faint dot = no run
              recorded
            </p>
          </div>
        )}
        {openCell && (
          <HeartbeatCellDetail
            heartbeat={openCell.heartbeat}
            day={openCell.day}
            onClose={() => {
              setOpenCell(null);
              setOpenLog(null);
            }}
          />
        )}
        {openLog && <LogViewer name={openLog} onClose={() => setOpenLog(null)} />}
      </Section>

      {/* Backups */}
      <Section condenseIndex={2} className="mt-5 p-5">
        <h2 className="font-display text-xl text-moonlight">Backups</h2>
        {backups.isLoading ? (
          <p className="voice mt-4">loading…</p>
        ) : !backups.data ? (
          <p className="mt-4 text-xs text-rose">The backup status could not be read.</p>
        ) : (
          <div className="mt-3">
            <p className="text-sm">
              <span className={backups.data.ok ? "text-jade" : "text-rose"}>
                {backups.data.ok ? "● " : "✕ "}
              </span>
              <span className="font-data text-moonlight">{backups.data.detail}</span>
              <span className="text-sounding">
                {" "}
                · ceiling {backups.data.ceiling_days}d
              </span>
            </p>
            {backups.data.status_detail && !backups.data.ok && (
              <p className="font-data mt-1.5 break-all text-xs text-rose/90">
                {backups.data.status_detail}
              </p>
            )}
            {backups.data.finished_at && (
              <p className="font-data mt-1 text-[11px] text-sounding">
                last attempt finished {backups.data.finished_at}
                {backups.data.duration_s !== null
                  ? ` in ${backups.data.duration_s}s`
                  : ""}
              </p>
            )}
            {backups.data.history.length > 0 && (
              <div className="mt-3 flex flex-wrap gap-1.5">
                {backups.data.history.slice(0, 10).map((log) => (
                  <button
                    key={log.log_name}
                    type="button"
                    onClick={() =>
                      setOpenLog(openLog === log.log_name ? null : log.log_name)
                    }
                    className="font-data rounded-lg border border-glass-tint/15 px-2 py-1 text-[10px] text-sounding hover:text-moonlight"
                    title={`${formatBytes(log.size)}`}
                  >
                    {log.date
                      ? `${log.date.slice(0, 4)}-${log.date.slice(4, 6)}-${log.date.slice(6)}`
                      : log.log_name}
                  </button>
                ))}
              </div>
            )}
          </div>
        )}
      </Section>

      {/* Cost ledger */}
      <Section condenseIndex={3} className="mt-5 p-5">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <div>
            <h2 className="font-display text-xl text-moonlight">Cost ledger</h2>
            <p className="voice mt-0.5">what the scanning actually costs, by week</p>
          </div>
          {costs.data && (
            <p className="font-data text-xs text-sounding">
              scans {formatUsd(costs.data.total_scan_usd)} · research{" "}
              {formatUsd(costs.data.total_research_usd)}
            </p>
          )}
        </div>
        {costs.isLoading ? (
          <p className="voice mt-4">loading…</p>
        ) : !costs.data ? (
          <p className="mt-4 text-xs text-rose">The cost ledger could not be read.</p>
        ) : (
          <>
            {chartRows.length > 0 && (
              <div className="mt-4">
                <div className="flex items-end gap-[6px]" role="img" aria-label="Weekly spend, stacked scan and research">
                  {chartRows.map((row) => (
                    <div
                      key={row.week}
                      className="flex w-8 flex-col items-center gap-1"
                      title={`${row.week} — scans ${formatUsd(row.scan)}, research ${formatUsd(row.research)}, total ${formatUsd(row.total)}`}
                    >
                      <div className="flex h-24 w-4 flex-col justify-end gap-[2px]">
                        <div
                          className="w-full rounded-t-[3px]"
                          style={{
                            height: `${Math.round(row.researchFrac * 96)}px`,
                            backgroundColor: RESEARCH_HUE,
                          }}
                        />
                        <div
                          className="w-full rounded-t-[3px]"
                          style={{
                            height: `${Math.round(row.scanFrac * 96)}px`,
                            backgroundColor: SCAN_HUE,
                          }}
                        />
                      </div>
                      <span className="font-data text-[9px] text-sounding/85">
                        {shortWeek(row.week)}
                      </span>
                    </div>
                  ))}
                </div>
                <p className="mt-2 flex items-center gap-4 text-[10px] text-sounding">
                  <span className="flex items-center gap-1.5">
                    <span
                      className="inline-block h-2.5 w-2.5 rounded-[2px]"
                      style={{ backgroundColor: SCAN_HUE }}
                    />
                    sector-scan spend
                  </span>
                  <span className="flex items-center gap-1.5">
                    <span
                      className="inline-block h-2.5 w-2.5 rounded-[2px]"
                      style={{ backgroundColor: RESEARCH_HUE }}
                    />
                    research synthesis spend
                  </span>
                  {costs.data.runs_without_cost > 0 && (
                    <span className="text-sounding/80">
                      {costs.data.runs_without_cost} runs carry no cost fields and are
                      not in these bars
                    </span>
                  )}
                </p>
              </div>
            )}
            <div className="mt-5 grid grid-cols-1 gap-5 lg:grid-cols-3">
              {(
                [
                  ["by band", costs.data.by_band],
                  ["by sector", costs.data.by_sector.slice(0, 8)],
                  ["by model", costs.data.by_model.slice(0, 8)],
                ] as const
              ).map(([label, buckets]) => (
                <div key={label}>
                  <p className="mb-1.5 text-[11px] text-sounding">{label}</p>
                  {buckets.length === 0 ? (
                    <p className="text-[11px] text-sounding/75">no recorded spend</p>
                  ) : (
                    <table className="w-full text-left text-[11px]">
                      <thead className="sr-only">
                        <tr>
                          <th>{label}</th>
                          <th>cost</th>
                          <th>activity</th>
                        </tr>
                      </thead>
                      <tbody>
                        {buckets.map((bucket) => (
                          <tr key={bucket.key} className="border-t border-glass-tint/8">
                            <td className="max-w-40 truncate py-1 pr-3 text-sounding" title={bucket.key}>
                              {humanizeToken(bucket.key)}
                            </td>
                            <td className="font-data py-1 pr-3 text-right text-moonlight">
                              {formatUsd(bucket.cost_usd)}
                            </td>
                            <td className="font-data py-1 text-right text-sounding/85">
                              {bucket.runs ?? bucket.calls ?? 0}
                              {bucket.cached_calls ? ` (${bucket.cached_calls} cached)` : ""}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  )}
                </div>
              ))}
            </div>
          </>
        )}
      </Section>

      {/* Legacy consoles */}
      <Section condenseIndex={4} className="mt-5 p-5">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <h2 className="font-display text-xl text-moonlight">Consoles</h2>
          <span className="font-data text-xs text-sounding">
            full legacy console →
          </span>
        </div>
        {consoles.isLoading ? (
          <p className="voice mt-4">loading…</p>
        ) : !consoles.data ? (
          <p className="mt-4 text-xs text-rose">The consoles could not be read.</p>
        ) : (
          <div className="mt-4 grid grid-cols-1 gap-5 lg:grid-cols-3">
            <div>
              <p className="mb-1.5 text-[11px] text-sounding">
                dead letters · {consoles.data.deadletters.length} ({consoles.data.backlog_size}{" "}
                queued)
              </p>
              {consoles.data.deadletters.length === 0 ? (
                <p className="text-[11px] text-sounding/75">nothing has died</p>
              ) : (
                <ul className="flex flex-col gap-1.5">
                  {consoles.data.deadletters.slice(0, 8).map((row) => (
                    <li key={row.id} className="text-[11px]">
                      <span className="font-data text-moonlight">{row.job_type}</span>
                      <span className="text-rose"> · {row.error_type}</span>
                      <p className="truncate text-sounding" title={row.error_message ?? undefined}>
                        {row.error_message}
                      </p>
                    </li>
                  ))}
                </ul>
              )}
            </div>
            <div>
              <p className="mb-1.5 text-[11px] text-sounding">
                research gaps · {consoles.data.research_gaps.length}
              </p>
              {consoles.data.research_gaps.length === 0 ? (
                <p className="text-[11px] text-sounding/75">no gap signals</p>
              ) : (
                <ul className="flex flex-col gap-1">
                  {consoles.data.research_gaps.slice(0, 8).map((row) => (
                    <li
                      key={`${row.ticker}-${row.as_of_date}-${row.run_id}`}
                      className="flex items-baseline justify-between gap-2 text-[11px]"
                    >
                      <span className="font-data text-moonlight">{row.ticker}</span>
                      <span className="font-data text-sounding">
                        {row.recency_days_min ?? "—"}d stale · {row.item_count_30d ?? 0} items
                      </span>
                    </li>
                  ))}
                </ul>
              )}
            </div>
            <div>
              <p className="mb-1.5 text-[11px] text-sounding">
                pipeline runs · {consoles.data.legacy_runs.length}
              </p>
              {consoles.data.legacy_runs.length === 0 ? (
                <p className="text-[11px] text-sounding/75">no indexed runs</p>
              ) : (
                <ul className="flex flex-col gap-1">
                  {consoles.data.legacy_runs.slice(0, 8).map((row) => (
                    <li key={row.run_id} className="text-[11px]">
                      <span className="font-data text-moonlight">
                        {row.run_id}
                      </span>
                      <span className="text-sounding"> · {row.as_of_date ?? "—"}</span>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </div>
        )}
      </Section>
    </div>
  );
}
