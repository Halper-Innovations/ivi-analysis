/**
 * Run detail — one scan opened on the bench. The funnel narrows stage by
 * stage; any candidate expands into its gate ladder (rule, observed vs
 * threshold, pass/fail) so "why exactly was this screened out" is a
 * ten-second answer. Costs render when the artifact recorded telemetry;
 * "no telemetry" is stated, never faked. The report reads in the Reader.
 */

import { useQuery } from "@tanstack/react-query";
import { Link, useParams } from "@tanstack/react-router";
import { useState } from "react";

import { GradeWord, StatusChip } from "../components/chips";
import { Section } from "../components/GlassPanel";
import type {
  RunCandidate,
  RunCosts,
  RunDetailResponse,
  RunFunnelStage,
} from "../lib/api";
import { fetchRunDetail, fetchRunReport } from "../lib/api";
import { formatPct, humanizeToken } from "../lib/floor";
import { VerdictWord } from "./Scans";
import {
  formatUsd,
  funnelLayout,
  laneBudgetFrac,
  stageTone,
} from "../lib/scans";

const TONE_BAR: Record<string, string> = {
  jade: "bg-jade/35",
  amber: "bg-amber/35",
  rose: "bg-rose/35",
  slate: "bg-slate-mist/25",
  water: "bg-glass-tint/25",
};

const TONE_TEXT: Record<string, string> = {
  jade: "text-jade",
  amber: "text-amber",
  rose: "text-rose",
  slate: "text-slate-mist",
  water: "text-moonlight",
};

function TickerLink({ ticker }: { ticker: string }) {
  return (
    <Link
      to="/company/$ticker"
      params={{ ticker }}
      className="font-data text-sm font-medium text-moonlight hover:text-jade"
    >
      {ticker}
    </Link>
  );
}

/** The narrowing waterfall. Clicking a stage lays out its tickers. */
function Funnel({ stages }: { stages: RunFunnelStage[] }) {
  const [open, setOpen] = useState<string | null>(null);
  const bars = funnelLayout(stages);
  if (bars.length === 0) return <p className="voice">The artifact records no pipeline stages.</p>;
  return (
    <div className="flex flex-col gap-1.5">
      {bars.map((bar) => {
        const tone = stageTone(bar.key);
        const expanded = open === bar.key;
        return (
          <div key={bar.key}>
            <button
              type="button"
              onClick={() => setOpen(expanded ? null : bar.key)}
              className="group flex w-full items-center gap-3 text-left"
              aria-expanded={expanded}
            >
              <span className="w-44 shrink-0 text-xs text-sounding group-hover:text-moonlight">
                {bar.label}
              </span>
              <span className="relative h-5 flex-1">
                <span
                  className={`absolute inset-y-0 left-0 rounded-r ${TONE_BAR[tone]}`}
                  style={{ width: `${bar.frac * 100}%` }}
                />
              </span>
              <span className={`font-data w-10 shrink-0 text-right text-sm ${TONE_TEXT[tone]}`}>
                {bar.count}
              </span>
            </button>
            {expanded && bar.tickers.length > 0 && (
              <div className="mt-1.5 flex flex-wrap gap-1.5 pl-44">
                {bar.tickers.map((ticker) => (
                  <TickerLink key={ticker} ticker={ticker} />
                ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

/** One candidate row; expanding reveals the gate ladder. */
function CandidateRow({ candidate }: { candidate: RunCandidate }) {
  const [open, setOpen] = useState(false);
  const tone = stageTone(candidate.terminal_state);
  const hasDetail =
    candidate.gates.length > 0 ||
    candidate.reason_codes.length > 0 ||
    candidate.underwriting_verdict !== null;
  return (
    <div className="border-t border-glass-tint/8">
      <button
        type="button"
        onClick={() => hasDetail && setOpen(!open)}
        className={`flex w-full items-center gap-3 py-2 text-left ${hasDetail ? "" : "cursor-default"}`}
        aria-expanded={open}
      >
        <span className="w-20 shrink-0">
          <TickerLink ticker={candidate.ticker} />
        </span>
        <span className={`w-44 shrink-0 text-xs lowercase ${TONE_TEXT[tone]}`}>
          {humanizeToken(candidate.terminal_state)}
        </span>
        <span className="flex-1 truncate text-xs text-sounding">
          {candidate.reason_codes.map(humanizeToken).join(" · ") || ""}
        </span>
        {candidate.failed_gates > 0 && (
          <span className="font-data shrink-0 text-xs text-rose">
            {candidate.failed_gates} gate{candidate.failed_gates > 1 ? "s" : ""} failed
          </span>
        )}
        {candidate.underwriting_verdict && (
          <span className="shrink-0">
            <GradeWord value={candidate.underwriting_verdict} />
          </span>
        )}
        {hasDetail && (
          <span className="font-data shrink-0 text-[10px] text-sounding">
            {open ? "−" : "+"}
          </span>
        )}
      </button>
      {open && (
        <div className="pb-3 pl-20">
          {candidate.gates.length > 0 ? (
            <table className="w-full border-collapse text-left">
              <thead>
                <tr className="text-[10px] uppercase tracking-wider text-sounding">
                  <th className="py-1.5 pr-3 font-medium">gate</th>
                  <th className="py-1.5 pr-3 font-medium">observed</th>
                  <th className="py-1.5 pr-3 font-medium">threshold</th>
                  <th className="py-1.5 pr-0 font-medium">result</th>
                </tr>
              </thead>
              <tbody>
                {candidate.gates.map((gate) => (
                  <tr key={gate.rule_id} className="border-t border-glass-tint/8 text-xs">
                    <td className="py-1.5 pr-3 text-sounding" title={gate.notes.join(" ")}>
                      {humanizeToken(gate.rule_id)}
                    </td>
                    <td className="font-data py-1.5 pr-3 text-moonlight">
                      {gate.observed_value ?? "—"}
                    </td>
                    <td className="font-data py-1.5 pr-3 text-sounding">
                      {gate.threshold ?? "—"}
                    </td>
                    <td className="py-1.5 pr-0">
                      {!gate.applicable ? (
                        <span className="text-xs text-slate-mist">not applicable</span>
                      ) : gate.status === "PASS" ? (
                        <span className="text-xs text-jade">● pass</span>
                      ) : gate.status === "FAIL" ? (
                        <span className="text-xs text-rose" title={gate.reason_code ?? undefined}>
                          ○ fail
                        </span>
                      ) : (
                        <span className="text-xs text-slate-mist">{gate.status.toLowerCase()}</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p className="text-xs text-sounding">No gate evaluations recorded for this stage.</p>
          )}
          {(candidate.underwriting_verdict || candidate.frontier_status) && (
            <p className="mt-2 text-xs text-sounding">
              {candidate.underwriting_verdict && (
                <>
                  underwriting: <GradeWord value={candidate.underwriting_verdict} />
                  {candidate.underwriting_confidence &&
                    ` · ${candidate.underwriting_confidence.toLowerCase()} confidence`}
                </>
              )}
              {candidate.frontier_status && (
                <span className="ml-2">
                  {humanizeToken(candidate.frontier_status)}
                  {candidate.frontier_dominated_by &&
                    ` (dominated by ${candidate.frontier_dominated_by})`}
                </span>
              )}
            </p>
          )}
        </div>
      )}
    </div>
  );
}

function CostsPanel({ costs }: { costs: RunCosts }) {
  if (!costs.available) {
    return <p className="voice">No cost telemetry recorded for this run.</p>;
  }
  return (
    <div>
      <p className="font-data text-2xl text-moonlight">
        {formatUsd(costs.aggregate_cost_usd)}
        <span className="ml-2 text-xs text-sounding">total recorded spend</span>
      </p>
      {costs.lanes.length > 0 && (
        <div className="mt-4 flex flex-col gap-2">
          {costs.lanes.map((lane) => {
            const frac = laneBudgetFrac(lane.cost_microdollars, lane.max_cost_microdollars);
            return (
              <div key={lane.lane} className="flex items-center gap-3">
                <span className="w-44 shrink-0 text-xs text-sounding">
                  {humanizeToken(lane.lane)}
                </span>
                <span className="relative h-2.5 flex-1 overflow-hidden rounded bg-deepwater">
                  {frac !== null && (
                    <span
                      className="absolute inset-y-0 left-0 rounded bg-glass-tint/50"
                      style={{ width: `${frac * 100}%` }}
                    />
                  )}
                </span>
                <span className="font-data w-28 shrink-0 text-right text-xs text-moonlight">
                  {formatUsd(lane.cost_usd)}
                  {lane.max_cost_usd !== null && (
                    <span className="text-sounding"> / {formatUsd(lane.max_cost_usd)}</span>
                  )}
                </span>
              </div>
            );
          })}
        </div>
      )}
      {costs.providers.length > 0 && (
        <table className="mt-4 w-full border-collapse text-left">
          <thead>
            <tr className="text-[10px] uppercase tracking-wider text-sounding">
              <th className="py-1.5 pr-3 font-medium">provider · model</th>
              <th className="py-1.5 pr-3 text-right font-medium">calls</th>
              <th className="py-1.5 pr-3 text-right font-medium">input</th>
              <th className="py-1.5 pr-3 text-right font-medium">cached</th>
              <th className="py-1.5 pr-3 text-right font-medium">output</th>
              <th className="py-1.5 pr-0 text-right font-medium">cost</th>
            </tr>
          </thead>
          <tbody>
            {costs.providers.map((provider) => (
              <tr
                key={`${provider.provider}-${provider.model}`}
                className="border-t border-glass-tint/8 text-xs"
              >
                <td className="font-data py-1.5 pr-3 text-moonlight">
                  {provider.provider} · {provider.model}
                </td>
                <td className="font-data py-1.5 pr-3 text-right text-sounding">
                  {provider.calls}
                  {provider.failed_calls > 0 && (
                    <span className="text-rose"> ({provider.failed_calls} failed)</span>
                  )}
                </td>
                <td className="font-data py-1.5 pr-3 text-right text-sounding">
                  {provider.input_tokens.toLocaleString()}
                </td>
                <td className="font-data py-1.5 pr-3 text-right text-sounding">
                  {provider.cached_input_tokens.toLocaleString()}
                </td>
                <td className="font-data py-1.5 pr-3 text-right text-sounding">
                  {provider.output_tokens.toLocaleString()}
                </td>
                <td className="font-data py-1.5 pr-0 text-right text-moonlight">
                  {formatUsd(provider.cost_estimate_usd)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

function Report({ slug }: { slug: string }) {
  const [open, setOpen] = useState(false);
  const report = useQuery({
    queryKey: ["run-report", slug],
    queryFn: () => fetchRunReport(slug),
    enabled: open,
    staleTime: Infinity,
    retry: 0,
  });
  if (!open) {
    return (
      <button
        type="button"
        onClick={() => setOpen(true)}
        className="glass rounded-xl px-4 py-2 text-xs text-sounding hover:text-moonlight"
      >
        Open the report
      </button>
    );
  }
  if (report.isLoading) return <p className="voice">loading…</p>;
  if (report.error || !report.data)
    return <p className="text-xs text-rose">The report could not be read from disk.</p>;
  return (
    <div
      className="reader max-h-[48rem] overflow-y-auto pr-2"
      dangerouslySetInnerHTML={{ __html: report.data.html }}
    />
  );
}

function TickerChips({ label, tickers }: { label: string; tickers: string[] }) {
  const [open, setOpen] = useState(false);
  if (tickers.length === 0) return null;
  return (
    <div className="text-xs text-sounding">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="hover:text-moonlight"
        aria-expanded={open}
      >
        {label}: <span className="font-data text-moonlight">{tickers.length}</span>{" "}
        <span className="font-data text-[10px]">{open ? "−" : "+"}</span>
      </button>
      {open && (
        <div className="mt-1.5 flex flex-wrap gap-1.5">
          {tickers.map((ticker) => (
            <TickerLink key={ticker} ticker={ticker} />
          ))}
        </div>
      )}
    </div>
  );
}

export function RunDetail() {
  const { _splat: ref = "" } = useParams({ from: "/scans/run/$" });
  const query = useQuery<RunDetailResponse>({
    queryKey: ["run", ref],
    queryFn: () => fetchRunDetail(ref),
    retry: 1,
  });

  if (query.isLoading) {
    return (
      <Section className="max-w-xl p-8">
        <p className="voice">loading…</p>
      </Section>
    );
  }
  if (query.error || !query.data) {
    return (
      <Section className="max-w-xl p-8">
        <h1 className="font-display text-2xl text-moonlight">Run not found</h1>
        <p className="font-data mt-3 break-all text-xs text-sounding">{ref}</p>
        <Link to="/scans" className="mt-4 inline-block text-xs text-jade">
          ← back to the gallery
        </Link>
      </Section>
    );
  }

  const detail = query.data;
  const { summary, decision, costs, selection } = detail;
  const isV2 = detail.candidates.length > 0;

  if (detail.quarantined) {
    return (
      <div className="max-w-4xl">
        <p className="eyebrow">excluded sector-run history</p>
        <h1 className="font-display mt-1 text-3xl text-moonlight">{summary.run_id}</h1>
        <Section condenseIndex={0} className="mt-6 border-rose/30 p-6">
          <p className="text-sm text-rose">
            This artifact could not be parsed and is excluded from current decisions.
          </p>
          <p className="mt-2 text-xs text-sounding">
            financial integrity:{" "}
            <span className="font-data text-rose">{summary.integrity_status}</span>
          </p>
          <p className="font-data mt-2 break-all text-xs text-sounding">{detail.parse_error}</p>
          <p className="font-data mt-2 break-all text-[10px] text-sounding/85">{summary.path}</p>
        </Section>
      </div>
    );
  }

  if (!summary.decision_eligible) {
    return (
      <div className="max-w-4xl">
        <p className="eyebrow">excluded sector-run history</p>
        <h1 className="font-display mt-1 text-3xl text-moonlight">
          {summary.sector ? humanizeToken(summary.sector) : summary.run_id}
        </h1>
        <Section condenseIndex={0} className="mt-6 border-amber/30 p-6">
          <p className="text-sm text-amber">Historical artifact — excluded from current decisions.</p>
          <p className="mt-2 text-xs text-sounding">
            financial integrity:{" "}
            <span className="font-data text-amber">{summary.integrity_status}</span>
          </p>
          <p className="mt-2 text-xs leading-relaxed text-sounding">
            Candidate rankings, valuation packets, selection, verdict, and watchlist links are
            suppressed. This page preserves only audit metadata.
          </p>
          <p className="font-data mt-3 break-all text-[10px] text-sounding/85">{summary.path}</p>
        </Section>
        {detail.report_available && (
          <Section condenseIndex={1} className="mt-4 p-5">
            <p className="eyebrow mb-3">suppressed report record</p>
            <Report slug={summary.slug ?? summary.run_id} />
          </Section>
        )}
        <Link to="/scans" search={{ history: "all" }} className="mt-4 inline-block text-xs text-jade">
          ← back to excluded history
        </Link>
      </div>
    );
  }

  return (
    <div className="max-w-[80rem]">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <p className="eyebrow">
            sector run · {summary.pipeline_version ?? "v1"}
            {summary.scan_family && summary.scan_family !== "normal"
              ? ` · ${summary.scan_family}`
              : ""}
          </p>
          <h1 className="font-display mt-1 text-4xl text-moonlight">
            {summary.sector ? humanizeToken(summary.sector) : summary.run_id}
          </h1>
          <p className="font-data mt-2 text-xs text-sounding" title={summary.run_id}>
            {summary.created_at?.slice(0, 10) ?? summary.as_of_date ?? "undated"}
            {summary.market_cap_focus ? ` · ${humanizeToken(summary.market_cap_focus)}` : ""}
            {costs.available ? ` · ${formatUsd(costs.aggregate_cost_usd)}` : ""}
          </p>
        </div>
        <div className="text-right">
          <VerdictWord verdict={decision.final_verdict} />
          {decision.selected_ticker && (
            <p className="mt-1">
              <TickerLink ticker={decision.selected_ticker} />
            </p>
          )}
        </div>
      </div>

      <Section condenseIndex={0} className="mt-7 p-5">
        <p className="eyebrow mb-4">selection funnel</p>
        <Funnel stages={detail.funnel} />
        <div className="mt-4 flex flex-wrap gap-x-6 gap-y-2 border-t border-glass-tint/8 pt-3">
          {selection.source && (
            <span className="text-xs text-sounding">
              source: <span className="font-data">{selection.source}</span>
            </span>
          )}
          <TickerChips label="excluded at intake" tickers={selection.excluded_tickers} />
          <TickerChips label="admitted" tickers={selection.admitted_tickers} />
        </div>
        {selection.warnings.length > 0 && (
          <div className="mt-2">
            <p className="text-[11px] text-amber">
              {selection.warnings.length} intake note
              {selection.warnings.length > 1 ? "s" : ""}
            </p>
            <ul className="font-data mt-1 break-all text-[10px] leading-relaxed text-sounding">
              {selection.warnings.map((warning) => (
                <li key={warning}>{warning}</li>
              ))}
            </ul>
          </div>
        )}
      </Section>

      {isV2 && (
        <Section condenseIndex={1} className="mt-4 p-5">
          <p className="eyebrow mb-2">candidates · gate-by-gate detail</p>
          <p className="mb-3 text-[11px] text-sounding">
            expand a name to see every gate it faced — observed value against threshold
          </p>
          <div>
            {detail.candidates.map((candidate) => (
              <CandidateRow key={candidate.ticker} candidate={candidate} />
            ))}
          </div>
        </Section>
      )}

      {detail.ranking.length > 0 && (
        <Section condenseIndex={isV2 ? 2 : 1} className="mt-4 p-5">
          <p className="eyebrow mb-3">relative ranking</p>
          <div className="overflow-x-auto">
            <table className="w-full border-collapse text-left">
              <thead>
                <tr className="text-[10px] uppercase tracking-wider text-sounding">
                  <th className="py-2 pr-3 font-medium">#</th>
                  <th className="py-2 pr-3 font-medium">ticker</th>
                  <th className="py-2 pr-3 font-medium">audit</th>
                  <th className="py-2 pr-3 text-right font-medium">base return</th>
                  <th className="py-2 pr-3 font-medium">blockers</th>
                  <th className="py-2 pr-0 font-medium">positioning</th>
                </tr>
              </thead>
              <tbody>
                {detail.ranking.map((row) => (
                  <tr key={row.ticker} className="border-t border-glass-tint/8 text-xs">
                    <td className="font-data py-2 pr-3 text-sounding">{row.rank ?? "—"}</td>
                    <td className="py-2 pr-3">
                      <TickerLink ticker={row.ticker} />
                    </td>
                    <td className="py-2 pr-3">
                      <GradeWord value={row.audit_status} />
                    </td>
                    <td className="font-data py-2 pr-3 text-right text-moonlight">
                      {formatPct(
                        row.best_base_annualized_return !== null
                          ? row.best_base_annualized_return * 100
                          : null,
                      )}
                    </td>
                    <td className="py-2 pr-3 text-sounding">
                      {row.hard_blockers.map(humanizeToken).join(" · ") || "—"}
                    </td>
                    <td
                      className="max-w-[22rem] truncate py-2 pr-0 text-sounding"
                      title={row.positioning_summary ?? undefined}
                    >
                      {row.positioning_summary ?? "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Section>
      )}

      <div className="mt-4 grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Section condenseIndex={3} className="p-5">
          <p className="eyebrow mb-3">the decision</p>
          <VerdictWord verdict={decision.final_verdict} />
          {decision.no_selection_reason && (
            <p className="mt-2 text-sm text-sounding">{decision.no_selection_reason}</p>
          )}
          {decision.confidence && (
            <p className="mt-2 text-xs text-sounding">
              confidence: <span className="lowercase">{decision.confidence.toLowerCase()}</span>
            </p>
          )}
          {decision.confidence_cap_reasons.length > 0 && (
            <ul className="mt-2 list-disc pl-4 text-xs text-sounding">
              {decision.confidence_cap_reasons.map((reason) => (
                <li key={reason}>{reason}</li>
              ))}
            </ul>
          )}
          {detail.watchlist_rows.length > 0 && (
            <div className="mt-4 border-t border-glass-tint/8 pt-3">
              <p className="mb-2 text-[11px] text-sounding">watchlist rows this run produced</p>
              <div className="flex flex-wrap gap-x-4 gap-y-1.5">
                {detail.watchlist_rows.map((row) => (
                  <span key={row.watchlist_id} className="flex items-center gap-1.5">
                    <TickerLink ticker={row.ticker} />
                    <StatusChip value={row.status} />
                  </span>
                ))}
              </div>
            </div>
          )}
        </Section>

        <Section condenseIndex={4} className="p-5">
          <p className="eyebrow mb-3">what it cost</p>
          <CostsPanel costs={costs} />
        </Section>
      </div>

      {detail.report_available && (
        <Section condenseIndex={5} className="mt-4 p-5">
          <div className="mb-3 flex items-baseline justify-between gap-3">
            <p className="eyebrow">the report</p>
            {summary.report_path && (
              <Link
                to="/reader"
                search={{ path: summary.report_path }}
                className="text-xs text-sounding hover:text-jade"
              >
                read in the Reader →
              </Link>
            )}
          </div>
          <Report slug={summary.slug ?? summary.run_id} />
        </Section>
      )}
    </div>
  );
}
