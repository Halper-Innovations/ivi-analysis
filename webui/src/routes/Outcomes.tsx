/**
 * Outcomes — calibration. The paper record (closed scan decisions measured
 * against the benchmark), the method scoreboard (which lens earns trust),
 * the live-decision journal with its standing goal, and stored calibration
 * reports.
 *
 * Chart discipline: the excess-return histogram encodes polarity with the
 * app's status poles (rose below / jade above) — sign is never color-alone:
 * the bins sit left/right of an emphasized zero divider with labeled edges
 * (the deutan ΔE 6.3 pair is legal with that positional encoding). Method
 * identity reuses the validated categorical trio from the evolution chart.
 */

import { useQuery } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";

import { Section } from "../components/GlassPanel";
import type {
  HistogramBin,
  JournalEntry,
  MethodScore,
  OutcomesResponse,
  ReturnStats,
} from "../lib/api";
import { OfflineError, fetchOutcomes } from "../lib/api";
import { formatPct, humanizeToken } from "../lib/floor";

/** Validated categorical trio (evolution chart) — method identity only. */
const METHOD_HUES: Record<string, string> = {
  dcf: "#3987e5",
  epv: "#d55181",
  graham: "#c98500",
};

function formatRate(rate: number | null): string {
  return rate === null ? "—" : `${(rate * 100).toFixed(1)}%`;
}

/* ------------------------------------------------------------------ */
/* Excess-return histogram                                             */
/* ------------------------------------------------------------------ */

function edgeLabel(bin: HistogramBin): string {
  if (bin.low === null) return `≤${bin.high}`;
  if (bin.high === null) return `≥${bin.low}`;
  return `${bin.low}…${bin.high}`;
}

function ExcessHistogram({ bins }: { bins: HistogramBin[] }) {
  const width = 640;
  const height = 190;
  const pad = { top: 20, right: 8, bottom: 22, left: 8 };
  const max = Math.max(...bins.map((b) => b.count), 1);
  const innerW = width - pad.left - pad.right;
  const barW = innerW / bins.length;
  const y = (count: number) =>
    height - pad.bottom - (count / max) * (height - pad.top - pad.bottom);
  // Zero sits between the last negative bin (high === 0) and the first
  // positive one; polarity is positional first, hue second.
  const zeroIndex = bins.findIndex((b) => b.high !== null && b.high === 0) + 1;
  const zeroX = pad.left + zeroIndex * barW;
  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      className="w-full"
      role="img"
      aria-label="Distribution of excess returns versus benchmark, percent buckets"
    >
      {bins.map((bin, index) => {
        const negative = bin.high !== null && bin.high <= 0;
        const hue = negative ? "var(--color-rose)" : "var(--color-jade)";
        // Opacity ramps outward from zero so the tails read as the extremes.
        const rank = negative ? zeroIndex - 1 - index : index - zeroIndex;
        const opacity = 0.45 + Math.min(rank, 4) * 0.11;
        const x = pad.left + index * barW;
        const top = y(bin.count);
        const barH = height - pad.bottom - top;
        return (
          <g key={index}>
            <rect
              x={x + 1}
              width={barW - 2}
              y={top}
              height={Math.max(barH, bin.count > 0 ? 2 : 0)}
              rx={4}
              fill={hue}
              fillOpacity={opacity}
            >
              <title>
                {edgeLabel(bin)}% excess: {bin.count.toLocaleString("en-US")} outcomes
              </title>
            </rect>
            {bin.count > 0 && (
              <text
                x={x + barW / 2}
                y={top - 5}
                textAnchor="middle"
                fontSize={8.5}
                fill="var(--color-sounding)"
                className="font-data"
              >
                {bin.count >= 1000 ? `${(bin.count / 1000).toFixed(1)}k` : bin.count}
              </text>
            )}
          </g>
        );
      })}
      <line
        x1={zeroX}
        x2={zeroX}
        y1={pad.top - 8}
        y2={height - pad.bottom}
        stroke="var(--color-moonlight)"
        strokeOpacity={0.4}
        strokeDasharray="2 3"
      />
      <text
        x={zeroX}
        y={height - pad.bottom + 12}
        textAnchor="middle"
        fontSize={9}
        fill="var(--color-moonlight)"
        className="font-data"
      >
        0
      </text>
      {bins.map((bin, index) => {
        if (bin.low === null || bin.low === 0) return null;
        const x = pad.left + index * barW;
        return (
          <text
            key={`edge-${index}`}
            x={x}
            y={height - pad.bottom + 12}
            textAnchor="middle"
            fontSize={8}
            fill="var(--color-sounding)"
            className="font-data"
          >
            {bin.low > 0 ? `+${bin.low}` : bin.low}
          </text>
        );
      })}
      <line
        x1={pad.left}
        x2={width - pad.right}
        y1={height - pad.bottom}
        y2={height - pad.bottom}
        stroke="var(--color-glass-tint)"
        strokeOpacity={0.15}
      />
    </svg>
  );
}

/* ------------------------------------------------------------------ */
/* Tiles                                                               */
/* ------------------------------------------------------------------ */

function StatsTile({ label, stats }: { label: string; stats: ReturnStats }) {
  return (
    <div className="glass rounded-xl p-4">
      <p className="eyebrow text-[10px]">{label}</p>
      <p className="font-data mt-1.5 text-lg text-moonlight">
        <span className={stats.avg_excess !== null && stats.avg_excess >= 0 ? "text-jade" : "text-rose"}>
          {formatPct(stats.avg_excess)}
        </span>{" "}
        <span className="text-xs text-sounding">avg excess</span>
      </p>
      <p className="font-data mt-1 text-[11px] text-sounding">
        {stats.n.toLocaleString("en-US")} closed · beats benchmark {formatRate(stats.hit_rate)}
        {stats.avg_realized !== null && ` · raw ${formatPct(stats.avg_realized)}`}
      </p>
    </div>
  );
}

function MethodTile({ score }: { score: MethodScore }) {
  const resolved = score.correct + score.incorrect;
  return (
    <div className="glass rounded-xl p-4">
      <p className="font-data flex items-center gap-1.5 text-xs text-moonlight">
        <span
          className="inline-block h-2 w-2 rounded-sm"
          style={{ background: METHOD_HUES[score.method] ?? "var(--color-glass-tint)" }}
        />
        {score.method}
        {score.unadjusted_source && (
          <span className="text-[9px] text-sounding" title="evaluated on the unadjusted estimate">
            unadjusted
          </span>
        )}
      </p>
      <p className="font-data mt-1.5 text-xl text-moonlight">{formatRate(score.hit_rate)}</p>
      <p className="font-data mt-1 text-[10px] text-sounding">
        {score.correct} correct · {score.incorrect} incorrect · {score.inconclusive}{" "}
        inconclusive
        {resolved === 0 && " — nothing resolved yet"}
      </p>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Journal                                                             */
/* ------------------------------------------------------------------ */

function JournalCard({ entry }: { entry: JournalEntry }) {
  return (
    <div className="glass rounded-xl p-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <Link
          to="/company/$ticker"
          params={{ ticker: entry.ticker }}
          search={{ tab: "decisions", basis: undefined }}
          className="font-data text-sm text-moonlight underline-offset-4 hover:underline"
        >
          {entry.ticker}
        </Link>
        <span className="font-data text-[10px] uppercase text-sounding">
          {entry.status} · {entry.decided_at?.slice(0, 10)}
        </span>
      </div>
      <p className="font-data mt-1 text-[10px] text-sounding">
        {humanizeToken(entry.kind)}
        {entry.reason_code && ` · ${humanizeToken(entry.reason_code)}`}
        {entry.operator && ` · by ${entry.operator}`}
      </p>
      {entry.rationale && (
        <p className="mt-1.5 max-w-[68ch] text-xs leading-relaxed text-moonlight/85">
          {entry.rationale}
        </p>
      )}
      {(entry.intended_size || entry.sizing_rationale) && (
        <p className="mt-1 text-[11px] text-sounding">
          {entry.intended_size && <span className="font-data">{entry.intended_size}</span>}
          {entry.intended_size && entry.sizing_rationale && " — "}
          {entry.sizing_rationale}
        </p>
      )}
    </div>
  );
}

function GoalPips({ decided, goal, deadline }: { decided: number; goal: number; deadline: string }) {
  return (
    <div className="flex items-center gap-3">
      <div className="flex gap-1.5" aria-hidden="true">
        {Array.from({ length: goal }, (_, index) => (
          <span
            key={index}
            className={`inline-block h-2.5 w-2.5 rounded-full ${
              index < decided ? "bg-jade" : "border border-sounding/40"
            }`}
          />
        ))}
      </div>
      <p className="font-data text-[11px] text-sounding">
        {Math.min(decided, goal)} of {goal} live decisions journaled · goal by {deadline}
      </p>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* The page                                                            */
/* ------------------------------------------------------------------ */

export function Outcomes() {
  const query = useQuery({ queryKey: ["outcomes"], queryFn: fetchOutcomes, retry: 1 });

  if (query.error instanceof OfflineError) {
    return (
      <Section className="max-w-xl p-8">
        <h1 className="font-display text-2xl text-moonlight">IVI offline</h1>
        <p className="font-data mt-3 text-sm text-rose">{query.error.precondition}</p>
      </Section>
    );
  }
  const data: OutcomesResponse | undefined = query.data;
  if (!data) {
    return <p className="text-sm text-sounding">loading…</p>;
  }

  const { scoreboard, returns, journal, calibration } = data;

  return (
    <div className="max-w-6xl">
      <h1 className="font-display text-4xl text-moonlight">Outcomes</h1>
      <p className="voice mt-2">
        realized results by valuation method — the paper record and the live journal
      </p>

      {/* The live journal leads: those are real decisions. */}
      <Section condenseIndex={0} className="mt-7 p-6">
        <div className="flex flex-wrap items-baseline justify-between gap-3">
          <p className="eyebrow">live decision journal</p>
          <GoalPips decided={journal.decided_count} goal={journal.goal} deadline={journal.goal_deadline} />
        </div>
        {journal.entries.length === 0 ? (
          <p className="voice mt-3 text-sm">
            nothing journaled yet — decisions land here as dispositions close.
          </p>
        ) : (
          <div className="mt-4 grid grid-cols-1 gap-3 lg:grid-cols-2">
            {journal.entries.map((entry) => (
              <JournalCard key={entry.id} entry={entry} />
            ))}
          </div>
        )}
        {journal.open_count > 0 && (
          <p className="font-data mt-3 text-[11px] text-sounding">
            {journal.open_count} disposition{journal.open_count === 1 ? "" : "s"} still open —{" "}
            <Link to="/" className="text-jade underline-offset-4 hover:underline">
              decide on Today →
            </Link>
          </p>
        )}
      </Section>

      {/* Method scoreboard */}
      <Section condenseIndex={1} className="mt-5 p-6">
        <p className="eyebrow">method scoreboard — hit rate by valuation method</p>
        {scoreboard.methods.length === 0 ? (
          <p className="voice mt-3 text-sm">
            no resolved method outcomes yet — the resolver grades dcf / epv / graham
            as research horizons close.
          </p>
        ) : (
          <>
            <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-3">
              {scoreboard.methods.map((score) => (
                <MethodTile key={score.method} score={score} />
              ))}
            </div>
            {scoreboard.monthly.length > 0 && (
              <details className="mt-4">
                <summary className="eyebrow cursor-pointer text-[10px]">
                  by month ({scoreboard.monthly.length})
                </summary>
                <table className="mt-2 w-full max-w-md border-collapse text-left">
                  <thead>
                    <tr className="text-[10px] uppercase tracking-wider text-sounding">
                      <th className="py-1 pr-3 font-medium">month</th>
                      <th className="py-1 pr-3 font-medium">method</th>
                      <th className="py-1 pr-3 text-right font-medium">correct</th>
                      <th className="py-1 pr-3 text-right font-medium">incorrect</th>
                      <th className="py-1 pr-0 text-right font-medium">n</th>
                    </tr>
                  </thead>
                  <tbody>
                    {scoreboard.monthly.map((row, index) => (
                      <tr key={index} className="border-t border-glass-tint/8 text-xs">
                        <td className="font-data py-1 pr-3 text-sounding">{row.month}</td>
                        <td className="font-data py-1 pr-3 text-moonlight">{row.method}</td>
                        <td className="font-data py-1 pr-3 text-right text-jade">{row.correct}</td>
                        <td className="font-data py-1 pr-3 text-right text-rose">{row.incorrect}</td>
                        <td className="font-data py-1 pr-0 text-right text-sounding">{row.n}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </details>
            )}
          </>
        )}
      </Section>

      {/* Paper record */}
      <Section condenseIndex={2} className="mt-5 p-6">
        <p className="eyebrow">realized returns — the paper record</p>
        <p className="voice mt-1 text-sm">
          closed scan decisions vs benchmark; not live trades.
        </p>
        {returns.overall.n === 0 ? (
          <p className="voice mt-3 text-sm">no closed outcomes with a benchmark yet.</p>
        ) : (
          <>
            <div className="mt-4">
              <p className="font-data text-[10px] text-sounding">
                excess return distribution, % vs benchmark
              </p>
              <div className="mt-1">
                <ExcessHistogram bins={returns.histogram} />
              </div>
            </div>
            <div className="mt-5 grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-3">
              <StatsTile label="all closed" stats={returns.overall} />
              {returns.by_decision.map((stats) => (
                <StatsTile
                  key={stats.decision}
                  label={`decision ${stats.decision}`}
                  stats={stats}
                />
              ))}
            </div>
            {returns.by_conviction.length > 0 && (
              <div className="mt-5">
                <p className="eyebrow text-[10px]">conviction vs outcome</p>
                <table className="mt-2 w-full max-w-lg border-collapse text-left">
                  <thead>
                    <tr className="text-[10px] uppercase tracking-wider text-sounding">
                      <th className="py-1 pr-3 font-medium">conviction</th>
                      <th className="py-1 pr-3 text-right font-medium">n</th>
                      <th className="py-1 pr-3 text-right font-medium">avg excess</th>
                      <th className="py-1 pr-3 text-right font-medium">median</th>
                      <th className="py-1 pr-0 text-right font-medium">beats benchmark</th>
                    </tr>
                  </thead>
                  <tbody>
                    {returns.by_conviction.map((row) => (
                      <tr key={row.conviction} className="border-t border-glass-tint/8 text-xs">
                        <td className="font-data py-1.5 pr-3 text-moonlight">c{row.conviction}</td>
                        <td className="font-data py-1.5 pr-3 text-right text-sounding">
                          {row.n.toLocaleString("en-US")}
                        </td>
                        <td
                          className={`font-data py-1.5 pr-3 text-right ${
                            (row.avg_excess ?? 0) >= 0 ? "text-jade" : "text-rose"
                          }`}
                        >
                          {formatPct(row.avg_excess)}
                        </td>
                        <td className="font-data py-1.5 pr-3 text-right text-sounding">
                          {formatPct(row.median_excess)}
                        </td>
                        <td className="font-data py-1.5 pr-0 text-right text-sounding">
                          {formatRate(row.hit_rate)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </>
        )}
      </Section>

      {/* Calibration reports */}
      <Section condenseIndex={3} className="mt-5 p-6">
        <p className="eyebrow">calibration reports</p>
        {calibration.length === 0 ? (
          <p className="voice mt-3 text-sm">no stored calibration reports.</p>
        ) : (
          <div className="mt-3 space-y-3">
            {calibration.map((report) => (
              <div key={report.run_id} className="glass rounded-xl p-4">
                <div className="flex flex-wrap items-baseline justify-between gap-2">
                  <span className="font-data text-xs text-moonlight">{report.run_id}</span>
                  <span className="font-data text-[10px] text-sounding">
                    {humanizeToken(report.report_family)} · {report.as_of_date}
                  </span>
                </div>
                <p className="font-data mt-2 text-[11px] text-sounding">
                  {report.overall.n === null || report.overall.n === 0 ? (
                    <span className="voice">empty window — no closed outcomes in scope</span>
                  ) : (
                    <>
                      n {report.overall.n} · hit {formatRate(report.overall.hit_rate)} · excess
                      hit {formatRate(report.overall.excess_hit_rate)} · avg{" "}
                      {formatPct(report.overall.avg_return)} · avg excess{" "}
                      {formatPct(report.overall.avg_excess)}
                    </>
                  )}
                  {report.headline_hit_metric && (
                    <span className="text-sounding/80"> · headline: {report.headline_hit_metric}</span>
                  )}
                </p>
              </div>
            ))}
          </div>
        )}
      </Section>
    </div>
  );
}
