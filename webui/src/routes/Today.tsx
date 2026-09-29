/**
 * Today — daily review. Health first, then the waterline (the hero: the
 * water around the buy line), the decisions the platform has queued, the
 * gate-blocked lane, and the daily digest.
 *
 * The friction mechanic is deliberate: a decision card's journal command
 * stays disabled until the falsifiers and the pre-mortem have both been
 * expanded — the UI physically makes you re-read why you'd be wrong.
 */

import { useQuery } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import type { CSSProperties } from "react";
import { Fragment, useState } from "react";

import { CopyChip, GradeWord, StatusChip, Term } from "../components/chips";
import { Section } from "../components/GlassPanel";
import { WaterlineStrip } from "../components/WaterlineStrip";
import type { DecisionCard, HealthBlock } from "../lib/api";
import { OfflineError, fetchToday } from "../lib/api";
import {
  formatAge,
  formatMoneyCompact,
  formatPct,
  humanizeToken,
} from "../lib/floor";

function EventPendingTerms({ raw }: { raw: string | null }) {
  if (!raw) return null;
  const seen = new Set<string>();
  const tokens = raw
    .split(",")
    .map((part) => part.trim().replace(/^EVENT_PENDING:/i, ""))
    .filter((token) => {
      const key = token.toUpperCase();
      if (!token || seen.has(key)) return false;
      seen.add(key);
      return true;
    });
  return tokens.map((token, index) => (
    <Fragment key={token.toUpperCase()}>
      {index > 0 ? " · " : null}
      <Term token={token} label={humanizeToken(token)} />
    </Fragment>
  ));
}

function OfflinePanel({ error }: { error: OfflineError }) {
  return (
    <Section className="max-w-xl p-8">
      <h1 className="font-display text-2xl text-moonlight">IVI offline</h1>
      <p className="mt-3 text-sm text-sounding">
        A precondition for reading the books of record failed:
      </p>
      <p className="font-data mt-2 text-sm text-rose">{error.precondition}</p>
      <p className="font-data mt-1 break-all text-xs text-sounding">{error.detail}</p>
      <p className="mt-4 text-sm text-sounding">
        Check the data volume is mounted, then run{" "}
        <code className="font-data text-glass-tint">ivi ops preflight</code>.
      </p>
    </Section>
  );
}

/** One chip (the state), one sentence, and a quiet count. Failing checks
 *  keep their detail lines; passing check names live in the tally tooltip. */
function HealthStrip({ health }: { health: HealthBlock }) {
  const failing = health.checks.filter((c) => !c.ok);
  const passing = health.checks.length - failing.length;
  return (
    <Section condenseIndex={0} className="px-1 py-3.5">
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <StatusChip
          value={health.state}
          title={health.blocking ? "database integrity failure" : undefined}
        />
        <span className="text-xs text-sounding">
          {health.state === "GREEN"
            ? "all checks green"
            : health.blocking
              ? "database integrity check failed — at-target sections are suppressed"
              : "data freshness needs attention — valuation and gating are unaffected"}
        </span>
        <span
          className="ml-auto text-xs text-sounding/80"
          title={health.checks
            .filter((c) => c.ok)
            .map((c) => c.name)
            .join("\n")}
        >
          {passing} of {health.checks.length} checks passing
        </span>
      </div>
      {failing.length > 0 && (
        <ul className="mt-1 space-y-0.5">
          {failing.map((check) => (
            <li
              key={check.name}
              className={`text-xs ${health.blocking ? "text-rose/90" : "text-amber/90"}`}
            >
              {humanizeToken(check.name)} — {check.detail}
            </li>
          ))}
        </ul>
      )}
    </Section>
  );
}

function DecisionCardView({ decision }: { decision: DecisionCard }) {
  const [falsifiersRead, setFalsifiersRead] = useState(false);
  const [preMortemRead, setPreMortemRead] = useState(false);
  const armed = falsifiersRead && preMortemRead;
  const inZone =
    decision.distance_from_buy_pct !== null && decision.distance_from_buy_pct <= 0;

  return (
    <div className="glass p-4">
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <Link
          to="/company/$ticker"
          params={{ ticker: decision.ticker }}
          className="font-display text-xl text-moonlight hover:text-jade"
        >
          {decision.ticker}
        </Link>
        <StatusChip value={decision.watchlist_status} />
        <GradeWord value={decision.conviction_grade} />
        {decision.source_state === "SUPERSEDED_STALE_SOURCE" && (
          <span
            className="text-xs text-amber"
            title="The source assessment is no longer authoritative; explicit PASSED or DEFERRED closure required."
          >
            superseded source
          </span>
        )}
        <span className="font-data ml-auto text-xs text-sounding">
          opened {decision.opened_at?.slice(0, 10) ?? "—"}
        </span>
      </div>

      <div className="font-data mt-2 flex flex-wrap gap-x-5 gap-y-1 text-xs text-sounding">
        <span>
          price <span className="text-moonlight">{formatMoneyCompact(decision.latest_price)}</span>
          <span className="ml-1 text-[10px]">({formatAge(decision.price_age_hours)})</span>
        </span>
        <span>
          target <span className="text-moonlight">{formatMoneyCompact(decision.buy_price_target)}</span>
        </span>
        <span>
          distance{" "}
          <span className={inZone ? "text-jade" : "text-moonlight"}>
            {formatPct(decision.distance_from_buy_pct)}
          </span>
        </span>
        {decision.confidence && <span>confidence {decision.confidence.toLowerCase()}</span>}
      </div>

      <details
        className="mt-3"
        onToggle={(e) => (e.currentTarget as HTMLDetailsElement).open && setFalsifiersRead(true)}
      >
        <summary className="cursor-pointer select-none text-xs text-amber/90 hover:text-amber">
          falsifiers — what would prove this wrong ({decision.falsifiers.length})
        </summary>
        {decision.falsifiers.length > 0 ? (
          <ul className="mt-1.5 space-y-1 pl-4 text-xs text-sounding">
            {decision.falsifiers.map((f) => (
              <li key={f} className="list-disc">{f}</li>
            ))}
          </ul>
        ) : (
          <p className="mt-1.5 pl-4 text-xs text-rose">
            No falsifiers recorded — that is itself a warning.
          </p>
        )}
      </details>

      <details
        className="mt-1.5"
        onToggle={(e) => (e.currentTarget as HTMLDetailsElement).open && setPreMortemRead(true)}
      >
        <summary className="cursor-pointer select-none text-xs text-amber/90 hover:text-amber">
          pre-mortem — how this loss would read in a year
        </summary>
        <p className="mt-1.5 pl-4 text-xs text-sounding">
          {decision.pre_mortem ?? "No pre-mortem recorded yet; it is required at journal time."}
        </p>
      </details>

      <div className="mt-3 flex items-center gap-3">
        <CopyChip
          command={decision.journal_command}
          disabled={!armed}
          disabledHint="Read the falsifiers and the pre-mortem first."
        />
        {!armed && (
          <span className="text-[10px] text-sounding/85">
            enabled after reading falsifiers + pre-mortem
          </span>
        )}
      </div>
    </div>
  );
}

export function Today() {
  const today = useQuery({ queryKey: ["today"], queryFn: fetchToday, retry: 1 });

  if (today.error instanceof OfflineError) {
    return <OfflinePanel error={today.error} />;
  }

  const data = today.data;
  if (!data) {
    return (
      <div className="max-w-6xl">
        <p className="eyebrow">daily review</p>
        <h1 className="font-display mt-1 text-4xl text-moonlight">Today</h1>
        <p className="voice mt-6">
          {today.isError ? `Failed to load: ${String(today.error)}` : "loading…"}
        </p>
      </div>
    );
  }

  const ready = data.decisions.filter((d) => d.is_current_actionable);
  const waiting = data.decisions.filter((d) => !d.is_current_actionable);

  return (
    <div className="max-w-6xl">
      <div className="flex items-baseline justify-between">
        <div>
          <p className="eyebrow">daily review</p>
          <h1 className="font-display mt-1 text-4xl text-moonlight">Today</h1>
        </div>
        <span className="font-data text-xs text-sounding">
          as of {data.generated_at.slice(0, 16).replace("T", " ")} UTC
        </span>
      </div>

      <div className="mt-7 flex flex-col gap-5">
        <HealthStrip health={data.health} />

        {/* The waterline: the water around the buy line. */}
        <div
          className="condense"
          style={{ ["--condense-i" as string]: 1 } as CSSProperties}
        >
          <div className="flex items-baseline justify-between px-6 pt-5">
            <h2 className="font-display text-2xl text-moonlight">The waterline</h2>
            <p className="voice text-sm">
              each name plotted by distance to its buy target — below the line, price is inside
              the buy zone
            </p>
          </div>
          <div className="mt-4">
            <WaterlineStrip items={data.waterline} deeper={data.waterline_deeper} />
          </div>
        </div>

        <div className="grid grid-cols-1 gap-5 xl:grid-cols-[3fr_2fr]">
          {/* Decisions due */}
          <Section condenseIndex={2} className="p-5">
            <h2 className="font-display text-xl text-moonlight">Decisions due</h2>
            <p className="mt-1 text-xs text-sounding">
              open dispositions — each requires an explicit journaled ACTED / PASSED / DEFERRED
            </p>
            {data.decisions.length === 0 && (
              <p className="voice mt-5">
                Nothing awaits a decision. The discipline is to do nothing.
              </p>
            )}
            {ready.length > 0 && (
              <div className="mt-4 flex flex-col gap-3">
                {ready.map((d) => (
                  <DecisionCardView key={d.id} decision={d} />
                ))}
              </div>
            )}
            {waiting.length > 0 && (
              <details className="mt-4" open={ready.length === 0 && waiting.length <= 4}>
                <summary className="cursor-pointer select-none text-xs text-sounding hover:text-moonlight">
                  {waiting.length} open decision{waiting.length === 1 ? "" : "s"} not currently
                  actionable (superseded source, event flag, or gate) — expand to review
                </summary>
                <div className="mt-3 flex flex-col gap-3">
                  {waiting.map((d) => (
                    <DecisionCardView key={d.id} decision={d} />
                  ))}
                </div>
              </details>
            )}
          </Section>

          <div className="flex flex-col gap-5">
            {/* Event holds */}
            <Section condenseIndex={3} className="p-5">
              <h2 className="font-display text-xl text-moonlight">Event holds</h2>
              <p className="mt-1 text-xs text-sounding">
                deploy-ready names on hold pending corporate-event review
              </p>
              {data.gate_blocked.length === 0 ? (
                <p className="voice mt-4">No open event holds.</p>
              ) : (
                <ul className="mt-3 max-h-72 space-y-1.5 overflow-y-auto pr-1">
                  {data.gate_blocked.map((item) => (
                    <li key={item.watchlist_id} className="flex items-baseline gap-2.5">
                      <Link
                        to="/company/$ticker"
                        params={{ ticker: item.ticker }}
                        className="font-data w-14 shrink-0 text-sm text-moonlight hover:text-jade"
                      >
                        {item.ticker}
                      </Link>
                      <span
                        className="truncate text-xs text-amber/85"
                        title={item.event_pending ?? undefined}
                      >
                        <EventPendingTerms raw={item.event_pending} />
                      </span>
                      <span className="font-data ml-auto shrink-0 text-xs text-sounding">
                        {formatPct(item.distance_from_buy_pct)}
                      </span>
                    </li>
                  ))}
                </ul>
              )}
              {data.gate_blocked.length > 0 && (
                <div className="mt-3 flex items-center justify-between gap-3">
                  <CopyChip command="ivi events list --open" label="events open for review" />
                  <Link
                    to="/events"
                    search={{ lane: "queue_protection" }}
                    className="shrink-0 text-xs text-sounding hover:text-jade"
                  >
                    review in Events →
                  </Link>
                </div>
              )}
            </Section>

            {/* Digest */}
            <Section condenseIndex={4} className="p-5">
              <div className="flex items-baseline justify-between">
                <h2 className="font-display text-xl text-moonlight">Daily digest</h2>
                {data.digest && (
                  <span className="font-data text-xs text-sounding">
                    {data.digest.date}
                    {data.digest.previous_date ? ` · prev ${data.digest.previous_date}` : ""}
                  </span>
                )}
              </div>
              {data.digest ? (
                <details className="mt-2">
                  <summary className="cursor-pointer select-none text-xs text-jade hover:underline">
                    read the digest (status transitions inside)
                  </summary>
                  <div
                    className="reader mt-3 max-h-[36rem] overflow-y-auto pr-2"
                    // Server-rendered by markdown-it with raw HTML escaped.
                    dangerouslySetInnerHTML={{ __html: data.digest.html }}
                  />
                </details>
              ) : (
                <p className="voice mt-4">No digest has been generated yet.</p>
              )}
            </Section>
          </div>
        </div>
      </div>
    </div>
  );
}
