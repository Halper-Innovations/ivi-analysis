/**
 * Events — corporate-event review. Two lanes from the corporate-event ledger:
 * opportunity (spinoffs, ch11 emergences, busted IPOs) and queue
 * protection (the filings that hold DEPLOY_READY names at the gate),
 * columns by lifecycle. Disposal stays in the CLI — every open card
 * carries its exact `ivi events dispose` command. The scan-integrity
 * strip renders the platform's own rule: an unscanned business day is a
 * hole in the record.
 */

import { useQuery } from "@tanstack/react-query";
import { Link, useNavigate, useSearch } from "@tanstack/react-router";
import { Fragment } from "react";

import { CopyChip, Term } from "../components/chips";
import { Section } from "../components/GlassPanel";
import type { EventCard, EventsLane, ScanStrip } from "../lib/api";
import { OfflineError, fetchEvents } from "../lib/api";
import { humanizeToken } from "../lib/floor";
import { detailHighlights, formatAgeDays, scanDayGlyph } from "../lib/ops";

export interface EventsSearch {
  lane?: string;
  type?: string;
  q?: string;
}

const LANE_TITLES: Record<string, { title: string; hint: string }> = {
  opportunity: {
    title: "Opportunity",
    hint: "special situations screened for entry",
  },
  queue_protection: {
    title: "Queue protection",
    hint: "filings holding watchlist names pending review",
  },
};

const TONE_TEXT = {
  jade: "text-jade",
  amber: "text-amber",
  rose: "text-rose",
  slate: "text-sounding/40",
} as const;

const TONE_BG = {
  jade: "bg-jade/70",
  amber: "bg-amber/70",
  rose: "bg-rose/80",
  slate: "bg-glass-tint/10",
} as const;

function ScanIntegrityStrip({ strip }: { strip: ScanStrip }) {
  const chronological = strip.days.slice().reverse();
  return (
    <div>
      <div className="flex items-center gap-2">
        <div className="flex items-end gap-[3px]" role="img" aria-label={`Scan ledger, last ${strip.window_business_days} business days: ${strip.gap_count} gaps`}>
          {chronological.map((day) => {
            const glyph = scanDayGlyph(day.status);
            return (
              <span
                key={day.date}
                className={`inline-block h-4 w-2 rounded-[3px] ${TONE_BG[glyph.tone]}`}
                title={
                  day.status === "OK"
                    ? `${day.date} — scanned: ${day.index_rows ?? "?"} index rows, ${day.events_created ?? 0} events created`
                    : `${day.date} — ${glyph.label}`
                }
              />
            );
          })}
        </div>
        <span
          className={`font-data text-xs ${strip.gap_count > 0 ? "text-rose" : "text-jade"}`}
        >
          {strip.gap_count === 0
            ? "no gaps"
            : `${strip.gap_count} unscanned ${strip.gap_count === 1 ? "day" : "days"}`}
        </span>
      </div>
      <p className="mt-1 text-[10px] text-sounding/85">
        scan ledger, last {strip.window_business_days} business days · an unscanned
        day is a hole in the record
      </p>
    </div>
  );
}

function EventCardView({ card }: { card: EventCard }) {
  const unknown = card.ticker_state === "UNKNOWN_TICKER" || !card.ticker;
  const highlights = detailHighlights(card.detail);
  return (
    <div className="rounded-xl border border-glass-tint/10 bg-deepwater/40 p-3">
      <div className="flex items-baseline justify-between gap-2">
        {unknown ? (
          <span className="text-xs italic text-sounding/80">unknown ticker</span>
        ) : card.lane === "queue_protection" ? (
          <Link
            to="/company/$ticker"
            params={{ ticker: card.ticker! }}
            className="font-data text-sm font-semibold text-moonlight hover:text-jade"
          >
            {card.ticker}
          </Link>
        ) : (
          <span className="font-data text-sm font-semibold text-moonlight">
            {card.ticker}
          </span>
        )}
        <span className="font-data shrink-0 text-[10px] text-sounding" title={`detected ${card.detection_date ?? "—"}`}>
          {formatAgeDays(card.age_days)}
        </span>
      </div>
      <p className="mt-0.5 truncate text-[11px] text-sounding" title={card.company_name ?? undefined}>
        {card.company_name ?? "—"}
      </p>
      <p className="mt-1 text-[11px] text-moonlight/90">
        <Term token={card.event_type} label={humanizeToken(card.event_type)} />
        {card.expiry_reason ? (
          <span className="text-sounding/85"> · {humanizeToken(card.expiry_reason)}</span>
        ) : null}
      </p>
      {card.filings.length > 0 && (
        <p className="font-data mt-1 truncate text-[10px] text-sounding/80">
          {card.filings
            .slice(-2)
            .map((filing) => `${filing.form_type ?? "?"} ${filing.filing_date ?? ""}`)
            .join(" · ")}
        </p>
      )}
      {highlights.length > 0 && (
        <p
          className="mt-1 truncate text-[10px] text-sounding/85"
          title={highlights.join("\n")}
        >
          {highlights.join(" · ")}
        </p>
      )}
      {card.dispose_command && (
        <div className="mt-2">
          <CopyChip command={card.dispose_command} label="dispose…" />
        </div>
      )}
    </div>
  );
}

function LaneSection({ lane }: { lane: EventsLane }) {
  const meta = LANE_TITLES[lane.lane] ?? { title: lane.lane, hint: "" };
  const types = Object.entries(lane.type_counts).sort((a, b) => b[1] - a[1]);
  return (
    <Section className="p-5">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div>
          <h2 className="font-display text-xl text-moonlight">{meta.title}</h2>
          <p className="voice mt-0.5">{meta.hint}</p>
        </div>
        <p className="font-data text-xs text-sounding">
          {lane.open_total} open · {lane.total} total
        </p>
      </div>
      {types.length > 0 && (
        <p className="mt-2 text-[11px] text-sounding/80">
          {types.map(([type, count], index) => (
            <Fragment key={type}>
              {index > 0 ? " · " : null}
              <Term token={type} label={humanizeToken(type)} /> {count}
            </Fragment>
          ))}
        </p>
      )}
      <div className="mt-4 grid grid-cols-2 gap-3 lg:grid-cols-5">
        {lane.columns.map((column) => (
          <div key={column.status} className="min-w-0">
            <p className="font-data mb-2 text-[10px] tracking-[0.08em] text-sounding">
              {column.status}
              <span className="ml-1.5 text-sounding/80">{column.total}</span>
            </p>
            <div className="flex max-h-[30rem] flex-col gap-2 overflow-y-auto pr-1">
              {column.cards.map((card) => (
                <EventCardView key={card.id} card={card} />
              ))}
              {column.total > column.cards.length && (
                <p className="pb-1 text-center text-[10px] text-sounding/80">
                  showing {column.cards.length} of {column.total}
                </p>
              )}
              {column.total === 0 && (
                <p className="text-[10px] text-sounding/40">none</p>
              )}
            </div>
          </div>
        ))}
      </div>
    </Section>
  );
}

export function Events() {
  const search = useSearch({ from: "/events" }) as EventsSearch;
  const navigate = useNavigate({ from: "/events" });
  const query = useQuery({
    queryKey: ["events", search.lane ?? "", search.type ?? "", search.q ?? ""],
    queryFn: () => fetchEvents({ lane: search.lane, type: search.type, q: search.q }),
    retry: 1,
  });

  const setSearch = (patch: Partial<EventsSearch>) => {
    void navigate({
      search: (previous: EventsSearch) => {
        const next = { ...previous, ...patch };
        for (const key of Object.keys(next) as (keyof EventsSearch)[]) {
          if (!next[key]) delete next[key];
        }
        return next;
      },
      replace: true,
    });
  };

  if (query.error instanceof OfflineError) {
    return (
      <Section className="max-w-xl p-8">
        <h1 className="font-display text-2xl text-moonlight">IVI offline</h1>
        <p className="font-data mt-3 text-sm text-rose">{query.error.precondition}</p>
        <p className="font-data mt-1 break-all text-xs text-sounding">{query.error.detail}</p>
      </Section>
    );
  }

  const deck = query.data;
  const typeOptions = new Set<string>(search.type ? [search.type] : []);
  for (const lane of deck?.lanes ?? []) {
    for (const type of Object.keys(lane.type_counts)) typeOptions.add(type);
  }

  return (
    <div className="max-w-[88rem]">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="eyebrow">corporate-event review</p>
          <h1 className="font-display mt-1 text-4xl text-moonlight">Events</h1>
        </div>
        {deck && <ScanIntegrityStrip strip={deck.scan_strip} />}
      </div>

      <div className="mt-6 flex flex-wrap items-center gap-2">
        <select
          value={search.lane ?? ""}
          onChange={(event) => setSearch({ lane: event.target.value || undefined })}
          className="glass rounded-lg border-none px-3 py-1.5 text-xs text-moonlight"
          aria-label="Filter by lane"
        >
          <option value="">both lanes</option>
          <option value="opportunity">opportunity</option>
          <option value="queue_protection">queue protection</option>
        </select>
        <select
          value={search.type ?? ""}
          onChange={(event) => setSearch({ type: event.target.value || undefined })}
          className="glass rounded-lg border-none px-3 py-1.5 text-xs text-moonlight"
          aria-label="Filter by event type"
        >
          <option value="">every type</option>
          {[...typeOptions].sort().map((type) => (
            <option key={type} value={type}>
              {humanizeToken(type)}
            </option>
          ))}
        </select>
        <input
          value={search.q ?? ""}
          onChange={(event) => setSearch({ q: event.target.value || undefined })}
          placeholder="ticker or company…"
          className="glass w-56 rounded-lg border-none px-3 py-1.5 text-xs text-moonlight placeholder:text-sounding/50"
          aria-label="Search events"
        />
        {deck && deck.unknown_ticker_total > 0 && (
          <span className="ml-auto text-[11px] text-sounding">
            {deck.unknown_ticker_total} events still resolving tickers
          </span>
        )}
      </div>

      {query.isLoading || !deck ? (
        <Section className="mt-6 p-8">
          <p className="voice">loading…</p>
        </Section>
      ) : (
        <div className="mt-6 flex flex-col gap-6">
          {deck.lanes.map((lane) => (
            <LaneSection key={lane.lane} lane={lane} />
          ))}
        </div>
      )}

      <p className="mt-3 text-[10px] text-sounding/85">
        lifecycle runs detected → qualified → surfaced → decided; expired is
        terminal · disposal happens in the CLI via each card's command chip ·{" "}
        <span className={deck && deck.scan_strip.gap_count > 0 ? TONE_TEXT.rose : TONE_TEXT.jade}>
          strip
        </span>{" "}
        = daily scan ledger
      </p>
    </div>
  );
}
