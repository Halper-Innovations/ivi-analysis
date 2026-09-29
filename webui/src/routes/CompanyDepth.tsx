/**
 * Company depth tabs — Research, Dossier, Decisions.
 *
 * Research is the deep-research pass unpacked: the thesis ledger (original →
 * adjusted per method), calibrated adjustments, analyst notes, filing
 * citations, evidence excerpts. Dossier rides the Reader's renderer with the
 * claims panel attached. Decisions put the falsifier checklist above the
 * disposition history — re-read why you'd be wrong before acting.
 */

import { Link } from "@tanstack/react-router";

import { CopyChip, StatusChip } from "../components/chips";
import type {
  AnalystNote,
  CompanyDecisionsResponse,
  CompanyDossierResponse,
  CompanyResearchResponse,
  DispositionRow,
  ResearchAdjustment,
} from "../lib/api";
import { formatPct, humanizeToken } from "../lib/floor";
import { formatMoney } from "../lib/gauge";
import { ClaimsPanel } from "./Reader";

function fmtValue(value: number | null): string {
  return value === null ? "—" : formatMoney(value);
}

/* ------------------------------------------------------------------ */
/* Research                                                            */
/* ------------------------------------------------------------------ */

/** Original → adjusted rows for the thesis ledger; Graham carries no
 *  adjusted figure (the resolver treats it as an unadjusted source). */
function thesisRows(thesis: NonNullable<CompanyResearchResponse["thesis"]>) {
  return [
    { method: "dcf", original: thesis.original_dcf, adjusted: thesis.adjusted_dcf },
    { method: "epv", original: thesis.original_epv, adjusted: thesis.adjusted_epv },
    { method: "graham", original: thesis.original_graham, adjusted: null },
  ];
}

function AdjustmentCard({ adjustment }: { adjustment: ResearchAdjustment }) {
  return (
    <div className="glass rounded-xl p-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <span className="font-data text-xs text-sounding">
          {humanizeToken(adjustment.affected_method)} ·{" "}
          {humanizeToken(adjustment.hypothesis_source)}
        </span>
        <span className="flex gap-1.5">
          <StatusChip value={adjustment.hypothesis_status} />
          <StatusChip value={adjustment.adjustment_confidence} />
        </span>
      </div>
      {adjustment.hypothesis_claim && (
        <p className="mt-2 text-[13px] leading-relaxed text-moonlight/95">
          {adjustment.hypothesis_claim}
        </p>
      )}
      <p className="font-data mt-2 text-sm">
        <span
          className={
            (adjustment.adjustment_magnitude ?? 0) < 0 ? "text-rose" : "text-jade"
          }
        >
          {adjustment.adjustment_magnitude === null
            ? "—"
            : `${adjustment.adjustment_magnitude > 0 ? "+" : "−"}$${Math.abs(
                adjustment.adjustment_magnitude,
              ).toFixed(2)}`}
        </span>
        <span className="text-sounding"> per share to {adjustment.affected_method}</span>
      </p>
      {adjustment.calibration_detail && (
        <p className="font-data mt-1.5 text-[10px] leading-relaxed text-sounding">
          {adjustment.calibration_detail}
        </p>
      )}
    </div>
  );
}

const NOTE_SECTIONS: {
  key: "positives" | "risks" | "surprises" | "adjustment_triggers";
  label: string;
  tone: string;
}[] = [
  { key: "positives", label: "positives", tone: "text-jade/90" },
  { key: "risks", label: "risks", tone: "text-rose/90" },
  { key: "surprises", label: "surprises", tone: "text-amber/90" },
  { key: "adjustment_triggers", label: "adjustment triggers", tone: "text-sounding" },
];

function NoteItem({ note }: { note: AnalystNote }) {
  return (
    <li className="border-t border-glass-tint/8 py-2 first:border-t-0">
      <p className="text-xs leading-relaxed text-moonlight/90">{note.claim}</p>
      <p className="font-data mt-1 text-[10px] text-sounding">
        {[note.severity, note.validation_status]
          .filter(Boolean)
          .map((token) => humanizeToken(token as string))
          .join(" · ")}
      </p>
      {note.suggested_adjustment && (
        <p className="mt-1 text-[11px] italic leading-relaxed text-sounding">
          {note.suggested_adjustment}
        </p>
      )}
      {note.citations.length > 0 && (
        <details className="mt-1">
          <summary className="cursor-pointer text-[10px] text-sounding/85">
            filing excerpt{note.citations.length === 1 ? "" : "s"} ({note.citations.length})
          </summary>
          {note.citations.map((citation, index) => (
            <p key={index} className="font-data mt-1 pl-3 text-[10px] leading-relaxed text-sounding">
              <span className="text-glass-tint/80">{citation.section}</span> {citation.excerpt}
            </p>
          ))}
        </details>
      )}
    </li>
  );
}

export function ResearchTab({ data }: { data: CompanyResearchResponse }) {
  if (!data.available) {
    return (
      <p className="voice text-sm">
        never deep-researched — the recursive pass has not visited this name.
      </p>
    );
  }
  const thesis = data.thesis;
  const mosPct =
    thesis?.adjusted_margin_of_safety != null
      ? thesis.adjusted_margin_of_safety * 100
      : null;
  return (
    <div className="space-y-5">
      <div className="font-data flex flex-wrap items-center gap-x-4 gap-y-1.5 text-[11px] text-sounding">
        <span>as of {data.as_of_date}</span>
        <StatusChip value={data.gate_action} />
        {data.conviction_class && <span>{humanizeToken(data.conviction_class)} conviction</span>}
        {data.tension_type && (
          <span title={data.tension_type}>tension: {humanizeToken(data.tension_type)}</span>
        )}
        {data.methods_agree !== null && (
          <span>{data.methods_agree ? "methods agree" : "methods disagree"}</span>
        )}
        {data.report_available && data.report_path && (
          <Link
            to="/reader"
            search={{ path: data.report_path, q: undefined }}
            className="text-jade underline-offset-4 hover:underline"
          >
            full report →
          </Link>
        )}
      </div>

      {thesis && (
        <div>
          <p className="eyebrow text-[10px]">the thesis ledger — original → adjusted</p>
          <table className="mt-2 w-full max-w-md border-collapse text-left">
            <thead>
              <tr className="text-[10px] uppercase tracking-wider text-sounding">
                <th className="py-1 pr-3 font-medium">method</th>
                <th className="py-1 pr-3 text-right font-medium">original</th>
                <th className="py-1 pr-0 text-right font-medium">adjusted</th>
              </tr>
            </thead>
            <tbody>
              {thesisRows(thesis).map((row) => (
                <tr key={row.method} className="border-t border-glass-tint/8 text-xs">
                  <td className="font-data py-1.5 pr-3 text-moonlight">{row.method}</td>
                  <td className="font-data py-1.5 pr-3 text-right text-sounding">
                    {fmtValue(row.original)}
                  </td>
                  <td className="font-data py-1.5 pr-0 text-right text-moonlight">
                    {row.method === "graham" ? (
                      <span className="text-sounding/80" title="unadjusted source">
                        (unadjusted)
                      </span>
                    ) : (
                      fmtValue(row.adjusted)
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="font-data mt-3 text-xs text-sounding">
            adjusted intrinsic mid{" "}
            <span className="text-moonlight">{fmtValue(thesis.adjusted_intrinsic_mid)}</span>
            {" · "}price {fmtValue(thesis.current_price)}
            {" · "}MoS{" "}
            <span className={mosPct !== null && mosPct >= 0 ? "text-jade" : "text-rose"}>
              {formatPct(mosPct)}
            </span>
            {thesis.adjusted_value_floored && (
              <span className="text-amber"> · value floored</span>
            )}
          </p>
          <p className="font-data mt-1 text-[10px] text-sounding">
            hypotheses: {data.hypotheses_generated ?? 0} generated ·{" "}
            {thesis.hypotheses_confirmed ?? 0} confirmed ·{" "}
            {thesis.hypotheses_contradicted ?? 0} contradicted ·{" "}
            {thesis.hypotheses_inconclusive ?? 0} inconclusive
            {thesis.average_coverage !== null &&
              ` · evidence coverage ${(thesis.average_coverage * 100).toFixed(0)}%`}
            {(thesis.high_priority_unresolved ?? 0) > 0 && (
              <span className="text-amber">
                {" "}
                · {thesis.high_priority_unresolved} high-priority unresolved
              </span>
            )}
          </p>
        </div>
      )}

      {data.adjustments.length > 0 && (
        <div>
          <p className="eyebrow text-[10px]">calibrated adjustments</p>
          <div className="mt-2 grid grid-cols-1 gap-3 xl:grid-cols-2">
            {data.adjustments.map((adjustment, index) => (
              <AdjustmentCard key={index} adjustment={adjustment} />
            ))}
          </div>
        </div>
      )}

      {data.unresolved.length > 0 && (
        <div>
          <p className="eyebrow text-[10px]">unresolved research needs</p>
          <ul className="mt-1.5 space-y-1 text-xs text-sounding">
            {data.unresolved.map((item, index) => (
              <li key={index} className="font-data">
                {item.description}{" "}
                <span className="text-sounding/80">
                  · {humanizeToken(item.importance)} · {humanizeToken(item.unresolved_reason)}
                  {item.hypothesis_priority === "HIGH" && (
                    <span className="text-amber"> · high priority</span>
                  )}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {data.analyst_notes && (
        <div>
          <p className="eyebrow text-[10px]">analyst notes — from the filing itself</p>
          {data.analyst_notes.overall_assessment && (
            <p className="mt-2 max-w-[68ch] text-[13px] leading-relaxed text-moonlight/95">
              {data.analyst_notes.overall_assessment}
            </p>
          )}
          <div className="mt-3 grid grid-cols-1 gap-4 lg:grid-cols-2">
            {NOTE_SECTIONS.map(({ key, label, tone }) => {
              const notes = data.analyst_notes![key];
              if (notes.length === 0) return null;
              return (
                <div key={key} className="glass rounded-xl p-4">
                  <p className={`eyebrow text-[10px] ${tone}`}>{label}</p>
                  <ul className="mt-1">
                    {notes.map((note, index) => (
                      <NoteItem key={index} note={note} />
                    ))}
                  </ul>
                </div>
              );
            })}
          </div>
          {data.analyst_notes.filing_sections_read.length > 0 && (
            <p className="font-data mt-2 text-[10px] text-sounding/85">
              sections read: {data.analyst_notes.filing_sections_read.join(", ")}
            </p>
          )}
        </div>
      )}

      {data.citations.length > 0 && (
        <details>
          <summary className="eyebrow cursor-pointer text-[10px]">
            filing citations ({data.citations.length})
          </summary>
          <table className="mt-2 w-full border-collapse text-left">
            <thead>
              <tr className="text-[10px] uppercase tracking-wider text-sounding">
                <th className="py-1 pr-3 font-medium">section</th>
                <th className="py-1 pr-3 font-medium">excerpt</th>
                <th className="py-1 pr-0 font-medium">source</th>
              </tr>
            </thead>
            <tbody>
              {data.citations.map((citation, index) => (
                <tr key={citation.citation_id ?? index} className="border-t border-glass-tint/8 align-top text-xs">
                  <td className="font-data py-1.5 pr-3 text-sounding">
                    {humanizeToken(citation.section)}
                  </td>
                  <td className="py-1.5 pr-3 leading-relaxed text-moonlight/85">
                    {citation.excerpt}
                  </td>
                  <td className="font-data py-1.5 pr-0 whitespace-nowrap text-[10px] text-sounding">
                    {citation.source_form_type ?? "—"}
                    {citation.source_filing_date ? ` · ${citation.source_filing_date}` : ""}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </details>
      )}

      {data.evidence.length > 0 && (
        <details>
          <summary className="eyebrow cursor-pointer text-[10px]">
            evidence items ({data.evidence.length})
          </summary>
          <ul className="mt-2 space-y-2">
            {data.evidence.map((item) => (
              <li key={item.evidence_id} className="glass rounded-xl p-3 text-xs">
                <p className="font-data text-[10px] text-sounding">
                  {humanizeToken(item.source_type)} · {item.as_of_date}
                  {item.source_url && (
                    <>
                      {" · "}
                      <a
                        href={item.source_url}
                        target="_blank"
                        rel="noreferrer"
                        className="text-jade underline-offset-2 hover:underline"
                      >
                        {item.source_title || "source"}
                      </a>
                    </>
                  )}
                </p>
                {item.excerpt && (
                  <p className="mt-1 leading-relaxed text-moonlight/85">{item.excerpt}</p>
                )}
              </li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Dossier                                                             */
/* ------------------------------------------------------------------ */

export function DossierTab({ data }: { data: CompanyDossierResponse }) {
  if (!data.available || !data.html) {
    return (
      <p className="voice text-sm">
        no 10-K dossier on file — the dossier pipeline has not visited this name.
      </p>
    );
  }
  return (
    <div>
      <div className="font-data flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-sounding">
        <span title={data.run_label ?? undefined}>run {data.run_label}</span>
        {data.others > 0 && <span>{data.others} earlier on disk</span>}
        {data.path && (
          <Link
            to="/reader"
            search={{ path: data.path, q: undefined }}
            className="text-jade underline-offset-4 hover:underline"
          >
            open in Reader (print-ready) →
          </Link>
        )}
      </div>
      <div
        className="reader mt-4"
        // Server-rendered markdown; artifact HTML is never trusted raw.
        dangerouslySetInnerHTML={{ __html: data.html }}
      />
      {data.claims && data.claims.length > 0 && <ClaimsPanel claims={data.claims} />}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Decisions                                                           */
/* ------------------------------------------------------------------ */

function DispositionCard({ row }: { row: DispositionRow }) {
  const trigger = row.trigger ?? {};
  const triggerEntries = Object.entries(trigger).filter(
    ([, value]) => value !== null && typeof value !== "object",
  );
  return (
    <div className="glass rounded-xl p-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <span className="text-sm text-moonlight">{humanizeToken(row.kind)}</span>
        <StatusChip value={row.status} />
      </div>
      <p className="font-data mt-1 text-[10px] text-sounding">
        opened {row.opened_at?.slice(0, 10)} by {row.opened_by ?? "—"}
        {row.decided_at && ` · decided ${row.decided_at.slice(0, 10)} by ${row.operator ?? "—"}`}
        {row.reason_code && ` · ${humanizeToken(row.reason_code)}`}
      </p>
      {row.rationale && (
        <p className="mt-2 max-w-[68ch] text-xs leading-relaxed text-moonlight/90">
          {row.rationale}
        </p>
      )}
      {(row.intended_size || row.sizing_rationale) && (
        <p className="mt-1.5 text-[11px] leading-relaxed text-sounding">
          {row.intended_size && <span className="font-data">{row.intended_size}</span>}
          {row.intended_size && row.sizing_rationale && " — "}
          {row.sizing_rationale}
        </p>
      )}
      {row.pre_mortem && (
        <details className="mt-2">
          <summary className="cursor-pointer text-[11px] text-amber/90">pre-mortem</summary>
          <p className="mt-1 max-w-[68ch] text-xs leading-relaxed text-sounding">
            {row.pre_mortem}
          </p>
        </details>
      )}
      {triggerEntries.length > 0 && (
        <dl className="font-data mt-2 grid grid-cols-2 gap-x-3 gap-y-0.5 text-[10px] sm:grid-cols-3">
          {triggerEntries.slice(0, 9).map(([key, value]) => (
            <div key={key} className="contents">
              <dt className="text-sounding" title={key}>
                {humanizeToken(key)}
              </dt>
              <dd className="text-right text-moonlight sm:col-span-2 sm:text-left">
                {typeof value === "number" ? String(value) : String(value)}
              </dd>
            </div>
          ))}
        </dl>
      )}
      {row.journal_command && (
        <div className="mt-3">
          <CopyChip command={row.journal_command} label="ivi investor journal …" />
        </div>
      )}
    </div>
  );
}

export function DecisionsTab({
  data,
  falsifiers,
}: {
  data: CompanyDecisionsResponse;
  falsifiers: string[];
}) {
  return (
    <div className="space-y-5">
      {falsifiers.length > 0 && (
        <div className="rounded-xl border border-amber/25 bg-amber/5 p-4">
          <p className="eyebrow text-[10px] text-amber/90">
            falsifiers — re-read before acting
          </p>
          <ul className="mt-1.5 space-y-1 pl-4 text-xs leading-relaxed text-moonlight/90">
            {falsifiers.map((falsifier) => (
              <li key={falsifier} className="list-disc">
                {falsifier}
              </li>
            ))}
          </ul>
        </div>
      )}

      <div>
        <p className="eyebrow text-[10px]">disposition history</p>
        {data.dispositions.length === 0 ? (
          <p className="voice mt-2 text-sm">
            no dispositions — price has never reached the zone for this name.
          </p>
        ) : (
          <div className="mt-2 space-y-3">
            {data.dispositions.map((row) => (
              <DispositionCard key={row.id} row={row} />
            ))}
          </div>
        )}
      </div>

      <div>
        <p className="eyebrow text-[10px]">
          outcome record — {data.outcomes_total} scans ({data.outcomes_open} open ·{" "}
          {data.outcomes_closed} closed)
        </p>
        {data.outcomes.length === 0 ? (
          <p className="voice mt-2 text-sm">no outcome rows for this name.</p>
        ) : (
          <table className="mt-2 w-full border-collapse text-left">
            <thead>
              <tr className="text-[10px] uppercase tracking-wider text-sounding">
                <th className="py-1 pr-3 font-medium">as of</th>
                <th className="py-1 pr-3 font-medium">decision</th>
                <th className="py-1 pr-3 text-right font-medium">entry</th>
                <th className="py-1 pr-3 text-right font-medium">realized</th>
                <th className="py-1 pr-3 text-right font-medium">benchmark</th>
                <th className="py-1 pr-3 text-right font-medium">excess</th>
                <th className="py-1 pr-0 font-medium">status</th>
              </tr>
            </thead>
            <tbody>
              {data.outcomes.map((row, index) => (
                <tr key={index} className="border-t border-glass-tint/8 text-xs">
                  <td className="font-data py-1.5 pr-3 text-sounding">{row.as_of_date}</td>
                  <td className="font-data py-1.5 pr-3 text-moonlight">
                    {row.decision}
                    {row.conviction !== null && (
                      <span className="text-sounding/85"> c{row.conviction}</span>
                    )}
                  </td>
                  <td className="font-data py-1.5 pr-3 text-right text-sounding">
                    {row.entry_price === null ? "—" : formatMoney(row.entry_price)}
                  </td>
                  <td className="font-data py-1.5 pr-3 text-right text-moonlight">
                    {formatPct(row.realized_return_pct)}
                  </td>
                  <td className="font-data py-1.5 pr-3 text-right text-sounding">
                    {formatPct(row.benchmark_return_pct)}
                  </td>
                  <td
                    className={`font-data py-1.5 pr-3 text-right ${
                      row.excess_return_pct === null
                        ? "text-sounding"
                        : row.excess_return_pct >= 0
                          ? "text-jade"
                          : "text-rose"
                    }`}
                  >
                    {formatPct(row.excess_return_pct)}
                  </td>
                  <td className="font-data py-1.5 pr-0 text-[10px] text-sounding">
                    {row.outcome_status}
                    {row.close_date ? ` ${row.close_date}` : ""}
                    {row.reached_buy_target === true && (
                      <span className="text-jade" title="reached buy target">
                        {" "}
                        ◉
                      </span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
