/**
 * Reader — one beautiful renderer for every markdown artifact: digests,
 * run reports, dossiers, peer reports, memos. Server-rendered commonmark
 * (raw HTML never trusted), a sticky TOC, dossier claims as a provenance
 * panel, and a print stylesheet clean enough to hand to an outside
 * investor.
 */

import { useQuery } from "@tanstack/react-query";
import { useNavigate, useSearch } from "@tanstack/react-router";
import { useMemo, useState } from "react";

import { Section } from "../components/GlassPanel";
import type { ArtifactClaim, ReaderFamily } from "../lib/api";
import { fetchReaderArtifact, fetchReaderLibrary } from "../lib/api";
import { formatBytes } from "../lib/ops";

export interface ReaderSearch {
  path?: string;
  q?: string;
}

function Library({ q, onOpen }: { q: string; onOpen: (path: string) => void }) {
  const query = useQuery({
    queryKey: ["reader-library"],
    queryFn: fetchReaderLibrary,
    staleTime: 60_000,
    retry: 1,
  });

  const families: ReaderFamily[] = useMemo(() => {
    if (!query.data) return [];
    const needle = q.trim().toLowerCase();
    if (!needle) return query.data.families;
    return query.data.families
      .map((family) => ({
        ...family,
        items: family.items.filter((item) =>
          item.title.toLowerCase().includes(needle),
        ),
      }))
      .filter((family) => family.items.length > 0);
  }, [query.data, q]);

  if (query.isLoading) {
    return (
      <Section className="mt-6 p-8">
        <p className="voice">loading…</p>
      </Section>
    );
  }
  if (!query.data) {
    return (
      <Section className="mt-6 p-8">
        <p className="text-xs text-rose">The library could not be read.</p>
      </Section>
    );
  }
  return (
    <div className="mt-6 flex flex-col gap-5">
      {families.map((family, index) => (
        <Section key={family.family} condenseIndex={index} className="p-5">
          <div className="flex items-baseline justify-between gap-3">
            <h2 className="font-display text-xl text-moonlight">{family.label}</h2>
            <span className="font-data text-xs text-sounding">{family.total}</span>
          </div>
          <ul className="mt-3 grid grid-cols-1 gap-x-6 gap-y-1 md:grid-cols-2 xl:grid-cols-3">
            {family.items.slice(0, 60).map((item) => (
              <li key={item.path}>
                <button
                  type="button"
                  onClick={() => onOpen(item.path)}
                  className="w-full truncate text-left text-xs text-sounding hover:text-jade"
                  title={item.path}
                >
                  {item.title}
                  <span className="font-data ml-2 text-[10px] text-sounding/75">
                    {item.mtime.slice(0, 10)}
                  </span>
                </button>
              </li>
            ))}
          </ul>
          {family.items.length > 60 && (
            <p className="mt-2 text-[10px] text-sounding/80">
              showing 60 of {family.items.length} — narrow with the search box
            </p>
          )}
        </Section>
      ))}
      {families.length === 0 && (
        <Section className="p-8">
          <p className="voice">no documents match.</p>
        </Section>
      )}
    </div>
  );
}

export function ClaimsPanel({ claims }: { claims: ArtifactClaim[] }) {
  const [needle, setNeedle] = useState("");
  const filtered = needle.trim()
    ? claims.filter((claim) =>
        claim.label.toLowerCase().includes(needle.trim().toLowerCase()),
      )
    : claims;
  return (
    <details className="no-print mt-4 rounded-xl border border-glass-tint/10 bg-deepwater/40 p-4">
      <summary className="cursor-pointer text-sm text-moonlight">
        Claims &amp; citations{" "}
        <span className="font-data text-xs text-sounding">
          {claims.length} — every number defends itself
        </span>
      </summary>
      <input
        value={needle}
        onChange={(event) => setNeedle(event.target.value)}
        placeholder="filter claims…"
        className="glass mt-3 w-64 rounded-lg border-none px-3 py-1.5 text-xs text-moonlight placeholder:text-sounding/50"
        aria-label="Filter claims"
      />
      <div className="mt-3 max-h-96 overflow-y-auto pr-1">
        <table className="w-full text-left text-[11px]">
          <thead>
            <tr className="text-[10px] uppercase tracking-wider text-sounding">
              <th className="pb-1 pr-4 font-medium">claim</th>
              <th className="pb-1 pr-4 text-right font-medium">value</th>
              <th className="pb-1 font-medium">source</th>
            </tr>
          </thead>
          <tbody>
            {filtered.slice(0, 200).map((claim) => (
              <tr key={claim.claim_id || claim.label} className="border-t border-glass-tint/8 align-top">
                <td className="font-data py-1 pr-4 text-sounding">{claim.label}</td>
                <td className="font-data py-1 pr-4 text-right text-moonlight">
                  {claim.value === null ? "—" : String(claim.value)}
                  {claim.unit ? <span className="text-sounding/85"> {claim.unit}</span> : null}
                </td>
                <td className="py-1">
                  {claim.citations.length === 0 ? (
                    <span className="text-sounding/75">—</span>
                  ) : (
                    <a
                      href={claim.citations[0].source_url}
                      target="_blank"
                      rel="noreferrer"
                      className="text-sounding underline decoration-glass-tint/30 hover:text-jade"
                      title={claim.citations[0].snippet || claim.citations[0].source_url}
                    >
                      {claim.citations[0].section_label || "source"}
                    </a>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </details>
  );
}

function ArtifactView({ path, onBack }: { path: string; onBack: () => void }) {
  const query = useQuery({
    queryKey: ["reader-artifact", path],
    queryFn: () => fetchReaderArtifact(path),
    staleTime: 60_000,
    retry: 1,
  });

  if (query.isLoading) {
    return (
      <Section className="mt-6 p-8">
        <p className="voice">loading…</p>
      </Section>
    );
  }
  if (!query.data) {
    return (
      <Section className="mt-6 p-8">
        <p className="text-xs text-rose">
          That artifact could not be read — it may have moved, or it is outside the
          document collection.
        </p>
        <button
          type="button"
          onClick={onBack}
          className="mt-3 text-xs text-sounding hover:text-moonlight"
        >
          ← back to the library
        </button>
      </Section>
    );
  }

  const artifact = query.data;
  const toc = artifact.toc.filter((entry) => entry.level <= 3);
  return (
    <div className="mt-6 flex gap-6">
      {toc.length > 1 && (
        <nav className="no-print sticky top-8 hidden max-h-[calc(100vh-6rem)] w-56 shrink-0 self-start overflow-y-auto lg:block">
          <p className="eyebrow mb-2">contents</p>
          <ul className="flex flex-col gap-1 border-l border-glass-tint/15 pl-3">
            {toc.map((entry) => (
              <li key={entry.anchor} style={{ paddingLeft: `${(entry.level - 1) * 10}px` }}>
                <button
                  type="button"
                  onClick={() =>
                    document
                      .getElementById(entry.anchor)
                      ?.scrollIntoView({ behavior: "smooth", block: "start" })
                  }
                  className="text-left text-[11px] leading-snug text-sounding hover:text-moonlight"
                >
                  {entry.text}
                </button>
              </li>
            ))}
          </ul>
        </nav>
      )}

      <div className="min-w-0 flex-1">
        {!artifact.decision_eligible && (
          <div className="no-print mb-3 rounded-xl border border-amber/30 bg-amber/5 px-4 py-3">
            <p className="text-xs text-amber">Excluded audit history</p>
            <p className="font-data mt-1 text-[10px] text-sounding">
              financial integrity: {artifact.integrity_status} · decision content suppressed
            </p>
          </div>
        )}
        <div className="no-print flex flex-wrap items-center justify-between gap-3">
          <button
            type="button"
            onClick={onBack}
            className="text-xs text-sounding hover:text-moonlight"
          >
            ← library
          </button>
          <div className="flex items-center gap-3">
            <span className="font-data text-[10px] text-sounding/85">
              {formatBytes(artifact.size)} · {artifact.mtime.slice(0, 10)}
            </span>
            <button
              type="button"
              onClick={() => window.print()}
              className="glass rounded-lg px-3 py-1.5 text-xs text-moonlight hover:text-jade"
            >
              print
            </button>
          </div>
        </div>

        <Section className="print-sheet mt-3 p-8">
          <article
            className="reader"
            // Server-rendered commonmark with html:false — artifact HTML
            // never reaches this string.
            dangerouslySetInnerHTML={{ __html: artifact.html }}
          />
        </Section>
        {artifact.claims && artifact.claims.length > 0 && (
          <ClaimsPanel claims={artifact.claims} />
        )}
        <p className="no-print font-data mt-2 break-all text-[10px] text-sounding/75">
          {artifact.path}
        </p>
      </div>
    </div>
  );
}

export function Reader() {
  const search = useSearch({ from: "/reader" }) as ReaderSearch;
  const navigate = useNavigate({ from: "/reader" });

  const open = (path: string | undefined) => {
    void navigate({
      search: (previous: ReaderSearch) => {
        const next: ReaderSearch = { ...previous, path };
        if (!next.path) delete next.path;
        return next;
      },
      replace: true,
    });
  };

  return (
    <div className="max-w-[80rem]">
      <div className="no-print flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="eyebrow">reports and memos</p>
          <h1 className="font-display mt-1 text-4xl text-moonlight">Reader</h1>
        </div>
        {!search.path && (
          <input
            value={search.q ?? ""}
            onChange={(event) =>
              void navigate({
                search: (previous: ReaderSearch) => {
                  const next = { ...previous, q: event.target.value || undefined };
                  if (!next.q) delete next.q;
                  return next;
                },
                replace: true,
              })
            }
            placeholder="search titles…"
            className="glass w-64 rounded-lg border-none px-3 py-1.5 text-xs text-moonlight placeholder:text-sounding/50"
            aria-label="Search the library"
          />
        )}
      </div>

      {search.path ? (
        <ArtifactView path={search.path} onBack={() => open(undefined)} />
      ) : (
        <Library q={search.q ?? ""} onOpen={open} />
      )}
    </div>
  );
}
