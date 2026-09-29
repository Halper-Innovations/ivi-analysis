/**
 * ⌘K palette: search tickers and company names across the whole registrant
 * census (via /api/search), fuzzy-jump to pages, and copy `ivi` command
 * chips — the read-only bridge to the CLI. Hand-built (input + list +
 * keyboard loop) rather than a library; the glass overlay is our own.
 *
 * Census rows the platform can't render a company page for yet appear in a
 * dim, non-interactive tail — known, not covered.
 */

import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { useNavigate } from "@tanstack/react-router";
import { useEffect, useMemo, useRef, useState } from "react";

import { fetchSearch } from "../lib/api";
import { presentRegistrantName, searchResultHint } from "../lib/palette";

interface PaletteEntry {
  kind: "page" | "command" | "ticker";
  title: string;
  hint: string;
  to?: string;
  command?: string;
}

const ENTRIES: PaletteEntry[] = [
  { kind: "page", title: "Today", hint: "daily review", to: "/" },
  { kind: "page", title: "Watchlist", hint: "every tracked name", to: "/watchlist" },
  { kind: "page", title: "Scans", hint: "the run gallery", to: "/scans" },
  { kind: "page", title: "Coverage", hint: "sweep coverage by sector and cap band", to: "/coverage" },
  { kind: "page", title: "Events", hint: "corporate-event review", to: "/events" },
  { kind: "page", title: "Outcomes", hint: "realized results by valuation method", to: "/outcomes" },
  { kind: "page", title: "Compare", hint: "2–4 tickers on one waterline", to: "/compare" },
  { kind: "page", title: "Ops", hint: "operational status", to: "/ops" },
  { kind: "page", title: "Reader", hint: "reports and memos", to: "/reader" },
  { kind: "page", title: "Gauge Lab", hint: "the depth gauge at three sizes", to: "/gauge-lab" },
  {
    kind: "command",
    title: "ivi investor today",
    hint: "copy — the CLI morning brief",
    command: "ivi investor today",
  },
  {
    kind: "command",
    title: "ivi ops deadman",
    hint: "copy — fail-loud health assertions",
    command: "ivi ops deadman",
  },
  {
    kind: "command",
    title: "ivi watchlist check-triggers",
    hint: "copy — price-trigger sweep",
    command: "ivi watchlist check-triggers",
  },
];

function fuzzyMatch(query: string, text: string): boolean {
  const q = query.toLowerCase().replace(/\s/g, "");
  const t = text.toLowerCase();
  let i = 0;
  for (const ch of t) {
    if (ch === q[i]) i += 1;
    if (i === q.length) return true;
  }
  return q.length === 0;
}

export function CommandPalette({ onClose }: { onClose: () => void }) {
  const [query, setQuery] = useState("");
  const [debounced, setDebounced] = useState("");
  const [selected, setSelected] = useState(0);
  const [copied, setCopied] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const navigate = useNavigate();

  useEffect(() => {
    const handle = window.setTimeout(() => setDebounced(query.trim()), 160);
    return () => window.clearTimeout(handle);
  }, [query]);

  // Ticker + company-name search over the full census; previous results
  // hold while the next keystroke's query is in flight, so the list never
  // flashes empty mid-word. Offline/errors fall back to pages-only.
  const search = useQuery({
    queryKey: ["search", debounced],
    queryFn: () => fetchSearch(debounced),
    enabled: debounced.length > 0,
    staleTime: 60_000,
    placeholderData: keepPreviousData,
    retry: 0,
  });

  const results = useMemo(
    () => (debounced.length > 0 ? (search.data?.results ?? []) : []),
    [debounced, search.data],
  );

  const matches = useMemo(() => {
    const pages = ENTRIES.filter((e) => fuzzyMatch(query, `${e.title} ${e.hint}`));
    const tickers: PaletteEntry[] = results
      .filter((r) => r.covered)
      .slice(0, 8)
      .map((r) => ({
        kind: "ticker" as const,
        title: r.ticker,
        hint: searchResultHint(r),
        to: `/company/${r.ticker}`,
      }));
    return [...tickers, ...pages];
  }, [query, results]);

  const uncovered = useMemo(
    () => results.filter((r) => !r.covered).slice(0, 4),
    [results],
  );

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  useEffect(() => {
    setSelected(0);
  }, [query]);

  const run = async (entry: PaletteEntry) => {
    if ((entry.kind === "page" || entry.kind === "ticker") && entry.to) {
      await navigate({ to: entry.to });
      onClose();
      return;
    }
    if (entry.command) {
      await navigator.clipboard.writeText(entry.command);
      setCopied(entry.command);
      window.setTimeout(onClose, 650);
    }
  };

  const onKeyDown = (event: React.KeyboardEvent) => {
    if (event.key === "Escape") onClose();
    if (event.key === "ArrowDown") {
      event.preventDefault();
      setSelected((s) => Math.min(s + 1, matches.length - 1));
    }
    if (event.key === "ArrowUp") {
      event.preventDefault();
      setSelected((s) => Math.max(s - 1, 0));
    }
    if (event.key === "Enter" && matches[selected]) {
      void run(matches[selected]);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center bg-abyss/60 pt-[18vh]"
      onClick={onClose}
      role="presentation"
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label="Search"
        className="glass-overlay w-[min(560px,90vw)] overflow-hidden"
        onClick={(event) => event.stopPropagation()}
        onKeyDown={onKeyDown}
      >
        <input
          ref={inputRef}
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search tickers, companies, pages…"
          aria-label="Search tickers, companies, and pages"
          className="w-full border-b border-glass-tint/10 bg-transparent px-5 py-4 text-sm text-moonlight outline-none placeholder:text-sounding/80"
        />
        <ul className="max-h-72 overflow-y-auto py-2" role="listbox" aria-label="Results">
          {matches.length === 0 && (
            <li className="px-5 py-3 text-sm text-sounding">
              Nothing matches — try a ticker, a company name, or a page.
            </li>
          )}
          {matches.map((entry, i) => (
            <li key={`${entry.kind}:${entry.title}`} role="option" aria-selected={i === selected}>
              <button
                type="button"
                onMouseEnter={() => setSelected(i)}
                onClick={() => void run(entry)}
                className={`flex w-full items-baseline justify-between px-5 py-2.5 text-left text-sm ${
                  i === selected ? "bg-glass-tint/10 text-moonlight" : "text-sounding"
                }`}
              >
                <span className={entry.kind === "command" ? "font-data" : undefined}>
                  {copied === entry.command ? "copied ✓" : entry.title}
                </span>
                <span className="ml-4 shrink-0 truncate text-xs text-sounding/85">
                  {entry.hint}
                </span>
              </button>
            </li>
          ))}
          {uncovered.length > 0 && (
            <li role="presentation" className="mt-1 border-t border-glass-tint/10 pt-1">
              <p className="px-5 pt-1.5 text-[10px] uppercase tracking-wider text-sounding/70">
                in the census — not covered yet
              </p>
              {uncovered.map((r) => (
                <p
                  key={r.ticker}
                  className="flex items-baseline justify-between px-5 py-1.5 text-sm text-sounding/55"
                >
                  <span>{r.ticker}</span>
                  <span className="ml-4 shrink-0 truncate text-xs">
                    {presentRegistrantName(r.name) ?? "—"}
                  </span>
                </p>
              ))}
            </li>
          )}
        </ul>
      </div>
    </div>
  );
}
