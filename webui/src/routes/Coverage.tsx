/**
 * Coverage — the atlas. The sector × cap-band grid of the loaded-set
 * ledger: cell glow = recency of complete coverage, the badge = tickers
 * under coverage, dark water = never swept. Click a cell for the tickers,
 * the runs that covered them, and the exact CLI to fill the gap — this is
 * `ivi sweep-delta-report` made visual, and the zero-spend cost gate
 * before any paid sweep.
 */

import { useQuery } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import { useState } from "react";

import { CopyChip } from "../components/chips";
import { Section } from "../components/GlassPanel";
import type { CoverageCell } from "../lib/api";
import { OfflineError, fetchCoverage, fetchCoverageCell } from "../lib/api";
import { humanizeToken } from "../lib/floor";
import { cellFor, coverageStrength, fillCommand } from "../lib/scans";

/** The validated method-blue at a recency-scaled presence. */
function cellStyle(cell: CoverageCell | undefined): React.CSSProperties {
  if (!cell) return {};
  const strength = coverageStrength(cell.age_days);
  return { backgroundColor: `rgba(57, 135, 229, ${(0.42 * strength).toFixed(3)})` };
}

function CellDetail({
  sector,
  band,
  onClose,
}: {
  sector: string;
  band: string;
  onClose: () => void;
}) {
  const query = useQuery({
    queryKey: ["coverage-cell", sector, band],
    queryFn: () => fetchCoverageCell(sector, band),
    staleTime: 5 * 60_000,
    retry: 1,
  });
  return (
    <Section className="fixed right-6 top-6 z-40 flex max-h-[calc(100vh-3rem)] w-[26rem] flex-col overflow-hidden p-5">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="eyebrow">{humanizeToken(band)}</p>
          <h2 className="font-display mt-0.5 text-xl text-moonlight">
            {humanizeToken(sector)}
          </h2>
        </div>
        <button
          type="button"
          onClick={onClose}
          className="font-data text-sm text-sounding hover:text-moonlight"
          aria-label="Close cell detail"
        >
          ✕
        </button>
      </div>

      <div className="mt-3">
        <CopyChip command={fillCommand(sector, band)} label={fillCommand(sector, band)} />
      </div>

      {query.isLoading ? (
        <p className="voice mt-4">loading…</p>
      ) : !query.data ? (
        <p className="mt-4 text-xs text-rose">The cell could not be read.</p>
      ) : (
        <div className="mt-4 flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto pr-1">
          <div>
            <p className="mb-1.5 text-[11px] text-sounding">
              runs that covered this cell · {query.data.runs.length}
            </p>
            {query.data.runs.length === 0 ? (
              <p className="text-xs text-sounding">
                Nothing has swept here — dark water.
              </p>
            ) : (
              <ul className="flex flex-col gap-1">
                {query.data.runs.map((run) => (
                  <li key={run.run_id} className="flex items-baseline justify-between gap-2 text-xs">
                    {run.slug ? (
                      <Link
                        to="/scans/run/$"
                        params={{ _splat: run.slug }}
                        className="font-data truncate text-moonlight hover:text-jade"
                        title={run.run_id}
                      >
                        {run.run_id}
                      </Link>
                    ) : (
                      <span className="font-data truncate text-sounding" title={run.run_id}>
                        {run.run_id}
                      </span>
                    )}
                    <span className="font-data shrink-0 text-[10px] text-sounding">
                      {run.loaded_at?.slice(0, 10) ?? "—"} · {run.tickers}
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </div>
          <div>
            <p className="mb-1.5 text-[11px] text-sounding">
              tickers under coverage · {query.data.tickers.length}
            </p>
            <div className="flex flex-wrap gap-x-2.5 gap-y-1">
              {query.data.tickers.map((ticker) => (
                <Link
                  key={ticker}
                  to="/company/$ticker"
                  params={{ ticker }}
                  className="font-data text-[11px] text-sounding hover:text-jade"
                >
                  {ticker}
                </Link>
              ))}
            </div>
          </div>
        </div>
      )}
    </Section>
  );
}

export function Coverage() {
  const query = useQuery({ queryKey: ["coverage"], queryFn: fetchCoverage, retry: 1 });
  const [selected, setSelected] = useState<{ sector: string; band: string } | null>(null);

  if (query.error instanceof OfflineError) {
    return (
      <Section className="max-w-xl p-8">
        <h1 className="font-display text-2xl text-moonlight">IVI offline</h1>
        <p className="font-data mt-3 text-sm text-rose">{query.error.precondition}</p>
        <p className="font-data mt-1 break-all text-xs text-sounding">{query.error.detail}</p>
      </Section>
    );
  }

  const atlas = query.data;
  const covered = atlas?.cells.length ?? 0;
  const totalCells = (atlas?.sectors.length ?? 0) * (atlas?.bands.length ?? 0);

  return (
    <div className="max-w-[88rem]">
      <div className="flex flex-wrap items-baseline justify-between gap-3">
        <div>
          <p className="eyebrow">sweep coverage by sector and cap band</p>
          <h1 className="font-display mt-1 text-4xl text-moonlight">Coverage</h1>
        </div>
        {atlas && (
          <span className="font-data text-xs text-sounding">
            {covered} of {totalCells} cells covered
          </span>
        )}
      </div>

      <Section condenseIndex={0} className="mt-7 p-5">
        {query.isLoading || !atlas ? (
          <p className="voice p-4">loading…</p>
        ) : atlas.bands.length === 0 ? (
          <p className="voice p-4">The ledger is empty — nothing has been swept yet.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="border-separate" style={{ borderSpacing: "3px" }}>
              <thead>
                <tr>
                  <th className="pr-3 text-left text-[10px] font-medium uppercase tracking-wider text-sounding">
                    sector
                  </th>
                  {atlas.bands.map((band) => (
                    <th
                      key={band}
                      className="px-1 pb-1 text-center text-[10px] font-medium uppercase tracking-wider text-sounding"
                    >
                      {humanizeToken(band)}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {atlas.sectors.map((sector) => (
                  <tr key={sector}>
                    <td className="whitespace-nowrap pr-3 text-xs text-sounding">
                      {humanizeToken(sector)}
                    </td>
                    {atlas.bands.map((band) => {
                      const cell = cellFor(atlas.cells, sector, band);
                      const isSelected =
                        selected?.sector === sector && selected?.band === band;
                      return (
                        <td key={band} className="p-0">
                          <button
                            type="button"
                            onClick={() =>
                              setSelected(isSelected ? null : { sector, band })
                            }
                            style={cellStyle(cell)}
                            className={`h-9 w-24 rounded-md border text-center transition-colors ${
                              isSelected
                                ? "border-glass-tint/70"
                                : cell
                                  ? "border-glass-tint/15 hover:border-glass-tint/50"
                                  : "border-glass-tint/8 hover:border-glass-tint/30"
                            }`}
                            title={
                              cell
                                ? `${humanizeToken(sector)} · ${humanizeToken(band)} — ${cell.tickers} tickers, last swept ${cell.last_loaded_at?.slice(0, 10) ?? "unknown"} (${cell.run_count} runs)`
                                : `${humanizeToken(sector)} · ${humanizeToken(band)} — never swept`
                            }
                          >
                            {cell ? (
                              <span className="font-data text-xs text-moonlight">
                                {cell.tickers}
                              </span>
                            ) : (
                              <span className="text-[10px] text-sounding/40">·</span>
                            )}
                          </button>
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Section>

      <p className="mt-3 text-[10px] text-sounding/85">
        glow = recency of complete coverage (bright within 30 days, dark water past 120) ·
        badge = tickers under coverage · probes never count · click a cell for its
        tickers, runs, and the command to fill the gap
      </p>

      {selected && (
        <CellDetail
          sector={selected.sector}
          band={selected.band}
          onClose={() => setSelected(null)}
        />
      )}
    </div>
  );
}
