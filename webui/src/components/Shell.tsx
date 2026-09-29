/**
 * The IVI shell: left rail, health beacon, ⌘K palette, outlet.
 * The rail lists the full atlas; surfaces that arrive in later phases are
 * shown dimmed so the map of the instrument room is honest from day one.
 */

import { useQuery } from "@tanstack/react-query";
import { Link, Outlet, useRouterState } from "@tanstack/react-router";
import { useEffect, useState } from "react";

import { fetchMeta, type TodayResponse } from "../lib/api";
import { CommandPalette } from "./CommandPalette";

interface RailItem {
  label: string;
  to?: string;
  phase?: string;
}

/** The full atlas. Surfaces that arrive in later phases stay on the map —
 *  dimmed, with the phase in the tooltip — so the room reads honest without
 *  wearing debug labels. */
const RAIL: RailItem[] = [
  { label: "Today", to: "/" },
  { label: "Watchlist", to: "/watchlist" },
  { label: "Scans", to: "/scans" },
  { label: "Coverage", to: "/coverage" },
  { label: "Events", to: "/events" },
  { label: "Outcomes", to: "/outcomes" },
  { label: "Ops", to: "/ops" },
  { label: "Reader", to: "/reader" },
];

export function Shell() {
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [theme, setTheme] = useState<"dark" | "light">(() =>
    typeof document !== "undefined" && document.documentElement.dataset.theme === "light"
      ? "light"
      : "dark",
  );
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  const meta = useQuery({ queryKey: ["meta"], queryFn: fetchMeta, refetchInterval: 60_000 });
  // Subscribe to the Today deck's health without fetching it from the shell:
  // any surface that loads /api/today upgrades the beacon.
  const todayHealth = useQuery<TodayResponse>({ queryKey: ["today"], enabled: false }).data
    ?.health;

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setPaletteOpen((open) => !open);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const healthy = meta.data?.engine_db_present ?? null;
  const beacon =
    healthy === null
      ? "bg-slate-mist"
      : !healthy || todayHealth?.state === "RED"
        ? "bg-rose"
        : todayHealth?.state === "AMBER"
          ? "bg-amber"
          : "bg-jade";
  const beaconLabel =
    healthy === null
      ? "loading status…"
      : !healthy
        ? "books of record offline"
        : todayHealth?.state === "RED"
          ? "data health RED"
          : todayHealth?.state === "AMBER"
            ? "health warning — see Today"
            : "books of record online";

  return (
    <div className="flex min-h-screen">
      <div className="ocean" aria-hidden="true" />
      <div className="ocean-rays" aria-hidden="true" />

      {/* Narrow screens: the rail becomes a slim top bar; navigation runs
          through the command palette (⌘K / Search). No .glass-rail here —
          that class pins position:relative and would defeat `fixed`. */}
      <header className="fixed inset-x-0 top-0 z-10 flex items-center justify-between border-b border-glass-tint/10 bg-deepwater/85 px-4 py-3 backdrop-blur-md md:hidden">
        <Link to="/" className="font-display text-xl tracking-wide text-moonlight">
          ◈ IVI
        </Link>
        <div className="flex items-center gap-4">
          <button
            type="button"
            onClick={() => {
              const next = theme === "dark" ? "light" : "dark";
              setTheme(next);
              if (next === "light") {
                document.documentElement.dataset.theme = "light";
              } else {
                delete document.documentElement.dataset.theme;
              }
              try {
                localStorage.setItem("ivi-theme", next);
              } catch {
                /* private mode: theme just won't persist */
              }
            }}
            aria-label={
              theme === "dark"
                ? "Rise to the surface — switch to the light theme"
                : "Descend to the abyss — switch to the dark theme"
            }
            className="text-xs text-sounding/80 transition-colors hover:text-moonlight"
          >
            {theme === "dark" ? "◐" : "◑"}
          </button>
          <button
            type="button"
            onClick={() => setPaletteOpen(true)}
            className="glass rounded-lg px-3 py-1.5 text-xs text-sounding hover:text-moonlight"
          >
            Search ⌘K
          </button>
        </div>
      </header>

      <aside className="glass-rail sticky top-0 hidden h-screen w-56 shrink-0 flex-col px-4 py-6 md:flex">
        <Link to="/" className="mb-9 block">
          <span className="font-display text-2xl tracking-wide text-moonlight">
            ◈ IVI
          </span>
          <span className="eyebrow mt-1.5 block">watch · wait · journal</span>
        </Link>

        <nav className="flex flex-col gap-0.5" aria-label="Primary">
          {RAIL.map((item) =>
            item.to ? (
              <Link
                key={item.label}
                to={item.to}
                className={`rounded-lg px-3 py-1.5 text-sm transition-colors ${
                  pathname === item.to ||
                  (item.to !== "/" && pathname.startsWith(`${item.to}/`))
                    ? "bg-glass-tint/10 text-moonlight"
                    : "text-sounding hover:text-moonlight"
                }`}
              >
                {item.label}
              </Link>
            ) : (
              <span
                key={item.label}
                className="cursor-default rounded-lg px-3 py-1.5 text-sm text-sounding/35"
                title={`Arrives in ${item.phase}`}
              >
                {item.label}
              </span>
            ),
          )}
        </nav>

        <div className="mt-auto flex flex-col gap-3">
          <button
            type="button"
            onClick={() => {
              const next = theme === "dark" ? "light" : "dark";
              setTheme(next);
              if (next === "light") {
                document.documentElement.dataset.theme = "light";
              } else {
                delete document.documentElement.dataset.theme;
              }
              try {
                localStorage.setItem("ivi-theme", next);
              } catch {
                /* private mode: theme just won't persist */
              }
            }}
            aria-label={
              theme === "dark"
                ? "Rise to the surface — switch to the light theme"
                : "Descend to the abyss — switch to the dark theme"
            }
            className="px-3 text-left text-xs text-sounding/80 transition-colors hover:text-moonlight"
          >
            {theme === "dark" ? "◐ rise to the surface" : "◑ descend to the abyss"}
          </button>

          <Link
            to="/gauge-lab"
            className={`px-3 text-xs transition-colors ${
              pathname === "/gauge-lab"
                ? "text-moonlight"
                : "text-sounding/75 hover:text-moonlight"
            }`}
          >
            Gauge Lab
          </Link>

          <button
            type="button"
            onClick={() => setPaletteOpen(true)}
            className="glass flex items-center justify-between rounded-xl px-3 py-2 text-xs text-sounding hover:text-moonlight"
          >
            <span>Search</span>
            <kbd className="font-data text-[10px] text-sounding">⌘K</kbd>
          </button>

          <div
            className="flex items-center gap-2 px-1 text-xs"
            role="status"
            aria-live="polite"
          >
            <span className={`inline-block h-2 w-2 rounded-full ${beacon}`} />
            <span className="text-sounding">{beaconLabel}</span>
          </div>
        </div>
      </aside>

      <main className="min-w-0 flex-1 px-4 pb-8 pt-16 md:px-10 md:pt-8">
        <Outlet />
      </main>

      {paletteOpen && <CommandPalette onClose={() => setPaletteOpen(false)} />}
    </div>
  );
}
