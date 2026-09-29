import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  RouterProvider,
  createRootRoute,
  createRoute,
  createRouter,
} from "@tanstack/react-router";

import { Shell } from "./components/Shell";
import { Company } from "./routes/Company";
import { Compare, type CompareSearch } from "./routes/Compare";
import { Coverage } from "./routes/Coverage";
import { Events, type EventsSearch } from "./routes/Events";
import { GaugeLab } from "./routes/GaugeLab";
import { Ops } from "./routes/Ops";
import { Outcomes } from "./routes/Outcomes";
import { Reader, type ReaderSearch } from "./routes/Reader";
import { RunDetail } from "./routes/RunDetail";
import { Scans, type ScansSearch } from "./routes/Scans";
import { Today } from "./routes/Today";
import { Watchlist, type FloorSearch } from "./routes/Watchlist";

const rootRoute = createRootRoute({ component: Shell });

const todayRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/",
  component: Today,
});

const watchlistRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/watchlist",
  component: Watchlist,
  validateSearch: (search: Record<string, unknown>): FloorSearch => {
    const str = (key: string) =>
      typeof search[key] === "string" && search[key] ? (search[key] as string) : undefined;
    const view = str("view");
    return {
      view: view === "board" || view === "ladder" ? view : undefined,
      q: str("q"),
      band: str("band"),
      grade: str("grade"),
      family: str("family"),
      status: str("status"),
    };
  },
});

const COMPANY_TABS = ["fundamentals", "research", "dossier", "decisions"] as const;
const ISO_DATE = /^\d{4}-\d{2}-\d{2}$/;

function isValidIsoDate(value: string): boolean {
  if (!ISO_DATE.test(value) || value.startsWith("0000-")) return false;
  const parsed = new Date(`${value}T00:00:00Z`);
  return !Number.isNaN(parsed.getTime()) && parsed.toISOString().slice(0, 10) === value;
}

const companyRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/company/$ticker",
  component: Company,
  validateSearch: (
    search: Record<string, unknown>,
  ): { tab?: (typeof COMPANY_TABS)[number]; basis?: "ttm"; as_of?: string } => {
    const asOf = typeof search.as_of === "string" ? search.as_of : "";
    return {
      tab: COMPANY_TABS.includes(search.tab as (typeof COMPANY_TABS)[number])
        ? (search.tab as (typeof COMPANY_TABS)[number])
        : undefined,
      basis: search.basis === "ttm" ? ("ttm" as const) : undefined,
      as_of: isValidIsoDate(asOf) ? asOf : undefined,
    };
  },
});

const compareRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/compare",
  component: Compare,
  validateSearch: (search: Record<string, unknown>): CompareSearch => ({
    tickers:
      typeof search.tickers === "string" && search.tickers ? search.tickers : undefined,
  }),
});

const gaugeLabRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/gauge-lab",
  component: GaugeLab,
});

const scansRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/scans",
  component: Scans,
  validateSearch: (search: Record<string, unknown>): ScansSearch => {
    const str = (key: string) =>
      typeof search[key] === "string" && search[key] ? (search[key] as string) : undefined;
    return {
      q: str("q"),
      sector: str("sector"),
      band: str("band"),
      verdict: str("verdict"),
      pipeline: str("pipeline"),
      history: str("history") === "all" ? "all" : undefined,
    };
  },
});

// Splat, not $ref: run references are path slugs ("root/leaf") with a slash.
// Lives under /scans/run/ so the splat can never swallow /scans itself.
const runDetailRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/scans/run/$",
  component: RunDetail,
});

const coverageRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/coverage",
  component: Coverage,
});

const eventsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/events",
  component: Events,
  validateSearch: (search: Record<string, unknown>): EventsSearch => {
    const str = (key: string) =>
      typeof search[key] === "string" && search[key] ? (search[key] as string) : undefined;
    const lane = str("lane");
    return {
      lane: lane === "opportunity" || lane === "queue_protection" ? lane : undefined,
      type: str("type"),
      q: str("q"),
    };
  },
});

const outcomesRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/outcomes",
  component: Outcomes,
});

const opsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/ops",
  component: Ops,
});

const readerRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/reader",
  component: Reader,
  validateSearch: (search: Record<string, unknown>): ReaderSearch => {
    const str = (key: string) =>
      typeof search[key] === "string" && search[key] ? (search[key] as string) : undefined;
    return { path: str("path"), q: str("q") };
  },
});

const routeTree = rootRoute.addChildren([
  todayRoute,
  watchlistRoute,
  companyRoute,
  compareRoute,
  gaugeLabRoute,
  scansRoute,
  runDetailRoute,
  coverageRoute,
  eventsRoute,
  outcomesRoute,
  opsRoute,
  readerRoute,
]);

const router = createRouter({ routeTree, defaultViewTransition: true });

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}

const queryClient = new QueryClient({
  defaultOptions: { queries: { staleTime: 30_000, refetchOnWindowFocus: false } },
});

export function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
    </QueryClientProvider>
  );
}
