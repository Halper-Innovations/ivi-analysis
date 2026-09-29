<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo-dark.svg">
    <img alt="IVI Analysis" src="docs/assets/logo.svg" width="360">
  </picture>
</p>

<p align="center">
  <b>Equity research on SEC filings: point-in-time fundamentals, deterministic valuation, and optional AI analyst memos.</b><br>
  Local-first. Free data by default. Every number traceable to the filing it came from.
</p>

<p align="center">
  <a href="https://github.com/Halper-Innovations/ivi-analysis/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/Halper-Innovations/ivi-analysis/actions/workflows/ci.yml/badge.svg"></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-0E6B5C"></a>
  <img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white">
  <img alt="Data: SEC EDGAR" src="https://img.shields.io/badge/data-SEC%20EDGAR-0B1F3A">
  <a href="https://github.com/astral-sh/ruff"><img alt="Ruff" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json"></a>
</p>

---

IVI Analysis reads what companies file with the SEC, turns it into clean, dated financial
statements, and values the company with transparent, deterministic methods — discounted cash
flow, earnings power, the Graham number, free-cash-flow yield, EV/EBIT, net current assets and a
tangible floor. When you want more, an AI analyst layer can read the filings and write a memo,
but the math never depends on it.

It runs on your machine, stores everything in SQLite, and needs no paid data: fundamentals come
free from SEC EDGAR, prices free from Yahoo Finance.

## Highlights

- **Point-in-time fundamentals.** Ask for any company's numbers *as they were filed on a past
  date*. Restatements filed later never leak into a historical view — the property every honest
  backtest needs.
- **Filing-grade data hygiene.** Restated figures win by filing date, not by the order SEC's data
  happens to list them. Cover-page share counts filed 1,000× off are caught by cross-checking the
  same filing's balance sheet, diluted count and net-income-per-share. Debt totals must reconcile
  to the balance sheet or they're marked unknown.
- **Deterministic valuation.** Seven methods, each with its inputs, assumptions and status stored
  beside the number. One-off swings in cash flow are smoothed both ways; genuine declines are not.
- **Fails loud, never quiet.** When a value can't be established from the filings, you get an
  explicit `UNKNOWN` with a reason code — never a silent fallback or a made-up zero.
- **Web UI.** A local React app for companies, watchlists, coverage and events (`ivi web`).
- **Optional AI analyst.** With an Anthropic or OpenAI key, research runs read filings, adjudicate
  evidence and write analyst memos with a conviction grade — always as an input, never a gate.

## Quickstart

Requires Python 3.11+.

```bash
git clone https://github.com/Halper-Innovations/ivi-analysis.git
cd ivi-analysis
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# The SEC asks every automated client to identify itself.
export VOE_SEC_USER_AGENT="Your Name you@yourdomain.com"

ivi value KO
```

`ivi value` downloads the company's SEC financial data (about 3–8 MB per company, cached
locally), runs every valuation method, fetches a free price and prints the result:

```text
KO  as of 2026-09-29  (CIK 0000021344)
Fundamentals: SEC companyfacts (FETCH_OK, 5.0 MB)
Price: 86.89 (yahoo, close of 2026-09-29, basis UNADJUSTED)
Method                     Value/share  Margin of safety  Status
DCF (base)                       21.66             -301%  OK
  range (low - high)                       20.02 - 23.37
Earnings power (EPV)             15.17             -473%  OK
Graham number                    20.51             -324%  OK
FCF yield                        27.71             -214%  OK
EV/EBIT                          12.45             -598%  OK
...
```

The engine is deliberately conservative — a 10% hurdle rate and no credit for growth it can't see
in the filings — so for quality companies the estimates often sit well below the market price.
That's the point: it tells you what the filings alone support.

## Where the data comes from

| Data | Source | Cost | Needs |
|---|---|---|---|
| Financial statements (XBRL) | SEC EDGAR `data.sec.gov` | Free | `VOE_SEC_USER_AGENT` |
| Filings and filing text | SEC EDGAR | Free | `VOE_SEC_USER_AGENT` |
| Stock prices | Yahoo Finance | Free | nothing |
| Stock prices (optional) | EODHD or Stooq | Paid / keyed | `VOE_EODHD_APIKEY` / `VOE_STOOQ_APIKEY` |
| Analyst memos (optional) | Anthropic or OpenAI | Paid | an API key |

Nothing is downloaded up front. Each company is fetched the first time you ask for it and cached
under `data/`. Covering a handful of companies takes tens of megabytes; sweeping thousands grows
into gigabytes over time.

## Configuration

Settings are `VOE_*` environment variables, optionally in a `.env` file (see
[`.env.example`](.env.example)). The `.env` in the IVI Analysis checkout is read; one in the
current directory only with `VOE_DOTENV_CWD=1`. Only `VOE_*` names are taken from a `.env`,
and a file other users can read is skipped with a warning (`chmod 600 .env`). The common ones:

| Variable | Default | Meaning |
|---|---|---|
| `VOE_SEC_USER_AGENT` | — (required) | Your name and a monitored email, sent to the SEC. |
| `VOE_DATA_DIR` | `./data` | Database, caches and reports. |
| `VOE_PRICE_PROVIDER` | `auto` | `auto` = EODHD if keyed, else Yahoo, then Stooq if keyed. `disabled` turns prices off. |
| `VOE_NET_PROVIDER` | `enabled` | The single switch for all network access; `disabled` = cache only. |
| `VOE_LLM_PROVIDER` | `disabled` | `anthropic`, `openai` or `deepseek` to enable the AI analyst layer. |
| `VOE_ISSUER_CLASSIFICATION_BY_SIC` | `true` | Classify banks/insurers by SEC industry code rather than tag names. |

## What you can do

| Command | What it does |
|---|---|
| `ivi value TICKER` | Fetch, normalize and value one company from free data. |
| `ivi web` | Serve the local web UI at http://127.0.0.1:8321 (build it first: `make webui`). |
| `ivi analyze TICKER` | Full research pipeline for a company (filings, evidence, conviction). |
| `ivi watchlist ...` | Maintain a persistent watchlist with buy targets and triggers. |
| `ivi events ...` | Detect corporate events (spin-offs, bankruptcies, busted IPOs) from filings. |
| `ivi universe ...` | Build and sweep a universe of SEC registrants. |

Run `ivi --help` for the full list.

## How it's built

```
app/
  ingest/      SEC submissions, filings and XBRL companyfacts → normalized, dated line items
  market/      prices, share counts, company-facts extraction and the share-count guard
  valuation/   DCF, EPV, Graham, multiples, net debt, normalization and quality modules
  research/    evidence gathering and research runs        analyst/  synthesis/  llm/
  watchlist/   persistent watchlist, triggers, margin of safety
  events/      corporate-event detection from filings
  universe/    registrant census, scouting and sweeps
  web/         FastAPI read model serving the React UI in webui/
```

## Development

```bash
pip install -e '.[dev]'
make test        # hermetic: no network, no local data, no subprocesses
make lint
make webui       # build the React UI (Node 20+)
```

The test suite (6,000+ tests) is hermetic by design — the harness blocks network access and your
local `data/` directory, so tests can't pass by accident against live data. See
[CONTRIBUTING.md](CONTRIBUTING.md).

## Limitations

- US SEC filers only; foreign filers reporting in other currencies are partially supported.
- Banks and insurers are recognized and routed away from methods that don't apply to them.
- Free prices are for the current day; historical price lookups need an EODHD key.
- SEC data is exactly what companies filed, including their mistakes. The guards catch many
  errors, not all.
- A reproduction-value method is not implemented yet.

## Disclaimer

IVI Analysis is research software. Nothing it produces is investment advice, and its estimates
can be wrong. Do your own work before making any investment decision.

## License

[MIT](LICENSE). Created by Ryan Halper.
