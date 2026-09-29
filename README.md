<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo-dark.svg">
    <img alt="IVI Analysis" src="docs/assets/logo.svg" width="360">
  </picture>
</p>

<p align="center">
  <a href="https://github.com/Halper-Innovations/ivi-analysis/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/Halper-Innovations/ivi-analysis/actions/workflows/ci.yml/badge.svg"></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-0E6B5C"></a>
  <img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white">
  <img alt="Data: SEC EDGAR" src="https://img.shields.io/badge/data-SEC%20EDGAR-0B1F3A">
  <a href="docs/mcp.md"><img alt="MCP server" src="https://img.shields.io/badge/MCP-server-5EEAD4"></a>
</p>

IVI Analysis values public companies using the financial statements they file with the SEC. It
downloads a company's XBRL data from EDGAR, cleans it up into annual and quarterly line items, and
runs several standard valuation methods on it: a discounted cash flow model, earnings power value,
the Graham number, free cash flow yield, EV/EBIT, net current asset value and a tangible book floor.

Everything runs locally and is stored in SQLite. You don't need any paid data. Financials come from
the SEC for free and prices come from Yahoo Finance. If you add an Anthropic or OpenAI key, there is
also an optional research layer that reads filings and writes an analyst memo, but the valuation
numbers never depend on it.

## Getting started

You need Python 3.11 or newer.

```bash
git clone https://github.com/Halper-Innovations/ivi-analysis.git
cd ivi-analysis
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# The SEC asks every automated client to identify itself with a name and email.
export VOE_SEC_USER_AGENT="Your Name you@yourdomain.com"

ivi value KO
```

The first run for a company downloads its SEC data (usually 3 to 8 MB) and caches it. Output looks
like this:

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

The models are conservative on purpose. They use a 10% discount rate and don't give credit for
growth that isn't already visible in the filings, so for a lot of well known companies the
estimates come out well below the share price. Treat them as a floor based on what the company
has actually reported, not a price target.

## What it does with the data

A few things the data layer handles that raw XBRL doesn't:

- You can ask for a company's numbers as of any past date and get only what had been filed by then.
  Later restatements don't leak into the historical view, which matters if you backtest anything.
- When a figure has been restated, the most recently filed value is used.
- Share counts on filing cover pages are sometimes off by a factor of 1,000. These are checked
  against other share counts in the same filing (balance sheet, diluted weighted average, and net
  income divided by EPS) and rejected when they don't agree.
- Total debt has to be consistent with the balance sheet. If the tagged pieces don't add up, debt
  is reported as unknown instead of guessed.
- Banks, insurers and REITs are identified by their SEC industry code and aren't run through
  methods that don't make sense for them.

When a value can't be worked out from the filings, the output says so and gives a reason code
rather than filling in a zero.

## Where the data comes from

| Data | Source | Cost | Setting needed |
|---|---|---|---|
| Financial statements (XBRL) | SEC EDGAR | Free | `VOE_SEC_USER_AGENT` |
| Filings and filing text | SEC EDGAR | Free | `VOE_SEC_USER_AGENT` |
| Stock prices | Yahoo Finance | Free | none |
| Stock prices (optional) | EODHD or Stooq | Paid | `VOE_EODHD_APIKEY` or `VOE_STOOQ_APIKEY` |
| Analyst memos (optional) | Anthropic or OpenAI | Paid | an API key |

Nothing is downloaded until you ask for a company. A handful of companies takes tens of megabytes.
If you sweep thousands of companies the cache grows into gigabytes.

## Configuration

All settings are environment variables starting with `VOE_`. You can also put them in a `.env`
file in the repo checkout (see [`.env.example`](.env.example)). Only `VOE_` names are read from
that file, and it's skipped with a warning if other users on the machine can read it, so run
`chmod 600 .env`. A `.env` in whatever directory you run the command from is ignored unless you set
`VOE_DOTENV_CWD=1`.

| Variable | Default | Meaning |
|---|---|---|
| `VOE_SEC_USER_AGENT` | required | Your name and an email address you check. Sent to the SEC. |
| `VOE_DATA_DIR` | `./data` | Where the database, caches and reports go. |
| `VOE_PRICE_PROVIDER` | `auto` | `auto` uses EODHD if you have a key, otherwise Yahoo. `disabled` turns prices off. |
| `VOE_NET_PROVIDER` | `enabled` | Set to `disabled` to work only from the local cache. |
| `VOE_LLM_PROVIDER` | `disabled` | `anthropic`, `openai` or `deepseek` to turn on the research layer. |

## Commands

| Command | What it does |
|---|---|
| `ivi value TICKER` | Download, clean and value one company. |
| `ivi web` | Start the local web UI at http://127.0.0.1:8321. Run `make webui` once first to build it. |
| `ivi analyze TICKER` | Run the full research pipeline on a company. |
| `ivi watchlist ...` | Keep a watchlist with buy targets and price triggers. |
| `ivi events ...` | Look for corporate events in filings (spin-offs, bankruptcies, broken IPOs). |
| `ivi universe ...` | Build and scan a list of SEC registrants. |
| `ivi-mcp` | Start the MCP server (see below). |

`ivi --help` lists everything.

## MCP server

There's also an MCP server, so Claude Desktop, Claude Code or another MCP client can look things up
in EDGAR directly.

```bash
pip install -e '.[mcp]'
claude mcp add ivi-analysis --env VOE_SEC_USER_AGENT="Your Name you@yourdomain.com" -- ivi-mcp
```

It has six read-only tools: company lookup, company profile, filing lists, filing text, financial
statements (annual or quarterly, optionally as of a past date) and raw XBRL concepts. You can ask
something like "What revenue did Coca-Cola report for 2018 in its original 10-K, and what does it
show now?" Setup for other clients is in [docs/mcp.md](docs/mcp.md).

## Project layout

```
app/
  ingest/      SEC submissions, filings and XBRL company facts
  market/      prices, share counts and company facts extraction
  valuation/   DCF, EPV, Graham, multiples, net debt and the normalization modules
  research/    evidence gathering and research runs (also analyst/, synthesis/, llm/)
  watchlist/   watchlist, triggers and margin of safety
  events/      corporate event detection
  universe/    registrant lists, scouting and sweeps
  web/         FastAPI backend for the React UI in webui/
  mcp_server/  the MCP server
```

## Development

```bash
pip install -e '.[dev]'
make test     # no network access, no local data
make lint
make webui    # builds the React UI, needs Node 20+
```

The tests block network access and the local `data/` directory, so they can't pass by accidentally
reading real data. More in [CONTRIBUTING.md](CONTRIBUTING.md).

## Limitations

- Only SEC filers are supported. Foreign filers that report in other currencies only partly work.
- Free prices are current prices. For historical prices you need an EODHD key.
- The data is whatever companies filed, mistakes included. The checks catch a lot of errors but
  not all of them.
- There's no reproduction value method yet.

## Disclaimer

This is research software. Nothing it outputs is investment advice and the estimates can be wrong.

## License

[MIT](LICENSE). Created by Ryan Halper.
