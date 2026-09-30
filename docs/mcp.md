# IVI Analysis MCP server

`ivi-mcp` is a [Model Context Protocol](https://modelcontextprotocol.io) server for SEC EDGAR data.
It works with any MCP client (Claude Desktop, Claude Code and others) and provides company lookup,
filing lists, filing text, standardized financial statements and raw XBRL facts. It is free and
read-only. The only credential it needs is the contact string the SEC asks every automated client
to send.

Every number comes with its source: the XBRL tag, unit, period, and the filing (accession number,
form and filed date). Financials can also be requested as of a past date. In that case they are
rebuilt only from what had been filed by then, so later restatements don't show up in a historical
view.

## Install

The package is not on PyPI yet, so install from a clone:

```bash
git clone https://github.com/Ryan-Halper/ivi-analysis.git
cd ivi-analysis
pip install -e '.[mcp]'
```

This gives you an `ivi-mcp` command. It needs Python 3.11 or newer and version 2.x of the official
`mcp` SDK (installed by the extra). Once the package is published, `pip install 'ivi-analysis[mcp]'`
will work without a clone, and running it with `uvx` is planned.

## Configure

| Variable | Required | Meaning |
|---|---|---|
| `VOE_SEC_USER_AGENT` | yes | Your name and a real contact email, e.g. `Jane Doe jane@yourdomain.com`. The SEC requires this of every automated client ([SEC fair access](https://www.sec.gov/os/accessing-edgar-data)). The server checks it at startup and refuses to start if it is missing or a placeholder (`example.com` and `yourdomain.com` addresses are rejected). |
| `VOE_DATA_DIR` | no | Where downloaded SEC data is cached. Default: the per-user cache directory (see [Caching](#caching)). |
| `VOE_SEC_RPS` | no | Maximum requests per second to SEC (default `5`; the SEC's own limit is 10). |
| `VOE_HTTP_TIMEOUT` | no | Seconds before an SEC request times out (default `30`). |
| `VOE_MCP_LOG_LEVEL` | no | Server log level on stderr (default `WARNING`). |

The server only talks to `sec.gov` hosts.

## Claude Desktop

Edit `claude_desktop_config.json` (macOS: `~/Library/Application Support/Claude/`, Windows:
`%APPDATA%\Claude\`) and restart Claude Desktop:

```json
{
  "mcpServers": {
    "ivi-analysis": {
      "command": "ivi-mcp",
      "env": {
        "VOE_SEC_USER_AGENT": "Your Name you@yourdomain.com"
      }
    }
  }
}
```

Claude Desktop does not read your shell's `PATH`. If `ivi-mcp` is in a virtual environment, put its
absolute path in `command` (find it with `which ivi-mcp`), for example
`"/Users/you/venvs/ivi/bin/ivi-mcp"`.

## Claude Code

```bash
claude mcp add ivi-analysis --env VOE_SEC_USER_AGENT="Your Name you@yourdomain.com" -- ivi-mcp
```

Add `--scope user` to make it available in every project. Run `/mcp` inside Claude Code to check
that it connected.

## Other clients

Run `ivi-mcp` as a stdio server with `VOE_SEC_USER_AGENT` in its environment.

## Tools

| Tool | What it returns | Key arguments |
|---|---|---|
| `lookup_company` | Companies matching a ticker, CIK or (partial) name: ticker, 10-digit CIK, SEC name, exchange. | `query`, `limit` (default 10) |
| `get_company_profile` | SEC registration profile: legal name, tickers, exchanges, SIC code and industry, fiscal year end, filer category, state of incorporation, address, former names, latest 10-K and 10-Q. | `ticker_or_cik` |
| `list_filings` | Filings newest first: accession, form, filed date, report period, primary document URL, index URL, and the item numbers of 8-Ks. | `ticker_or_cik`, `forms` (exact names, e.g. `["10-K","10-Q"]`), `since`, `until` (filed date, `YYYY-MM-DD`), `limit` (default 20, max 200) |
| `get_filing_text` | Clean plain text of one filing document, in pages. The first page also lists the document's PART / Item headings with their character offsets, so you can jump to Risk Factors or MD&A. | `url_or_accession` (a sec.gov URL, or an accession number plus `ticker_or_cik`), `max_chars` (default 20000), `offset` |
| `get_financials` | Standardized line items per fiscal year or quarter, in millions, each with its XBRL tag and filing. | `ticker_or_cik`, `period` (`annual` or `quarterly`), `years` (default 10 annual, 3 quarterly), `as_of`, `line_items` |
| `get_concept` | The full history of one XBRL concept (anything `get_financials` does not cover): one point per reporting period, raw units, with the originally reported value when a later filing changed it. | `ticker_or_cik`, `concept` (e.g. `InventoryNet`, `dei:EntityPublicFloat`), `taxonomy`, `unit`, `as_of`, `period` (`all`, `annual`, `quarterly`, `instant`), `limit` (default 40) |

All tools are read-only. Failures come back as a short tool error the assistant can act on, for
example `Unknown ticker 'ZZZZ'. Use lookup_company to search by name, or pass the CIK.`, not a stack
trace.

### get_financials line items

Default line items: `revenue`, `gross_profit`, `operating_income`, `net_income`, `cfo`
(operating cash flow), `capex`, `depreciation_amortization`, `sbc`, `cash`, `total_debt`,
`total_assets`, `total_liabilities`, `equity`, `shares_outstanding`, `share_repurchases_amount`,
`dividends_paid_amount`. Also available: `income_continuing`, `retained_earnings`,
`r_and_d_total`, `deferred_revenue`, `depreciation`, `deposits`, `loans`,
`investment_securities`, `assets_under_management`, `allowance_for_credit_losses`,
`provision_for_credit_losses`, `net_charge_offs`, `nonaccrual_loans`, `interest_expense`, `sga`,
`preferred_equity`, `noncontrolling_interest`, `accounts_receivable`, `inventory`,
`accounts_payable`, `current_assets`, `current_liabilities`, `gross_ppe`,
`intangible_amortization`, `goodwill`, `intangible_assets`, `operating_lease_liability`,
`restructuring_charges`, `income_tax_expense`, `pretax_income`, `cost_of_revenue`. Pass
`line_items=["all"]` for everything.

Values are in USD millions (`shares_outstanding` in millions of shares). `capex`, buybacks and
dividends are positive outflows. A trimmed real response (Coca-Cola, `as_of="2019-03-01"`):

```json
{
  "company": {"cik": "0000021344", "ticker": "KO", "name": "COCA COLA CO"},
  "period": "annual",
  "as_of": "2019-03-01",
  "periods": [
    {"fiscal_year": 2018, "fiscal_period": "FY", "start": "2018-01-01", "end": "2018-12-31",
     "values": {
       "revenue": {"value": 31856.0, "tag": "us-gaap:Revenues", "accn": "0000021344-19-000014"},
       "operating_income": {"value": 8700.0, "tag": "us-gaap:OperatingIncomeLoss", "accn": "0000021344-19-000014"}}}
  ],
  "filings": {"0000021344-19-000014": {"form": "10-K", "filed": "2019-02-21"}},
  "source": "https://data.sec.gov/api/xbrl/companyfacts/CIK0000021344.json"
}
```

Each value carries `tag` and `accn`, and `filings` maps each accession to its form and filed date.
A value the filer reports only in parts (for example current plus non-current debt) comes back as
`sum_of` with each component's tag and filing. A value whose date differs from the period end (the
cover-page share count is dated weeks after year end) carries its own `end`.

## Point-in-time data

Pass `as_of="YYYY-MM-DD"` to `get_financials` or `get_concept` and the result is built only from
facts filed on or before that date.

- Every XBRL fact filed after `as_of` is removed before any normalization runs, and the output is
  checked again on the way out. Facts filed on `as_of` itself are included. Facts with no filing
  date are excluded.
- A year that was later restated shows the value investors saw at the time. Without `as_of` you get
  today's view, where the most recently filed value for each period wins.

For example, Coca-Cola's 2018 revenue as of 2019-03-01 is $31,856M, from the 10-K filed 2019-02-21.
Today it is $34,300M, because later 10-Ks recast the year (the latest figure comes from the 10-K
filed 2021-02-25). `get_concept` shows both in one call: the latest value plus `first_reported`.

What `as_of` does not cover:

- Ticker lookup is today's, and tickers get reassigned. For historical or delisted companies, use
  the CIK.
- Filing dates are days, not times. A 10-K filed after the market closed on `as_of` is still
  included. For trading backtests, use the day before your decision date.
- Profiles, filing lists and filing text are not filtered. Use `list_filings(until=...)` to
  restrict filings by date. `get_company_profile` describes the company as it is today.

## Caching

Unless `VOE_DATA_DIR` is set, data is cached in the per-user cache directory:

| Platform | Location |
|---|---|
| macOS | `~/Library/Caches/ivi-analysis/` |
| Linux | `$XDG_CACHE_HOME/ivi-analysis/`, or `~/.cache/ivi-analysis/` |
| Windows | `%LOCALAPPDATA%\ivi-analysis\` |

Downloaded SEC responses sit under `cache/http/` inside that directory. Roughly, per company:
0.1-0.2 MB for the filing index, 1-6 MB for the XBRL company facts (Coca-Cola 5.0 MB, Nathan's
Famous 3.0 MB), and 1-4 MB for each filing document read (Coca-Cola's 10-K is 3.8 MB of HTML).
Researching a company typically costs 2-10 MB, plus about 1.3 MB once for SEC's ticker lists.

Refresh intervals: ticker lists 7 days, filing indexes 1 hour, company facts 24 hours. Filing
documents are kept indefinitely, since EDGAR documents never change. Deleting the directory is
always safe.

## Limits

- US SEC registrants only. Structured financials exist only for companies that file XBRL (large
  filers since 2009, smaller ones since 2011). Older history is in the filings themselves, so use
  `get_filing_text`.
- US GAAP only. The standardized set reads `us-gaap` tags. Foreign private issuers that report
  under IFRS (many 20-F and 40-F filers) have no standardized values. Use `get_concept` with
  `taxonomy="ifrs-full"`.
- Standardization is a fixed tag list and the first match wins, so check `tag`. For example,
  `operating_income` falls back to a pre-tax income tag when a company does not tag operating
  income, and the `total_debt` definition is being revised. Values outside plausibility bounds are
  dropped (for example revenue under $10,000).
- Quarters come from 10-Qs. The fourth quarter is never reported on its own; derive it as the full
  year minus Q1-Q3. Quarterly `fiscal_year` and `fiscal_period` follow the company's own fiscal
  calendar. Annual `fiscal_year` is the calendar year in which the fiscal year ends.
- SEC's company-facts feed can lag the filing index by weeks for some companies. If the newest 10-Q
  is missing from `get_financials`, read it with `get_filing_text`.
- Text extraction handles HTML and plain-text documents, not PDFs or images. The section index is a
  heuristic for 10-K and 10-Q style "PART / Item" headings. Exhibits (such as an 8-K's EX-99.1
  press release) are separate documents: read the filing's `index_url` to find their file names,
  then fetch them by URL.
- Requests are rate-limited (`VOE_SEC_RPS`). The first request for a large company's facts
  downloads several megabytes and can take a few seconds.
- The repository's valuation engine is not exposed yet. Its math has open, documented defects that
  will be fixed first.

The data is exactly what companies filed with the SEC, including their mistakes. It is not
investment advice.

## Troubleshooting

- `ivi-mcp: refusing to start. SEC_USER_AGENT_INVALID ...`: set `VOE_SEC_USER_AGENT` in the
  client's config (not only in your shell) to your name and a real email address.
- `ivi-mcp: refusing to start. ENV_PERMISSIONS_INSECURE ...`: the server found a `.env` file in the
  IVI Analysis source checkout that other users on the machine can read, and it will not load
  credentials from it. Run the `chmod 600` it suggests, or configure the server through the
  client's `env` settings instead. The server never reads a `.env` from the folder the client
  launches it in, so an unrelated project's `.env` can't stop it.
- `SEC EDGAR refused the request (HTTP 403)`: SEC rejected the identity or throttled you. Check the
  email address, lower `VOE_SEC_RPS`, and wait a minute.
- The client says the server failed to start: run `ivi-mcp` in a terminal with the same
  environment. Startup errors print to stderr there. Claude Desktop also keeps server logs (macOS:
  `~/Library/Logs/Claude/mcp-server-ivi-analysis.log`).
