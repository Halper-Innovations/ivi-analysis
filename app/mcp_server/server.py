"""The ``ivi-mcp`` MCP server: six read-only SEC EDGAR tools over stdio.

Run ``ivi-mcp`` (or ``python -m app.mcp_server.server``). Requires the optional
``mcp`` extra and ``VOE_SEC_USER_AGENT``; see docs/mcp.md.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import Field

# Only light imports at module level: anything that reaches app.config would
# load .env files at import time, before main() can report a problem cleanly.
from app.mcp_server.runtime import prepare_runtime
from app.util.credential_hygiene import InsecureEnvPermissionsError, InvalidSecUserAgentError

if TYPE_CHECKING:
    from mcp.server.mcpserver import MCPServer

SERVER_NAME = "ivi-analysis"
TOOL_NAMES = (
    "lookup_company",
    "get_company_profile",
    "list_filings",
    "get_filing_text",
    "get_financials",
    "get_concept",
)

INSTRUCTIONS = (
    "Free, read-only access to SEC EDGAR data for US-listed companies. Typical flow: "
    "lookup_company (if you only have a name) -> get_company_profile or list_filings -> "
    "get_filing_text for what a filing says, get_financials for standardized numbers "
    "(as_of=YYYY-MM-DD for what was known on a date), get_concept for any other XBRL tag. "
    "Numbers are as filed with the SEC; cite the tag and accession each value carries."
)

TickerOrCik = Annotated[
    str,
    Field(description="Ticker (e.g. 'KO', 'BRK.B') or SEC CIK (e.g. '21344' or 'CIK0000021344')."),
]
AsOf = Annotated[
    str | None,
    Field(
        description=(
            "Point-in-time date YYYY-MM-DD: use only facts filed on or before this date "
            "(as originally reported). Omit for the latest, restated view."
        )
    ),
]

_DESCRIPTIONS = {
    "lookup_company": (
        "Find SEC registrants by ticker, CIK or company name (partial names work). Returns "
        "ticker, 10-digit CIK, SEC-registered name and exchange. Use first when you only have "
        "a name or are unsure of the ticker."
    ),
    "get_company_profile": (
        "SEC registration profile: legal name, tickers, exchanges, SIC industry code and "
        "description, fiscal year end, filer category, state of incorporation, business "
        "address, former names, and the latest 10-K and 10-Q."
    ),
    "list_filings": (
        "List a company's SEC filings, newest first: accession, form, filed date, report "
        "period, primary document URL (pass it to get_filing_text) and, for 8-Ks, the item "
        "numbers. Filter by exact form names (e.g. ['10-K', '10-Q', '8-K', 'DEF 14A']; "
        "amendments such as '10-K/A' are separate forms) and a filed-date range."
    ),
    "get_filing_text": (
        "Plain text of an SEC filing document, paged. Pass a sec.gov document URL (from "
        "list_filings) or an accession number plus ticker_or_cik. Returns total_chars and "
        "next_offset; call again with offset=next_offset to read on. The first page lists "
        "'sections' (PART / Item headings with their offsets) so you can jump straight to, "
        "e.g., Item 1A Risk Factors or Item 7 MD&A."
    ),
    "get_financials": (
        "Standardized financial-statement line items from a company's XBRL filings "
        "(10-K / 10-Q), one entry per fiscal period, in USD millions (shares in millions). "
        "Each value names its XBRL tag and filing accession. as_of=YYYY-MM-DD gives a "
        "point-in-time view built only from facts filed on or before that date, so later "
        "restatements never leak in (use it for backtests or 'what was known then'). "
        "Default line items: {default}. Also available: {other}. Pass line_items=['all'] "
        "for every item."
    ),
    "get_concept": (
        "Time series for any single XBRL concept (e.g. 'AccountsPayableCurrent', "
        "'dei:EntityCommonStockSharesOutstanding') for data get_financials does not cover. "
        "One point per reporting period, newest first, with the latest filed value in raw "
        "units, its filing, and 'first_reported' when a later filing changed the number. "
        "Supports as_of (point-in-time) and a period filter (annual, quarterly, instant)."
    ),
}


def tool_descriptions() -> dict[str, str]:
    """What each tool tells the model, with the live list of financial line items."""

    from app.mcp_server.financials import ALL_LINE_ITEMS, DEFAULT_LINE_ITEMS

    other = [item for item in ALL_LINE_ITEMS if item not in DEFAULT_LINE_ITEMS]
    descriptions = dict(_DESCRIPTIONS)
    descriptions["get_financials"] = descriptions["get_financials"].format(
        default=", ".join(DEFAULT_LINE_ITEMS), other=", ".join(other)
    )
    return descriptions


def to_json(payload: Any) -> str:
    """Compact JSON: tool results are read by a model, so whitespace is wasted tokens."""

    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False, default=str)


def _call(fn: Callable[..., dict[str, Any]], **kwargs: Any) -> str:
    from mcp.server.mcpserver.exceptions import ToolError

    from app.mcp_server.errors import describe_exception
    from app.util.http import reset_domain_request_counts

    # The HTTP layer's per-host budgets are sized for one batch run; here they
    # cap a single tool call instead of the server's whole lifetime.
    reset_domain_request_counts()
    try:
        return to_json(fn(**kwargs))
    except Exception as exc:  # every failure becomes a readable tool error
        raise ToolError(describe_exception(exc)) from None


def build_server() -> MCPServer:
    """Create the MCP server with the six tools registered (no I/O happens here)."""

    from mcp.server.mcpserver import MCPServer
    from mcp.types import ToolAnnotations

    from app.mcp_server import companies, filing_text, financials

    descriptions = tool_descriptions()

    log_level = os.environ.get("VOE_MCP_LOG_LEVEL", "WARNING").strip().upper() or "WARNING"
    if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        log_level = "WARNING"
    server = MCPServer(
        name=SERVER_NAME,
        title="IVI Analysis: SEC EDGAR data",
        instructions=INSTRUCTIONS,
        version=_package_version(),
        log_level=log_level,  # type: ignore[arg-type]
    )
    read_only = ToolAnnotations(read_only_hint=True, idempotent_hint=True, open_world_hint=True)

    def register(name: str, title: str) -> Callable[[Callable[..., str]], Callable[..., str]]:
        return server.tool(
            name=name,
            title=title,
            description=descriptions[name],
            annotations=read_only,
            structured_output=False,
        )

    @register("lookup_company", "Look up a company")
    def lookup_company(
        query: Annotated[str, Field(description="Ticker, CIK, or part of a company name.")],
        limit: Annotated[int, Field(description="Maximum matches to return (1-25).")] = 10,
    ) -> str:
        return _call(companies.lookup_company, query=query, limit=limit)

    @register("get_company_profile", "Company profile")
    def get_company_profile(ticker_or_cik: TickerOrCik) -> str:
        return _call(companies.get_company_profile, ticker_or_cik=ticker_or_cik)

    @register("list_filings", "List SEC filings")
    def list_filings(
        ticker_or_cik: TickerOrCik,
        forms: Annotated[
            list[str] | str | None,
            Field(description="Exact form types to keep, e.g. ['10-K', '10-Q']. Omit for all."),
        ] = None,
        since: Annotated[str | None, Field(description="Earliest filed date, YYYY-MM-DD.")] = None,
        until: Annotated[str | None, Field(description="Latest filed date, YYYY-MM-DD.")] = None,
        limit: Annotated[int, Field(description="Maximum filings to return (1-200).")] = 20,
    ) -> str:
        return _call(
            companies.list_filings,
            ticker_or_cik=ticker_or_cik,
            forms=forms,
            since=since,
            until=until,
            limit=limit,
        )

    @register("get_filing_text", "Read a filing")
    def get_filing_text(
        url_or_accession: Annotated[
            str,
            Field(
                description=(
                    "A sec.gov document URL, or an accession number such as "
                    "'0000021344-24-000009' (then ticker_or_cik is required)."
                )
            ),
        ],
        ticker_or_cik: Annotated[
            str | None,
            Field(description="The filer's ticker or CIK; required with an accession number."),
        ] = None,
        max_chars: Annotated[
            int, Field(description="Characters per page (500-100000).")
        ] = 20_000,
        offset: Annotated[
            int, Field(description="Character offset to start from; use next_offset to continue.")
        ] = 0,
    ) -> str:
        return _call(
            filing_text.get_filing_text,
            url_or_accession=url_or_accession,
            ticker_or_cik=ticker_or_cik,
            max_chars=max_chars,
            offset=offset,
        )

    @register("get_financials", "Standardized financials")
    def get_financials(
        ticker_or_cik: TickerOrCik,
        period: Annotated[
            Literal["annual", "quarterly"], Field(description="Fiscal years or fiscal quarters.")
        ] = "annual",
        years: Annotated[
            int | None,
            Field(description="How many recent fiscal years to return (default 10 annual, 3 quarterly)."),
        ] = None,
        as_of: AsOf = None,
        line_items: Annotated[
            list[str] | str | None,
            Field(description="Line item names to return (see description), or ['all']."),
        ] = None,
    ) -> str:
        return _call(
            financials.get_financials,
            ticker_or_cik=ticker_or_cik,
            period=period,
            years=years,
            as_of=as_of,
            line_items=line_items,
        )

    @register("get_concept", "XBRL concept series")
    def get_concept(
        ticker_or_cik: TickerOrCik,
        concept: Annotated[
            str,
            Field(description="XBRL concept name, e.g. 'InventoryNet' or 'dei:EntityPublicFloat'."),
        ],
        taxonomy: Annotated[
            str, Field(description="Taxonomy: us-gaap (default), dei, ifrs-full, srt, ...")
        ] = "us-gaap",
        unit: Annotated[
            str | None, Field(description="Unit such as USD, shares, USD/shares. Default: USD if present.")
        ] = None,
        as_of: AsOf = None,
        period: Annotated[
            Literal["all", "annual", "quarterly", "instant"],
            Field(description="Keep only ~12-month, ~3-month, or point-in-time (balance) facts."),
        ] = "all",
        limit: Annotated[int, Field(description="Maximum points, newest first (1-500).")] = 40,
    ) -> str:
        return _call(
            financials.get_concept,
            ticker_or_cik=ticker_or_cik,
            concept=concept,
            taxonomy=taxonomy,
            unit=unit,
            as_of=as_of,
            period=period,
            limit=limit,
        )

    return server


def _package_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("ivi-analysis")
    except PackageNotFoundError:
        return "0.0.0"


def main() -> None:
    """Console entry point for ``ivi-mcp``: validate the environment, then serve over stdio."""

    try:
        import mcp.server.mcpserver  # noqa: F401
    except ImportError:
        print(
            "ivi-mcp needs the MCP SDK: pip install 'ivi-analysis[mcp]'",
            file=sys.stderr,
        )
        raise SystemExit(2) from None
    # Never read a .env from the project the MCP client happens to launch us in.
    os.environ.setdefault("VOE_DOTENV_CWD", "0")
    try:
        prepare_runtime()
    except (InvalidSecUserAgentError, InsecureEnvPermissionsError) as exc:
        print(f"ivi-mcp: refusing to start. {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    build_server().run("stdio")


if __name__ == "__main__":
    main()
