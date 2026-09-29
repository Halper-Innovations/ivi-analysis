"""Pure parsers for registry-first U.S. equity census source snapshots.

This module deliberately performs no network access and no writes.  Callers
provide already-captured source content, then decide how and where to persist
the parsed records and the explicit parse issues returned alongside them.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Generic, Mapping, Sequence, TypeVar


SEC_COMPANY_TICKERS_EXCHANGE_URL = "https://www.sec.gov/files/company_tickers_exchange.json"

NASDAQ_LISTED_SOURCE = "nasdaqlisted"
OTHER_LISTED_SOURCE = "otherlisted"

# Nasdaq Trader symbol-directory definitions:
# Q = Global Select, G = Global Market, S = Capital Market.
NASDAQ_MARKET_CATEGORIES: Mapping[str, str] = {
    "Q": "NASDAQ_GLOBAL_SELECT_MARKET",
    "G": "NASDAQ_GLOBAL_MARKET",
    "S": "NASDAQ_CAPITAL_MARKET",
}

# Consolidated exchange codes from the Nasdaq Trader symbol directory.  Q/G/S
# are also accepted here so callers can use one normalization helper for both
# files without collapsing the market-tier evidence.
NASDAQ_TRADER_EXCHANGES: Mapping[str, str] = {
    "A": "NYSE American",
    "M": "NYSE Texas",
    "N": "NYSE",
    "P": "NYSE Arca",
    "Q": "Nasdaq",
    "G": "Nasdaq",
    "S": "Nasdaq",
    "V": "IEX",
    "Z": "Cboe BZX",
}


@dataclass(frozen=True)
class ParseIssue:
    """A source anomaly that must remain visible to census accounting."""

    source: str
    code: str
    message: str
    row_number: int | None = None
    raw_record: Any = None


T = TypeVar("T")


@dataclass(frozen=True)
class ParseResult(Generic[T]):
    records: tuple[T, ...]
    issues: tuple[ParseIssue, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SecExchangeSecurity:
    cik: str
    name: str
    ticker: str
    exchange: str | None
    source_url: str
    source_row_number: int
    source_fields: tuple[str, ...]
    source_row: tuple[Any, ...]


@dataclass(frozen=True)
class NasdaqTraderSecurity:
    symbol: str
    security_name: str
    exchange: str | None
    exchange_code: str
    market_tier: str | None
    etf: bool | None
    test_issue: bool | None
    financial_status: str | None
    round_lot_size: int | None
    next_shares: bool | None
    cqs_symbol: str | None
    nasdaq_symbol: str | None
    file_creation_timestamp: str | None
    file_creation_timestamp_raw: str | None
    source_file: str
    source_line_number: int
    source_fields: tuple[str, ...]
    source_row: tuple[str, ...]


@dataclass(frozen=True)
class CompaniesMarketCapSecurity:
    rank: int
    name: str
    ticker: str
    market_cap_usd: int
    source_url: str
    as_of_date: str
    market_cap_literal: str
    market_cap_data_sort_usd: int | None
    source_line_number: int


@dataclass(frozen=True)
class StockAnalysisSecurity:
    symbol: str
    name: str
    market_cap_usd: int | None
    source_url: str
    as_of_date: str
    source_row_number: int
    source_line_number: int
    source_object: str
    industry: str | None = None


@dataclass(frozen=True)
class SourceFileManifest:
    source_id: str
    path: str
    size_bytes: int
    sha256: str


def _decode_json_payload(payload: str | bytes | Mapping[str, Any]) -> tuple[Any, str | None]:
    if isinstance(payload, Mapping):
        return payload, None
    try:
        if isinstance(payload, bytes):
            return json.loads(payload.decode("utf-8-sig")), None
        return json.loads(payload), None
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
        return None, str(exc)


def parse_sec_company_tickers_exchange(
    payload: str | bytes | Mapping[str, Any],
    *,
    source_url: str = SEC_COMPANY_TICKERS_EXCHANGE_URL,
) -> ParseResult[SecExchangeSecurity]:
    """Parse the SEC exchange registry without collapsing security rows."""

    decoded, decode_error = _decode_json_payload(payload)
    if decode_error is not None:
        return ParseResult(
            records=(),
            issues=(
                ParseIssue(
                    source=source_url,
                    code="INVALID_JSON",
                    message=f"SEC exchange registry is not valid JSON: {decode_error}",
                ),
            ),
        )
    if not isinstance(decoded, Mapping):
        return ParseResult(
            records=(),
            issues=(
                ParseIssue(
                    source=source_url,
                    code="INVALID_TOP_LEVEL",
                    message="SEC exchange registry must be a JSON object",
                    raw_record=decoded,
                ),
            ),
        )

    fields_value = decoded.get("fields")
    if not isinstance(fields_value, list) or not all(
        isinstance(value, str) for value in fields_value
    ):
        return ParseResult(
            records=(),
            issues=(
                ParseIssue(
                    source=source_url,
                    code="INVALID_FIELDS",
                    message="SEC exchange registry fields must be a list of strings",
                    raw_record=fields_value,
                ),
            ),
        )
    fields = tuple(fields_value)
    required_fields = ("cik", "name", "ticker", "exchange")
    missing_fields = tuple(field_name for field_name in required_fields if field_name not in fields)
    if missing_fields:
        return ParseResult(
            records=(),
            issues=(
                ParseIssue(
                    source=source_url,
                    code="MISSING_REQUIRED_FIELDS",
                    message="SEC exchange registry is missing fields: " + ", ".join(missing_fields),
                    raw_record=fields,
                ),
            ),
            metadata={"fields": fields},
        )

    data = decoded.get("data")
    if not isinstance(data, list):
        return ParseResult(
            records=(),
            issues=(
                ParseIssue(
                    source=source_url,
                    code="INVALID_DATA",
                    message="SEC exchange registry data must be a list",
                    raw_record=data,
                ),
            ),
            metadata={"fields": fields},
        )

    records: list[SecExchangeSecurity] = []
    issues: list[ParseIssue] = []
    for row_number, row_value in enumerate(data, start=1):
        if not isinstance(row_value, (list, tuple)):
            issues.append(
                ParseIssue(
                    source=source_url,
                    code="INVALID_ROW_TYPE",
                    message="SEC exchange registry row must be a list",
                    row_number=row_number,
                    raw_record=row_value,
                )
            )
            continue
        row = tuple(row_value)
        if len(row) != len(fields):
            issues.append(
                ParseIssue(
                    source=source_url,
                    code="ROW_FIELD_COUNT_MISMATCH",
                    message=f"SEC row has {len(row)} values; expected {len(fields)}",
                    row_number=row_number,
                    raw_record=row,
                )
            )
            continue
        mapped = dict(zip(fields, row, strict=True))
        normalized: dict[str, str | None] = {
            "cik": str(mapped["cik"] if mapped["cik"] is not None else "").strip(),
            "name": str(mapped["name"] if mapped["name"] is not None else "").strip(),
            "ticker": str(mapped["ticker"] if mapped["ticker"] is not None else "").strip().upper(),
            # A null exchange is a valid SEC registry state for an off-exchange
            # security.  Preserve it for an explicit downstream disposition.
            "exchange": (
                str(mapped["exchange"]).strip() if mapped["exchange"] is not None else None
            ),
        }
        missing_values = tuple(key for key in ("cik", "name", "ticker") if not normalized[key])
        if missing_values:
            issues.append(
                ParseIssue(
                    source=source_url,
                    code="MISSING_REQUIRED_VALUE",
                    message="SEC row is missing values: " + ", ".join(missing_values),
                    row_number=row_number,
                    raw_record=row,
                )
            )
            continue
        records.append(
            SecExchangeSecurity(
                cik=str(normalized["cik"]),
                name=str(normalized["name"]),
                ticker=str(normalized["ticker"]),
                exchange=normalized["exchange"],
                source_url=source_url,
                source_row_number=row_number,
                source_fields=fields,
                source_row=row,
            )
        )

    return ParseResult(
        records=tuple(records),
        issues=tuple(issues),
        metadata={"fields": fields, "source_row_count": len(data)},
    )


def normalize_nasdaq_trader_exchange(code: str) -> str | None:
    """Return the canonical exchange label for a Nasdaq Trader code."""

    return NASDAQ_TRADER_EXCHANGES.get(code.strip().upper())


def _parse_creation_timestamp(raw: str) -> str | None:
    try:
        return datetime.strptime(raw, "%m%d%Y%H:%M").isoformat(timespec="seconds")
    except ValueError:
        return None


def _yes_no_flag(
    raw: str,
    *,
    field_name: str,
    source: str,
    row_number: int,
    raw_record: tuple[str, ...],
    issues: list[ParseIssue],
) -> bool | None:
    value = raw.strip().upper()
    if value == "Y":
        return True
    if value == "N":
        return False
    issues.append(
        ParseIssue(
            source=source,
            code="INVALID_YES_NO_FLAG",
            message=f"{field_name} must be Y or N; got {raw!r}",
            row_number=row_number,
            raw_record=raw_record,
        )
    )
    return None


def _round_lot_size(
    raw: str,
    *,
    source: str,
    row_number: int,
    raw_record: tuple[str, ...],
    issues: list[ParseIssue],
) -> int | None:
    try:
        value = int(raw.strip())
    except ValueError:
        value = -1
    if value >= 0:
        return value
    issues.append(
        ParseIssue(
            source=source,
            code="INVALID_ROUND_LOT_SIZE",
            message=f"Round Lot Size must be a non-negative integer; got {raw!r}",
            row_number=row_number,
            raw_record=raw_record,
        )
    )
    return None


def _parse_nasdaq_trader_file(
    text: str,
    *,
    source_file: str,
    expected_fields: Sequence[str],
) -> ParseResult[NasdaqTraderSecurity]:
    lines = text.lstrip("\ufeff").splitlines()
    nonblank = [(line_number, line) for line_number, line in enumerate(lines, start=1) if line]
    if not nonblank:
        return ParseResult(
            records=(),
            issues=(
                ParseIssue(
                    source=source_file,
                    code="EMPTY_FILE",
                    message="Nasdaq Trader symbol-directory snapshot is empty",
                ),
            ),
        )

    header_line_number, header_line = nonblank[0]
    fields = tuple(header_line.split("|"))
    missing_fields = tuple(field_name for field_name in expected_fields if field_name not in fields)
    if missing_fields:
        return ParseResult(
            records=(),
            issues=(
                ParseIssue(
                    source=source_file,
                    code="MISSING_REQUIRED_FIELDS",
                    message="Nasdaq Trader file is missing fields: " + ", ".join(missing_fields),
                    row_number=header_line_number,
                    raw_record=fields,
                ),
            ),
            metadata={"fields": fields},
        )

    issues: list[ParseIssue] = []
    creation_rows: list[tuple[int, str, tuple[str, ...]]] = []
    for line_number, line in nonblank[1:]:
        row = tuple(line.split("|"))
        if row and row[0].startswith("File Creation Time:"):
            creation_rows.append((line_number, row[0].partition(":")[2].strip(), row))

    creation_raw: str | None = None
    creation_timestamp: str | None = None
    if not creation_rows:
        issues.append(
            ParseIssue(
                source=source_file,
                code="MISSING_FILE_CREATION_TIME",
                message="Nasdaq Trader file has no File Creation Time footer",
            )
        )
    else:
        creation_line, creation_raw, creation_row = creation_rows[-1]
        creation_timestamp = _parse_creation_timestamp(creation_raw)
        if creation_timestamp is None:
            issues.append(
                ParseIssue(
                    source=source_file,
                    code="INVALID_FILE_CREATION_TIME",
                    message=f"Invalid Nasdaq Trader File Creation Time: {creation_raw!r}",
                    row_number=creation_line,
                    raw_record=creation_row,
                )
            )
        if len(creation_rows) > 1:
            issues.append(
                ParseIssue(
                    source=source_file,
                    code="MULTIPLE_FILE_CREATION_TIMES",
                    message="Nasdaq Trader file has multiple File Creation Time footers",
                    raw_record=tuple(row[1] for row in creation_rows),
                )
            )

    records: list[NasdaqTraderSecurity] = []
    for line_number, line in nonblank[1:]:
        row = tuple(line.split("|"))
        if row and row[0].startswith("File Creation Time:"):
            continue
        if len(row) != len(fields):
            issues.append(
                ParseIssue(
                    source=source_file,
                    code="ROW_FIELD_COUNT_MISMATCH",
                    message=f"Nasdaq Trader row has {len(row)} values; expected {len(fields)}",
                    row_number=line_number,
                    raw_record=row,
                )
            )
            continue
        mapped = dict(zip(fields, row, strict=True))
        symbol_field = "Symbol" if source_file == NASDAQ_LISTED_SOURCE else "ACT Symbol"
        symbol = mapped[symbol_field].strip().upper()
        security_name = mapped["Security Name"].strip()
        exchange_code = (
            mapped["Market Category"].strip().upper()
            if source_file == NASDAQ_LISTED_SOURCE
            else mapped["Exchange"].strip().upper()
        )
        missing_values = tuple(
            key
            for key, value in (
                (symbol_field, symbol),
                ("Security Name", security_name),
                (
                    "Market Category" if source_file == NASDAQ_LISTED_SOURCE else "Exchange",
                    exchange_code,
                ),
            )
            if not value
        )
        if missing_values:
            issues.append(
                ParseIssue(
                    source=source_file,
                    code="MISSING_REQUIRED_VALUE",
                    message="Nasdaq Trader row is missing values: " + ", ".join(missing_values),
                    row_number=line_number,
                    raw_record=row,
                )
            )
            continue

        market_tier = NASDAQ_MARKET_CATEGORIES.get(exchange_code)
        exchange = normalize_nasdaq_trader_exchange(exchange_code)
        if exchange is None or (source_file == NASDAQ_LISTED_SOURCE and market_tier is None):
            issues.append(
                ParseIssue(
                    source=source_file,
                    code="UNKNOWN_EXCHANGE_CODE",
                    message=f"Unknown Nasdaq Trader exchange code: {exchange_code!r}",
                    row_number=line_number,
                    raw_record=row,
                )
            )

        records.append(
            NasdaqTraderSecurity(
                symbol=symbol,
                security_name=security_name,
                exchange=exchange,
                exchange_code=exchange_code,
                market_tier=market_tier,
                etf=_yes_no_flag(
                    mapped["ETF"],
                    field_name="ETF",
                    source=source_file,
                    row_number=line_number,
                    raw_record=row,
                    issues=issues,
                ),
                test_issue=_yes_no_flag(
                    mapped["Test Issue"],
                    field_name="Test Issue",
                    source=source_file,
                    row_number=line_number,
                    raw_record=row,
                    issues=issues,
                ),
                financial_status=mapped.get("Financial Status", "").strip() or None,
                round_lot_size=_round_lot_size(
                    mapped["Round Lot Size"],
                    source=source_file,
                    row_number=line_number,
                    raw_record=row,
                    issues=issues,
                ),
                next_shares=(
                    _yes_no_flag(
                        mapped["NextShares"],
                        field_name="NextShares",
                        source=source_file,
                        row_number=line_number,
                        raw_record=row,
                        issues=issues,
                    )
                    if "NextShares" in mapped
                    else None
                ),
                cqs_symbol=mapped.get("CQS Symbol", "").strip() or None,
                nasdaq_symbol=mapped.get("NASDAQ Symbol", "").strip() or None,
                file_creation_timestamp=creation_timestamp,
                file_creation_timestamp_raw=creation_raw,
                source_file=source_file,
                source_line_number=line_number,
                source_fields=fields,
                source_row=row,
            )
        )

    return ParseResult(
        records=tuple(records),
        issues=tuple(issues),
        metadata={
            "fields": fields,
            "file_creation_timestamp": creation_timestamp,
            "file_creation_timestamp_raw": creation_raw,
            "source_row_count": len(records)
            + sum(
                issue.code in {"ROW_FIELD_COUNT_MISMATCH", "MISSING_REQUIRED_VALUE"}
                for issue in issues
            ),
        },
    )


def parse_nasdaqlisted(text: str) -> ParseResult[NasdaqTraderSecurity]:
    """Parse a captured ``nasdaqlisted.txt`` snapshot."""

    return _parse_nasdaq_trader_file(
        text,
        source_file=NASDAQ_LISTED_SOURCE,
        expected_fields=(
            "Symbol",
            "Security Name",
            "Market Category",
            "Test Issue",
            "Financial Status",
            "Round Lot Size",
            "ETF",
            "NextShares",
        ),
    )


def parse_otherlisted(text: str) -> ParseResult[NasdaqTraderSecurity]:
    """Parse a captured ``otherlisted.txt`` snapshot."""

    return _parse_nasdaq_trader_file(
        text,
        source_file=OTHER_LISTED_SOURCE,
        expected_fields=(
            "ACT Symbol",
            "Security Name",
            "Exchange",
            "CQS Symbol",
            "ETF",
            "Round Lot Size",
            "Test Issue",
            "NASDAQ Symbol",
        ),
    )


_MARKET_CAP_LITERAL = re.compile(
    r"^(?:US\s*)?\$?\s*([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?)"
    r"\s*(K|M|B|T|THOUSAND|MILLION|BILLION|TRILLION)?$",
    re.IGNORECASE,
)
_MARKET_CAP_MULTIPLIERS: Mapping[str, Decimal] = {
    "": Decimal(1),
    "K": Decimal(1_000),
    "THOUSAND": Decimal(1_000),
    "M": Decimal(1_000_000),
    "MILLION": Decimal(1_000_000),
    "B": Decimal(1_000_000_000),
    "BILLION": Decimal(1_000_000_000),
    "T": Decimal(1_000_000_000_000),
    "TRILLION": Decimal(1_000_000_000_000),
}


def parse_market_cap_usd_literal(raw: str) -> int:
    """Parse a CompaniesMarketCap display literal into exact whole USD."""

    normalized = " ".join(raw.replace("\xa0", " ").split())
    match = _MARKET_CAP_LITERAL.fullmatch(normalized)
    if match is None:
        raise ValueError(f"invalid USD market-cap literal: {raw!r}")
    try:
        amount = Decimal(match.group(1).replace(",", ""))
    except InvalidOperation as exc:
        raise ValueError(f"invalid USD market-cap literal: {raw!r}") from exc
    value = amount * _MARKET_CAP_MULTIPLIERS[(match.group(2) or "").upper()]
    integral = value.to_integral_value()
    if value != integral or integral < 0:
        raise ValueError(f"market-cap literal is not a non-negative whole USD value: {raw!r}")
    return int(integral)


@dataclass
class _HtmlCell:
    classes: frozenset[str]
    data_sort: str | None
    text_parts: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join("".join(self.text_parts).split())


@dataclass
class _HtmlCompanyRow:
    source_line_number: int
    cells: list[_HtmlCell] = field(default_factory=list)
    company_name_parts: list[str] = field(default_factory=list)
    company_code_parts: list[str] = field(default_factory=list)

    @property
    def company_name(self) -> str:
        return " ".join("".join(self.company_name_parts).split())

    @property
    def company_code(self) -> str:
        return " ".join("".join(self.company_code_parts).split())


class _CompaniesMarketCapHtmlParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[_HtmlCompanyRow] = []
        self.current_row: _HtmlCompanyRow | None = None
        self.current_cell: _HtmlCell | None = None
        self.capture_company_name = 0
        self.capture_company_code = 0

    @staticmethod
    def _attributes(attrs: list[tuple[str, str | None]]) -> dict[str, str]:
        return {key: value or "" for key, value in attrs}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = self._attributes(attrs)
        classes = frozenset(values.get("class", "").split())
        if tag == "tr":
            if self.current_row is not None:
                self.rows.append(self.current_row)
            self.current_row = _HtmlCompanyRow(source_line_number=self.getpos()[0])
            self.current_cell = None
            self.capture_company_name = 0
            self.capture_company_code = 0
            return
        if self.current_row is None:
            return
        if tag == "td":
            self.current_cell = _HtmlCell(classes=classes, data_sort=values.get("data-sort"))
            self.current_row.cells.append(self.current_cell)
        if tag == "div" and "company-name" in classes:
            self.capture_company_name = 1
        elif tag == "div" and "company-code" in classes:
            self.capture_company_code = 1
        else:
            if self.capture_company_name and tag == "div":
                self.capture_company_name += 1
            if self.capture_company_code and tag == "div":
                self.capture_company_code += 1

    def handle_endtag(self, tag: str) -> None:
        if tag == "td":
            self.current_cell = None
        elif tag == "div":
            if self.capture_company_name:
                self.capture_company_name -= 1
            if self.capture_company_code:
                self.capture_company_code -= 1
        elif tag == "tr" and self.current_row is not None:
            self.rows.append(self.current_row)
            self.current_row = None
            self.current_cell = None
            self.capture_company_name = 0
            self.capture_company_code = 0

    def handle_data(self, data: str) -> None:
        if self.current_cell is not None:
            self.current_cell.text_parts.append(data)
        if self.current_row is not None and self.capture_company_name:
            self.current_row.company_name_parts.append(data)
        if self.current_row is not None and self.capture_company_code:
            self.current_row.company_code_parts.append(data)

    def close(self) -> None:
        super().close()
        if self.current_row is not None:
            self.rows.append(self.current_row)
            self.current_row = None


def _canonical_as_of(value: str | date) -> str:
    if isinstance(value, date):
        return value.isoformat()
    return date.fromisoformat(value).isoformat()


def parse_companiesmarketcap_usa_html(
    html: str,
    *,
    source_url: str,
    as_of_date: str | date,
) -> ParseResult[CompaniesMarketCapSecurity]:
    """Parse one USA ranking page while retaining malformed row evidence."""

    canonical_as_of = _canonical_as_of(as_of_date)
    parser = _CompaniesMarketCapHtmlParser()
    parser.feed(html)
    parser.close()

    records: list[CompaniesMarketCapSecurity] = []
    issues: list[ParseIssue] = []
    candidate_rows = 0
    for row_number, row in enumerate(parser.rows, start=1):
        rank_cells = [cell for cell in row.cells if "rank-td" in cell.classes]
        is_candidate = bool(rank_cells or row.company_name or row.company_code)
        if not is_candidate:
            continue
        candidate_rows += 1
        name_cells = [cell for cell in row.cells if "name-td" in cell.classes]
        rank_literal = rank_cells[0].text if rank_cells else ""
        market_cap_cell: _HtmlCell | None = None
        if name_cells:
            name_index = row.cells.index(name_cells[0])
            for cell in row.cells[name_index + 1 :]:
                if "td-right" in cell.classes:
                    market_cap_cell = cell
                    break
        missing = []
        if not rank_literal:
            missing.append("rank")
        if not row.company_name:
            missing.append("name")
        if not row.company_code:
            missing.append("ticker")
        if market_cap_cell is None or not market_cap_cell.text:
            missing.append("market_cap")
        if missing:
            issues.append(
                ParseIssue(
                    source=source_url,
                    code="MALFORMED_COMPANY_ROW",
                    message="CompaniesMarketCap row is missing: " + ", ".join(missing),
                    row_number=row_number,
                    raw_record={
                        "source_line_number": row.source_line_number,
                        "rank": rank_literal,
                        "name": row.company_name,
                        "ticker": row.company_code,
                        "cells": tuple(cell.text for cell in row.cells),
                    },
                )
            )
            continue
        try:
            rank = int(rank_literal.replace(",", ""))
            if rank <= 0:
                raise ValueError
        except ValueError:
            issues.append(
                ParseIssue(
                    source=source_url,
                    code="INVALID_RANK",
                    message=f"CompaniesMarketCap rank must be a positive integer: {rank_literal!r}",
                    row_number=row_number,
                    raw_record=rank_literal,
                )
            )
            continue

        assert market_cap_cell is not None
        try:
            market_cap_usd = parse_market_cap_usd_literal(market_cap_cell.text)
        except ValueError as exc:
            issues.append(
                ParseIssue(
                    source=source_url,
                    code="INVALID_MARKET_CAP_LITERAL",
                    message=str(exc),
                    row_number=row_number,
                    raw_record=market_cap_cell.text,
                )
            )
            continue

        market_cap_data_sort_usd: int | None = None
        if market_cap_cell.data_sort:
            try:
                market_cap_data_sort_usd = int(market_cap_cell.data_sort)
                if market_cap_data_sort_usd < 0:
                    raise ValueError
            except ValueError:
                issues.append(
                    ParseIssue(
                        source=source_url,
                        code="INVALID_MARKET_CAP_DATA_SORT",
                        message=(
                            "CompaniesMarketCap market-cap data-sort must be a non-negative "
                            f"integer: {market_cap_cell.data_sort!r}"
                        ),
                        row_number=row_number,
                        raw_record=market_cap_cell.data_sort,
                    )
                )
                market_cap_data_sort_usd = None

        records.append(
            CompaniesMarketCapSecurity(
                rank=rank,
                name=row.company_name,
                ticker=row.company_code.strip().upper(),
                market_cap_usd=market_cap_usd,
                source_url=source_url,
                as_of_date=canonical_as_of,
                market_cap_literal=market_cap_cell.text,
                market_cap_data_sort_usd=market_cap_data_sort_usd,
                source_line_number=row.source_line_number,
            )
        )

    if candidate_rows == 0:
        issues.append(
            ParseIssue(
                source=source_url,
                code="NO_COMPANY_ROWS",
                message="CompaniesMarketCap page contained no company ranking rows",
            )
        )
    return ParseResult(
        records=tuple(records),
        issues=tuple(issues),
        metadata={"as_of_date": canonical_as_of, "candidate_row_count": candidate_rows},
    )


_STOCK_ANALYSIS_DATA_MARKER = re.compile(r"\bcount\s*:\s*([0-9]+)\s*,\s*data\s*:\s*\[")
_JS_STRING_TOKEN = r'"(?:\\(?:u\{[0-9a-fA-F]+\}|u[0-9a-fA-F]{4}|x[0-9a-fA-F]{2}|.)|[^"\\])*"'
_STOCK_ANALYSIS_ROW_PREFIX = re.compile(
    rf"^\{{\s*s\s*:\s*(?P<symbol>{_JS_STRING_TOKEN})\s*,\s*"
    rf"n\s*:\s*(?P<name>{_JS_STRING_TOKEN})\s*,\s*"
    r"marketCap\s*:\s*(?P<market_cap>[^,}]*)",
    re.DOTALL,
)
_STOCK_ANALYSIS_INDUSTRY = re.compile(
    rf"(?:^|,)\s*industry\s*:\s*(?P<industry>{_JS_STRING_TOKEN}|null)(?:,|\}})",
    re.DOTALL,
)


def _find_js_closing_bracket(text: str, opening_index: int) -> int | None:
    depth = 0
    quote: str | None = None
    escaped = False
    for index in range(opening_index, len(text)):
        character = text[index]
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            continue
        if character in {'"', "'", "`"}:
            quote = character
        elif character == "[":
            depth += 1
        elif character == "]":
            depth -= 1
            if depth == 0:
                return index
    return None


def _split_js_array_items(text: str, *, absolute_start: int) -> list[tuple[str, int]]:
    items: list[tuple[str, int]] = []
    start = 0
    brace_depth = 0
    bracket_depth = 0
    parenthesis_depth = 0
    quote: str | None = None
    escaped = False
    for index, character in enumerate(text):
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            continue
        if character in {'"', "'", "`"}:
            quote = character
        elif character == "{":
            brace_depth += 1
        elif character == "}":
            brace_depth -= 1
        elif character == "[":
            bracket_depth += 1
        elif character == "]":
            bracket_depth -= 1
        elif character == "(":
            parenthesis_depth += 1
        elif character == ")":
            parenthesis_depth -= 1
        elif (
            character == "," and brace_depth == 0 and bracket_depth == 0 and parenthesis_depth == 0
        ):
            raw_item = text[start:index]
            stripped = raw_item.strip()
            if stripped:
                leading = len(raw_item) - len(raw_item.lstrip())
                items.append((stripped, absolute_start + start + leading))
            start = index + 1
    raw_item = text[start:]
    stripped = raw_item.strip()
    if stripped:
        leading = len(raw_item) - len(raw_item.lstrip())
        items.append((stripped, absolute_start + start + leading))
    return items


def _decode_js_string_literal(token: str) -> str:
    """Decode the string escape forms used by Svelte's hydration payload."""

    try:
        return str(json.loads(token))
    except json.JSONDecodeError:
        pass

    if len(token) < 2 or token[0] != '"' or token[-1] != '"':
        raise ValueError(f"invalid JavaScript string literal: {token!r}")
    result: list[str] = []
    index = 1
    escapes = {
        '"': '"',
        "'": "'",
        "\\": "\\",
        "/": "/",
        "b": "\b",
        "f": "\f",
        "n": "\n",
        "r": "\r",
        "t": "\t",
        "v": "\v",
        "0": "\0",
    }
    while index < len(token) - 1:
        character = token[index]
        if character != "\\":
            result.append(character)
            index += 1
            continue
        index += 1
        if index >= len(token) - 1:
            raise ValueError(f"unterminated JavaScript escape: {token!r}")
        escape = token[index]
        if escape in escapes:
            result.append(escapes[escape])
            index += 1
        elif escape == "x":
            digits = token[index + 1 : index + 3]
            if len(digits) != 2 or not re.fullmatch(r"[0-9a-fA-F]{2}", digits):
                raise ValueError(f"invalid JavaScript hex escape: {token!r}")
            result.append(chr(int(digits, 16)))
            index += 3
        elif escape == "u" and token[index + 1 : index + 2] == "{":
            closing = token.find("}", index + 2)
            if closing < 0:
                raise ValueError(f"invalid JavaScript Unicode escape: {token!r}")
            digits = token[index + 2 : closing]
            if not re.fullmatch(r"[0-9a-fA-F]+", digits):
                raise ValueError(f"invalid JavaScript Unicode escape: {token!r}")
            result.append(chr(int(digits, 16)))
            index = closing + 1
        elif escape == "u":
            digits = token[index + 1 : index + 5]
            if len(digits) != 4 or not re.fullmatch(r"[0-9a-fA-F]{4}", digits):
                raise ValueError(f"invalid JavaScript Unicode escape: {token!r}")
            result.append(chr(int(digits, 16)))
            index += 5
        elif escape in {"\n", "\r"}:
            # JavaScript line continuation contributes no character.
            index += 1
        else:
            raise ValueError(f"unsupported JavaScript escape \\{escape}: {token!r}")
    return "".join(result)


def _parse_js_market_cap(token: str) -> int | None:
    normalized = token.strip()
    if normalized == "null":
        return None
    if not re.fullmatch(r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?", normalized):
        raise ValueError(f"invalid StockAnalysis marketCap value: {token!r}")
    try:
        value = Decimal(normalized)
    except InvalidOperation as exc:
        raise ValueError(f"invalid StockAnalysis marketCap value: {token!r}") from exc
    integral = value.to_integral_value()
    if not value.is_finite() or value != integral or integral < 0:
        raise ValueError(
            f"StockAnalysis marketCap is not a non-negative whole USD value: {token!r}"
        )
    return int(integral)


def parse_stockanalysis_screener_html(
    html: str,
    *,
    source_url: str,
    as_of_date: str | date,
) -> ParseResult[StockAnalysisSecurity]:
    """Parse the full StockAnalysis screener hydration array without JS execution."""

    canonical_as_of = _canonical_as_of(as_of_date)
    candidate_arrays: list[tuple[int, int, list[tuple[str, int]]]] = []
    for marker in _STOCK_ANALYSIS_DATA_MARKER.finditer(html):
        opening_index = marker.end() - 1
        closing_index = _find_js_closing_bracket(html, opening_index)
        if closing_index is None:
            continue
        items = _split_js_array_items(
            html[opening_index + 1 : closing_index], absolute_start=opening_index + 1
        )
        if items and any(item.lstrip().startswith("{s:") for item, _ in items[:3]):
            candidate_arrays.append((int(marker.group(1)), opening_index, items))

    if not candidate_arrays:
        return ParseResult(
            records=(),
            issues=(
                ParseIssue(
                    source=source_url,
                    code="NO_STOCK_DATA_ARRAY",
                    message="StockAnalysis page contained no screener hydration data array",
                ),
            ),
            metadata={"as_of_date": canonical_as_of},
        )

    # The real screener array is the largest matching hydration array.  Keep an
    # explicit issue if a page unexpectedly contains another plausible array.
    declared_count, _, items = max(candidate_arrays, key=lambda candidate: len(candidate[2]))
    issues: list[ParseIssue] = []
    if len(candidate_arrays) > 1:
        issues.append(
            ParseIssue(
                source=source_url,
                code="MULTIPLE_STOCK_DATA_ARRAYS",
                message=(
                    f"StockAnalysis page contained {len(candidate_arrays)} candidate screener arrays; "
                    "parsed the largest"
                ),
                raw_record=tuple((count, len(rows)) for count, _, rows in candidate_arrays),
            )
        )
    if declared_count != len(items):
        issues.append(
            ParseIssue(
                source=source_url,
                code="DECLARED_COUNT_MISMATCH",
                message=(
                    f"StockAnalysis declared {declared_count} rows but hydration contained "
                    f"{len(items)} items"
                ),
                raw_record={"declared_count": declared_count, "item_count": len(items)},
            )
        )

    records: list[StockAnalysisSecurity] = []
    for row_number, (source_object, absolute_index) in enumerate(items, start=1):
        source_line_number = html.count("\n", 0, absolute_index) + 1
        match = _STOCK_ANALYSIS_ROW_PREFIX.match(source_object)
        if match is None:
            issues.append(
                ParseIssue(
                    source=source_url,
                    code="MALFORMED_STOCK_ROW",
                    message="StockAnalysis hydration row lacks leading s/n/marketCap fields",
                    row_number=row_number,
                    raw_record=source_object,
                )
            )
            continue
        try:
            symbol = _decode_js_string_literal(match.group("symbol")).strip().upper()
            name = _decode_js_string_literal(match.group("name")).strip()
        except ValueError as exc:
            issues.append(
                ParseIssue(
                    source=source_url,
                    code="INVALID_JS_STRING",
                    message=str(exc),
                    row_number=row_number,
                    raw_record=source_object,
                )
            )
            continue
        if not symbol or not name:
            issues.append(
                ParseIssue(
                    source=source_url,
                    code="MISSING_REQUIRED_VALUE",
                    message="StockAnalysis hydration row has an empty symbol or name",
                    row_number=row_number,
                    raw_record=source_object,
                )
            )
            continue
        try:
            market_cap_usd = _parse_js_market_cap(match.group("market_cap"))
        except ValueError as exc:
            issues.append(
                ParseIssue(
                    source=source_url,
                    code="INVALID_MARKET_CAP_VALUE",
                    message=str(exc),
                    row_number=row_number,
                    raw_record=source_object,
                )
            )
            market_cap_usd = None
        if match.group("market_cap").strip() == "null":
            issues.append(
                ParseIssue(
                    source=source_url,
                    code="MISSING_MARKET_CAP",
                    message="StockAnalysis hydration row has marketCap:null",
                    row_number=row_number,
                    raw_record=source_object,
                )
            )
        industry: str | None = None
        industry_match = _STOCK_ANALYSIS_INDUSTRY.search(source_object)
        if industry_match is not None and industry_match.group("industry") != "null":
            try:
                industry = (
                    _decode_js_string_literal(industry_match.group("industry")).strip() or None
                )
            except ValueError as exc:
                issues.append(
                    ParseIssue(
                        source=source_url,
                        code="INVALID_INDUSTRY_JS_STRING",
                        message=str(exc),
                        row_number=row_number,
                        raw_record=source_object,
                    )
                )
        records.append(
            StockAnalysisSecurity(
                symbol=symbol,
                name=name,
                market_cap_usd=market_cap_usd,
                source_url=source_url,
                as_of_date=canonical_as_of,
                source_row_number=row_number,
                source_line_number=source_line_number,
                source_object=source_object,
                industry=industry,
            )
        )

    return ParseResult(
        records=tuple(records),
        issues=tuple(issues),
        metadata={
            "as_of_date": canonical_as_of,
            "declared_count": declared_count,
            "source_row_count": len(items),
        },
    )


def sha256_file_manifest(path: str | Path, *, source_id: str | None = None) -> SourceFileManifest:
    """Read one source snapshot and return its immutable SHA-256 manifest row."""

    source_path = Path(path)
    hasher = hashlib.sha256()
    size_bytes = 0
    with source_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 * 1024), b""):
            hasher.update(chunk)
            size_bytes += len(chunk)
    return SourceFileManifest(
        source_id=source_id or source_path.name,
        path=str(source_path),
        size_bytes=size_bytes,
        sha256=hasher.hexdigest(),
    )


def build_sha256_file_manifest(
    sources: Mapping[str, str | Path],
) -> tuple[SourceFileManifest, ...]:
    """Build manifest rows in stable logical-source order."""

    return tuple(
        sha256_file_manifest(path, source_id=source_id)
        for source_id, path in sorted(sources.items())
    )


def deterministic_source_fingerprint(manifest: Sequence[SourceFileManifest]) -> str:
    """Fingerprint logical source IDs and contents, independent of file paths/order."""

    canonical_rows = [
        {
            "sha256": row.sha256,
            "size_bytes": row.size_bytes,
            "source_id": row.source_id,
        }
        for row in sorted(manifest, key=lambda item: item.source_id)
    ]
    source_ids = [row["source_id"] for row in canonical_rows]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("source manifest contains duplicate source_id values")
    canonical_json = json.dumps(canonical_rows, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
