from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from app.universe.us_equity_census_sources import (
    CompaniesMarketCapSecurity,
    SecExchangeSecurity,
    StockAnalysisSecurity,
    build_sha256_file_manifest,
    deterministic_source_fingerprint,
    normalize_nasdaq_trader_exchange,
    parse_companiesmarketcap_usa_html,
    parse_market_cap_usd_literal,
    parse_nasdaqlisted,
    parse_otherlisted,
    parse_sec_company_tickers_exchange,
    parse_stockanalysis_screener_html,
    sha256_file_manifest,
)


def test_sec_exchange_parser_preserves_every_security_row_and_raw_values() -> None:
    result = parse_sec_company_tickers_exchange(
        {
            "fields": ["cik", "name", "ticker", "exchange"],
            "data": [
                [1001, "Issuer One", "one.a", "NYSE"],
                [1001, "Issuer One", "one.b", "NYSE"],
                [1002, "Off Exchange", "OFFX", None],
            ],
        },
        source_url="https://sec.example/exchange.json",
    )

    assert result.issues == ()
    assert result.metadata == {
        "fields": ("cik", "name", "ticker", "exchange"),
        "source_row_count": 3,
    }
    assert result.records == (
        SecExchangeSecurity(
            cik="1001",
            name="Issuer One",
            ticker="ONE.A",
            exchange="NYSE",
            source_url="https://sec.example/exchange.json",
            source_row_number=1,
            source_fields=("cik", "name", "ticker", "exchange"),
            source_row=(1001, "Issuer One", "one.a", "NYSE"),
        ),
        SecExchangeSecurity(
            cik="1001",
            name="Issuer One",
            ticker="ONE.B",
            exchange="NYSE",
            source_url="https://sec.example/exchange.json",
            source_row_number=2,
            source_fields=("cik", "name", "ticker", "exchange"),
            source_row=(1001, "Issuer One", "one.b", "NYSE"),
        ),
        SecExchangeSecurity(
            cik="1002",
            name="Off Exchange",
            ticker="OFFX",
            exchange=None,
            source_url="https://sec.example/exchange.json",
            source_row_number=3,
            source_fields=("cik", "name", "ticker", "exchange"),
            source_row=(1002, "Off Exchange", "OFFX", None),
        ),
    )


def test_sec_exchange_parser_surfaces_each_malformed_row() -> None:
    result = parse_sec_company_tickers_exchange(
        '{"fields":["cik","name","ticker","exchange"],'
        '"data":[[1001,"Good","GOOD","Nasdaq"],{"cik":2},'
        '[1003,"Short","MISS"],[1004,"No Ticker","","NYSE"]]}'
    )

    assert len(result.records) == 1
    assert [issue.code for issue in result.issues] == [
        "INVALID_ROW_TYPE",
        "ROW_FIELD_COUNT_MISMATCH",
        "MISSING_REQUIRED_VALUE",
    ]
    assert [issue.row_number for issue in result.issues] == [2, 3, 4]
    assert result.issues[1].raw_record == (1003, "Short", "MISS")
    assert result.issues[2].message == "SEC row is missing values: ticker"


def test_sec_exchange_parser_returns_issue_for_invalid_json() -> None:
    result = parse_sec_company_tickers_exchange("not-json")

    assert result.records == ()
    assert len(result.issues) == 1
    assert result.issues[0].code == "INVALID_JSON"
    assert result.issues[0].row_number is None


def test_nasdaqlisted_parser_preserves_listing_metadata_and_creation_time() -> None:
    text = (
        "Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares\n"
        "ABCD|Example Common Stock|Q|N|N|100|N|N\n"
        "EFGX|Example Market ETF|S|N|D|50|Y|Y\n"
        "File Creation Time: 0717202621:31|||||||\n"
    )
    result = parse_nasdaqlisted(text)

    assert result.issues == ()
    assert result.metadata == {
        "fields": (
            "Symbol",
            "Security Name",
            "Market Category",
            "Test Issue",
            "Financial Status",
            "Round Lot Size",
            "ETF",
            "NextShares",
        ),
        "file_creation_timestamp": "2026-07-17T21:31:00",
        "file_creation_timestamp_raw": "0717202621:31",
        "source_row_count": 2,
    }
    assert result.records[0].symbol == "ABCD"
    assert result.records[0].exchange == "Nasdaq"
    assert result.records[0].exchange_code == "Q"
    assert result.records[0].market_tier == "NASDAQ_GLOBAL_SELECT_MARKET"
    assert result.records[0].etf is False
    assert result.records[0].test_issue is False
    assert result.records[0].round_lot_size == 100
    assert result.records[0].next_shares is False
    assert result.records[0].file_creation_timestamp == "2026-07-17T21:31:00"
    assert result.records[0].source_line_number == 2
    assert result.records[0].source_row == (
        "ABCD",
        "Example Common Stock",
        "Q",
        "N",
        "N",
        "100",
        "N",
        "N",
    )
    assert result.records[1].market_tier == "NASDAQ_CAPITAL_MARKET"
    assert result.records[1].etf is True
    assert result.records[1].next_shares is True


def test_otherlisted_parser_normalizes_documented_exchange_codes() -> None:
    text = (
        "ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol\n"
        "AAA|NYSE Common Stock|N|AAA|N|100|N|AAA\n"
        "BBB|American Common Stock|A|BBB|N|100|N|BBB\n"
        "CCC|Arca ETF|P|CCC|Y|100|N|CCC\n"
        "DDD|Texas Test Stock|M|DDD|N|100|Y|DDD\n"
        "File Creation Time: 0717202621:31||||||\n"
    )
    result = parse_otherlisted(text)

    assert result.issues == ()
    assert [record.exchange for record in result.records] == [
        "NYSE",
        "NYSE American",
        "NYSE Arca",
        "NYSE Texas",
    ]
    assert result.records[0].cqs_symbol == "AAA"
    assert result.records[0].nasdaq_symbol == "AAA"
    assert result.records[2].etf is True
    assert result.records[3].test_issue is True
    assert normalize_nasdaq_trader_exchange("Q") == "Nasdaq"
    assert normalize_nasdaq_trader_exchange("g") == "Nasdaq"
    assert normalize_nasdaq_trader_exchange("?") is None


def test_nasdaq_parser_surfaces_bad_rows_flags_and_codes_without_dropping_record() -> None:
    text = (
        "Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares\n"
        "BAD|Bad Flag Equity|X|maybe|N|lots|?|N\n"
        "SHORT|Too Few|Q|N\n"
        "File Creation Time: invalid|||||||\n"
    )
    result = parse_nasdaqlisted(text)

    assert len(result.records) == 1
    assert result.records[0].symbol == "BAD"
    assert result.records[0].exchange is None
    assert result.records[0].test_issue is None
    assert result.records[0].round_lot_size is None
    assert result.records[0].etf is None
    assert [issue.code for issue in result.issues] == [
        "INVALID_FILE_CREATION_TIME",
        "UNKNOWN_EXCHANGE_CODE",
        "INVALID_YES_NO_FLAG",
        "INVALID_YES_NO_FLAG",
        "INVALID_ROUND_LOT_SIZE",
        "ROW_FIELD_COUNT_MISMATCH",
    ]
    assert result.issues[-1].row_number == 3


@pytest.mark.parametrize(
    ("literal", "expected"),
    [
        ("$4.912 T", 4_912_000_000_000),
        ("958.79 B", 958_790_000_000),
        ("US$ 12.5 million", 12_500_000),
        ("750 K", 750_000),
        ("$1,234", 1_234),
    ],
)
def test_market_cap_literal_parsing_is_exact(literal: str, expected: int) -> None:
    assert parse_market_cap_usd_literal(literal) == expected


def test_market_cap_literal_rejects_ambiguous_or_fractional_usd() -> None:
    with pytest.raises(ValueError, match="invalid USD market-cap literal"):
        parse_market_cap_usd_literal("about $10 B")
    with pytest.raises(ValueError, match="not a non-negative whole USD"):
        parse_market_cap_usd_literal("$1.2345 K")


def test_companiesmarketcap_parser_extracts_rows_and_preserves_literal_sort_value() -> None:
    html = """
    <table>
      <tr><td class="rank-td td-right" data-sort="1">1</td>
        <td class="name-td"><div class="company-name">Alpha &amp; Sons</div>
          <div class="company-code"><span></span>ALP.A</div></td>
        <td class="td-right" data-sort="4912260841472"><span>$</span>4.912 T</td></tr>
      <tr><td class="rank-td td-right" data-sort="2">2</td>
        <td class="name-td"><div class="company-name">Beta</div>
          <div class="company-code">BETA</div></td>
        <td class="td-right" data-sort="958798299136">958.79 B</td></tr>
    </table>
    """
    result = parse_companiesmarketcap_usa_html(
        html,
        source_url="https://companies.example/usa?page=1",
        as_of_date="2026-07-17",
    )

    assert result.issues == ()
    assert result.metadata == {"as_of_date": "2026-07-17", "candidate_row_count": 2}
    assert result.records == (
        CompaniesMarketCapSecurity(
            rank=1,
            name="Alpha & Sons",
            ticker="ALP.A",
            market_cap_usd=4_912_000_000_000,
            source_url="https://companies.example/usa?page=1",
            as_of_date="2026-07-17",
            market_cap_literal="$4.912 T",
            market_cap_data_sort_usd=4_912_260_841_472,
            source_line_number=3,
        ),
        CompaniesMarketCapSecurity(
            rank=2,
            name="Beta",
            ticker="BETA",
            market_cap_usd=958_790_000_000,
            source_url="https://companies.example/usa?page=1",
            as_of_date="2026-07-17",
            market_cap_literal="958.79 B",
            market_cap_data_sort_usd=958_798_299_136,
            source_line_number=7,
        ),
    )


def test_companiesmarketcap_parser_accounts_for_malformed_candidate_rows() -> None:
    html = """
    <tr><td class="rank-td">not-a-rank</td><td class="name-td">
      <div class="company-name">Bad Rank</div><div class="company-code">BAD</div></td>
      <td class="td-right" data-sort="10000000000">10 B</td></tr>
    <tr><td class="rank-td">2</td><td class="name-td">
      <div class="company-name">No Symbol</div></td><td class="td-right">9 B</td></tr>
    <tr><td class="rank-td">3</td><td class="name-td">
      <div class="company-name">Bad Cap</div><div class="company-code">BCAP</div></td>
      <td class="td-right">unknown</td></tr>
    """
    result = parse_companiesmarketcap_usa_html(
        html, source_url="https://companies.example/page", as_of_date="2026-07-17"
    )

    assert result.records == ()
    assert [issue.code for issue in result.issues] == [
        "INVALID_RANK",
        "MALFORMED_COMPANY_ROW",
        "INVALID_MARKET_CAP_LITERAL",
    ]
    assert [issue.row_number for issue in result.issues] == [1, 2, 3]


def test_stockanalysis_parser_extracts_hydration_rows_and_decodes_js_names() -> None:
    html = r'''
    <script>data:[{type:"ignore"}],other:{count:3,data:[
      {s:"ALP.A",n:"Alpha \"Quoted\" & Co.",marketCap:4912164720882,price:1},
      {s:"BETA",n:"Beta\x20Holdings",marketCap:1.25e10,price:2},
      {s:"NOCAP",n:"No Cap",marketCap:null,price:3}
    ]}</script>
    '''
    result = parse_stockanalysis_screener_html(
        html, source_url="https://stock.example/screener", as_of_date="2026-07-17"
    )

    assert result.metadata == {
        "as_of_date": "2026-07-17",
        "declared_count": 3,
        "source_row_count": 3,
    }
    assert [issue.code for issue in result.issues] == ["MISSING_MARKET_CAP"]
    assert result.issues[0].row_number == 3
    assert result.records == (
        StockAnalysisSecurity(
            symbol="ALP.A",
            name='Alpha "Quoted" & Co.',
            market_cap_usd=4_912_164_720_882,
            source_url="https://stock.example/screener",
            as_of_date="2026-07-17",
            source_row_number=1,
            source_line_number=3,
            source_object=r'{s:"ALP.A",n:"Alpha \"Quoted\" & Co.",marketCap:4912164720882,price:1}',
        ),
        StockAnalysisSecurity(
            symbol="BETA",
            name="Beta Holdings",
            market_cap_usd=12_500_000_000,
            source_url="https://stock.example/screener",
            as_of_date="2026-07-17",
            source_row_number=2,
            source_line_number=4,
            source_object=r'{s:"BETA",n:"Beta\x20Holdings",marketCap:1.25e10,price:2}',
        ),
        StockAnalysisSecurity(
            symbol="NOCAP",
            name="No Cap",
            market_cap_usd=None,
            source_url="https://stock.example/screener",
            as_of_date="2026-07-17",
            source_row_number=3,
            source_line_number=5,
            source_object=r'{s:"NOCAP",n:"No Cap",marketCap:null,price:3}',
        ),
    )


def test_stockanalysis_parser_surfaces_malformed_items_and_count_mismatch() -> None:
    html = '<script>x={count:3,data:[{s:"GOOD",n:"Good",marketCap:10},{s:"BAD",marketCap:9}]}</script>'
    result = parse_stockanalysis_screener_html(
        html, source_url="https://stock.example/screener", as_of_date="2026-07-17"
    )

    assert len(result.records) == 1
    assert result.records[0].symbol == "GOOD"
    assert [issue.code for issue in result.issues] == [
        "DECLARED_COUNT_MISMATCH",
        "MALFORMED_STOCK_ROW",
    ]
    assert result.issues[0].raw_record == {"declared_count": 3, "item_count": 2}
    assert result.issues[1].row_number == 2


def test_stockanalysis_parser_reports_missing_hydration_array() -> None:
    result = parse_stockanalysis_screener_html(
        "<html></html>", source_url="https://stock.example/screener", as_of_date="2026-07-17"
    )

    assert result.records == ()
    assert [issue.code for issue in result.issues] == ["NO_STOCK_DATA_ARRAY"]


def test_file_manifests_and_fingerprint_are_exact_and_order_independent(tmp_path: Path) -> None:
    alpha = tmp_path / "alpha.txt"
    beta = tmp_path / "beta.txt"
    alpha.write_bytes(b"alpha\n")
    beta.write_bytes(b"beta\n")

    alpha_manifest = sha256_file_manifest(alpha, source_id="sec")
    assert alpha_manifest.source_id == "sec"
    assert alpha_manifest.path == str(alpha)
    assert alpha_manifest.size_bytes == 6
    assert alpha_manifest.sha256 == "b6a98d9ce9a2d9149288fa3df42d377c3e42737afdcdaf714e33c0a100b51060"

    manifest = build_sha256_file_manifest({"stock": beta, "sec": alpha})
    assert [row.source_id for row in manifest] == ["sec", "stock"]
    assert manifest[1].size_bytes == 5
    assert manifest[1].sha256 == "f2c82decdd7181cf98945929a62598db7e6b477e11f6e0eb0ae97020eff151ad"
    assert deterministic_source_fingerprint(manifest) == (
        "5ece55acaf0736756e590c7dc0b5d102318d738c5c9da4b9e7899ccce379c3c7"
    )
    assert deterministic_source_fingerprint(tuple(reversed(manifest))) == (
        "5ece55acaf0736756e590c7dc0b5d102318d738c5c9da4b9e7899ccce379c3c7"
    )


def test_fingerprint_rejects_duplicate_source_ids(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_bytes(b"source")
    row = sha256_file_manifest(source, source_id="same")

    with pytest.raises(ValueError, match="duplicate source_id"):
        deterministic_source_fingerprint((row, replace(row, path="copy.txt")))
