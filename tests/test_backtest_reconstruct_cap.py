from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass
class _Snap:
    ticker: str
    as_of_date: str
    price: float
    raw_price: float


class _FakeProvider:
    def get_price_asof(self, ticker, as_of_date):
        return _Snap(ticker, as_of_date, 50.0, 50.0)


def _writer_record(method: str, outputs: dict[str, Any]) -> dict[str, Any]:
    from app.valuation.lineage import valuation_source_record

    record = valuation_source_record(
        {
            "ticker": "AAA",
            "as_of_date": "2023-10-04",
            "method": method,
            "inputs_json": "{}",
            "outputs_json": json.dumps(outputs, sort_keys=True),
            "warnings_json": "[]",
            "created_at": "2024-01-02T00:00:00+00:00",
            "valuation_writer_version": "test",
            "quality_gate_verdict": None,
            "confidence_class": None,
            "gate_reason_codes": None,
            "valuation_headwinds": None,
            "valuation_supports": None,
            "source_run_id": "backtest_recon_2024-01-02",
        }
    )
    assert record is not None
    return record


def test_reconstruct_sets_cap_category(monkeypatch):
    from app.backtest import reconstruct

    records = [
        _writer_record(
            "scorecard",
            {
                "pricing_zone": "MARGIN_OF_SAFETY",
                "pricing_zone_detail": {"dcf_base": 100.0, "epv_adjusted": 90.0},
            },
        ),
        _writer_record(
            "reverse_dcf",
            {"expectations_gap": {"bucket": "CHEAP_VS_EXPECTATIONS"}},
        ),
    ]
    ensure_call = {}

    def _fake_ensure(*args, **kwargs):
        ensure_call["args"] = args
        ensure_call["kwargs"] = kwargs
        return records

    monkeypatch.setattr(reconstruct, "ensure_valuation", _fake_ensure)
    monkeypatch.setattr(reconstruct, "_live_sector_model_routed", lambda *args: False)

    def _forbid_db_read():
        raise AssertionError("reconstruction must not reread mutable valuation rows")

    monkeypatch.setattr("app.db.get_db", _forbid_db_read)
    # Force a large-cap classification regardless of shares data; capture the cutoff passed in.
    captured = {}

    def _fake_cap(ticker, cutoff, price):
        captured["cutoff"] = cutoff
        return "large_cap"

    monkeypatch.setattr(reconstruct, "cap_category_asof", _fake_cap)

    provider = _FakeProvider()
    result = reconstruct.reconstruct_signal_asof("AAA", "2024-01-02", provider=provider)
    assert result.signal is not None
    assert result.signal.cap_category == "large_cap"
    assert result.signal.expectations_gap_bucket == "CHEAP_VS_EXPECTATIONS"
    assert captured["cutoff"] == "2023-10-04"
    assert ensure_call["args"] == ("AAA", "2023-10-04")
    assert ensure_call["kwargs"] == {
        "provider": provider,
        "run_id": "backtest_recon_2024-01-02",
        "price_override": 50.0,
        "force_refresh": True,
        "require_filed_asof": True,
        "raise_on_error": True,
    }
