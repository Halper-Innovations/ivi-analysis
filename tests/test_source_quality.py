from app.research.source_quality import classify_source_quality, source_quality_label


def test_classify_source_quality_marks_external_news_as_uncalibrated_secondary():
    quality = classify_source_quality(
        source_type="external_news",
        source_url="https://reputable.example.com/story",
        published_at="2026-04-17T13:00:00+00:00",
        as_of_date="2026-04-18",
    )

    assert quality["source_family"] == "external_news"
    assert quality["source_origin"] == "secondary"
    assert quality["source_independence"] == "independent_or_third_party"
    assert quality["source_domain"] == "reputable.example.com"
    assert quality["freshness_days"] == 1
    assert quality["freshness_bucket"] == "recent_7d"
    assert quality["source_quality_score"] == 0.71
    assert quality["calibration_status"] == "heuristic_unvalidated"
    assert quality["reason_codes"] == [
        "SOURCE_EXTERNAL_SECONDARY",
        "SECONDARY_SOURCE_CALIBRATION_PENDING",
        "FRESHNESS_RECENT_7D",
    ]


def test_classify_source_quality_marks_company_transcript_as_management_source():
    quality = classify_source_quality(
        source_type="TRANSCRIPT",
        source_url="https://www.alphavantage.co/query?function=EARNINGS_CALL_TRANSCRIPT&symbol=AAA&quarter=2026Q1",
        published_at="2026-03-31",
        as_of_date="2026-04-18",
    )

    assert quality["source_family"] == "management_transcript"
    assert quality["source_origin"] == "primary_company_controlled"
    assert quality["source_independence"] == "company_controlled"
    assert quality["freshness_days"] == 18
    assert quality["freshness_bucket"] == "current_30d"
    assert quality["source_quality_score"] == 0.88
    assert quality["calibration_status"] == "deterministic_heuristic"
    assert quality["reason_codes"] == [
        "SOURCE_MANAGEMENT_TRANSCRIPT",
        "SOURCE_ISSUER_BIAS_POSSIBLE",
        "FRESHNESS_CURRENT_30D",
    ]


def test_classify_source_quality_applies_domain_reputation_history(tmp_path):
    reputation_path = tmp_path / "source_reputation_history.csv"
    reputation_path.write_text(
        "domain,as_of_date,reputation_score,sample_size,reason_codes\n"
        "reputable.example.com,2026-03-01,0.40,3,OLD_RECORD\n"
        "reputable.example.com,2026-04-01,0.92,12,THIRD_PARTY_VERIFIED\n",
        encoding="utf-8",
    )

    quality = classify_source_quality(
        source_type="external_news",
        source_url="https://reputable.example.com/story",
        published_at="2026-04-17T13:00:00+00:00",
        as_of_date="2026-04-18",
        source_reputation_path=reputation_path,
    )

    assert quality["source_family"] == "external_news"
    assert quality["source_domain"] == "reputable.example.com"
    assert quality["source_quality_score"] == 0.82
    assert quality["calibration_status"] == "domain_reputation_calibrated"
    assert quality["source_reputation_status"] == "known"
    assert quality["source_reputation_score"] == 0.92
    assert quality["source_reputation_as_of_date"] == "2026-04-01"
    assert quality["source_reputation_sample_size"] == 12
    assert quality["reason_codes"] == [
        "SOURCE_EXTERNAL_SECONDARY",
        "FRESHNESS_RECENT_7D",
        "SOURCE_REPUTATION_HISTORY",
        "SOURCE_REPUTATION_HIGH",
        "THIRD_PARTY_VERIFIED",
    ]


def test_classify_source_quality_marks_missing_reputation_when_registry_exists(tmp_path):
    reputation_path = tmp_path / "source_reputation_history.csv"
    reputation_path.write_text(
        "domain,as_of_date,reputation_score,sample_size\n"
        "other.example.com,2026-04-01,0.90,15\n",
        encoding="utf-8",
    )

    quality = classify_source_quality(
        source_type="external_news",
        source_url="https://unscored.example.com/story",
        published_at="2026-04-17T13:00:00+00:00",
        as_of_date="2026-04-18",
        source_reputation_path=reputation_path,
    )

    assert quality["source_quality_score"] == 0.71
    assert quality["calibration_status"] == "heuristic_unvalidated"
    assert quality["source_reputation_status"] == "missing"
    assert quality["reason_codes"] == [
        "SOURCE_EXTERNAL_SECONDARY",
        "SECONDARY_SOURCE_CALIBRATION_PENDING",
        "FRESHNESS_RECENT_7D",
        "SOURCE_REPUTATION_MISSING",
    ]


def test_source_quality_label_is_stable_and_compact():
    label = source_quality_label(
        {
            "source_family": "external_news",
            "freshness_bucket": "recent_7d",
            "source_quality_score": 0.71,
            "calibration_status": "heuristic_unvalidated",
        }
    )

    assert label == "external_news; freshness=recent_7d; score=0.71; calibration=heuristic_unvalidated"


def test_source_quality_label_includes_reputation_when_present():
    label = source_quality_label(
        {
            "source_family": "external_news",
            "freshness_bucket": "recent_7d",
            "source_quality_score": 0.82,
            "calibration_status": "domain_reputation_calibrated",
            "source_reputation_status": "known",
            "source_reputation_score": 0.92,
        }
    )

    assert (
        label
        == "external_news; freshness=recent_7d; score=0.82; "
        "calibration=domain_reputation_calibrated; reputation=known:0.92"
    )
