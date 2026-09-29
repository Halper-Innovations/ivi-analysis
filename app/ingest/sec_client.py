from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any
from urllib.parse import urlparse

from app.config import get_config
from app.util.http import HttpClient


@dataclass
class FilingStub:
    cik: str
    accession: str
    accession_nodash: str
    form_type: str
    filing_date: date
    period_end: str | None
    primary_document: str
    primary_doc_url: str
    filing_index_url: str


class SecClient:
    def __init__(self) -> None:
        self.cfg = get_config()
        self.http = HttpClient(self.cfg)

    def submissions(self, cik: str) -> dict[str, Any]:
        cik10 = cik.zfill(10)
        url = f"https://data.sec.gov/submissions/CIK{cik10}.json"
        return self.http.get_json(url, use_cache=True, cache_ttl_seconds=3600)

    def records_from_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            return {}
        recent = payload.get("filings", {}).get("recent", {})
        if isinstance(recent, dict) and recent.get("accessionNumber"):
            return recent
        if payload.get("accessionNumber"):
            return payload
        return {}

    def filing_stubs_from_records(
        self,
        *,
        cik: str,
        records: dict[str, Any],
        forms: list[str],
        start_date: date,
        end_date: date,
    ) -> list[FilingStub]:
        accessions = records.get("accessionNumber", [])
        form_types = records.get("form", [])
        filing_dates = records.get("filingDate", [])
        period_ends = records.get("reportDate", [])
        primary_docs = records.get("primaryDocument", [])
        out: list[FilingStub] = []
        for idx, accession in enumerate(accessions):
            form = str(form_types[idx]).upper()
            if forms and form not in forms:
                continue
            try:
                filed = date.fromisoformat(filing_dates[idx])
            except Exception:
                continue
            if filed < start_date or filed > end_date:
                continue
            accession_no_dash = str(accession).replace("-", "")
            primary_doc = str(primary_docs[idx])
            url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_no_dash}/{primary_doc}"
            index_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_no_dash}/index.json"
            out.append(
                FilingStub(
                    cik=str(int(cik)),
                    accession=str(accession),
                    accession_nodash=accession_no_dash,
                    form_type=form,
                    filing_date=filed,
                    period_end=period_ends[idx] if idx < len(period_ends) else None,
                    primary_document=primary_doc,
                    primary_doc_url=url,
                    filing_index_url=index_url,
                )
            )
        return out

    @staticmethod
    def dedupe_and_sort_stubs(stubs: list[FilingStub]) -> list[FilingStub]:
        deduped: dict[str, FilingStub] = {}
        for stub in stubs:
            existing = deduped.get(stub.accession)
            if not existing:
                deduped[stub.accession] = stub
                continue
            if stub.filing_date > existing.filing_date:
                deduped[stub.accession] = stub
                continue
            if stub.filing_date == existing.filing_date and stub.form_type > existing.form_type:
                deduped[stub.accession] = stub
        return sorted(deduped.values(), key=lambda row: (row.filing_date, row.accession), reverse=True)

    def cached_submissions_payloads(self, cik: str) -> list[dict[str, Any]]:
        cik10 = cik.zfill(10)
        root_url = f"https://data.sec.gov/submissions/CIK{cik10}.json"
        root_payload = self.http.read_cached_json(root_url)
        if not root_payload:
            return []
        payloads = [root_payload]
        files = root_payload.get("filings", {}).get("files", [])
        for entry in files:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name") or "").strip()
            if not name:
                continue
            payload = self.http.read_cached_json(f"https://data.sec.gov/submissions/{name}")
            if payload:
                payloads.append(payload)
        return payloads

    def list_cached_filings_window(
        self,
        cik: str,
        *,
        start_date: date,
        end_date: date,
        forms: list[str],
    ) -> list[FilingStub]:
        all_stubs: list[FilingStub] = []
        for payload in self.cached_submissions_payloads(cik):
            records = self.records_from_payload(payload)
            if not records:
                continue
            all_stubs.extend(
                self.filing_stubs_from_records(
                    cik=cik,
                    records=records,
                    forms=forms,
                    start_date=start_date,
                    end_date=end_date,
                )
            )
        return self.dedupe_and_sort_stubs(all_stubs)

    def list_recent_filings(self, cik: str, since: date, forms: list[str]) -> list[FilingStub]:
        payload = self.submissions(cik)
        records = self.records_from_payload(payload)
        stubs = self.filing_stubs_from_records(
            cik=cik,
            records=records,
            forms=forms,
            start_date=since,
            end_date=date.max,
        )
        return self.dedupe_and_sort_stubs(stubs)

    def list_filings_window(self, cik: str, *, start_date: date, end_date: date, forms: list[str]) -> list[FilingStub]:
        payload = self.submissions(cik)
        all_stubs: list[FilingStub] = []

        records = self.records_from_payload(payload)
        all_stubs.extend(
            self.filing_stubs_from_records(
                cik=cik,
                records=records,
                forms=forms,
                start_date=start_date,
                end_date=end_date,
            )
        )

        files = payload.get("filings", {}).get("files", [])
        for entry in files:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name") or "").strip()
            if not name:
                continue
            try:
                file_from = date.fromisoformat(str(entry.get("filingFrom")))
                file_to = date.fromisoformat(str(entry.get("filingTo")))
            except Exception:
                file_from = date.min
                file_to = date.max
            if file_to < start_date or file_from > end_date:
                continue
            url = f"https://data.sec.gov/submissions/{name}"
            file_payload = self.http.get_json(url, use_cache=True, cache_ttl_seconds=3600)
            file_records = self.records_from_payload(file_payload)
            all_stubs.extend(
                self.filing_stubs_from_records(
                    cik=cik,
                    records=file_records,
                    forms=forms,
                    start_date=start_date,
                    end_date=end_date,
                )
            )

        return self.dedupe_and_sort_stubs(all_stubs)

    # Backward-compatible aliases while call sites migrate to the public helpers.
    def _records_from_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.records_from_payload(payload)

    def _stubs_from_records(
        self,
        *,
        cik: str,
        records: dict[str, Any],
        forms: list[str],
        start_date: date,
        end_date: date,
    ) -> list[FilingStub]:
        return self.filing_stubs_from_records(
            cik=cik,
            records=records,
            forms=forms,
            start_date=start_date,
            end_date=end_date,
        )

    @staticmethod
    def _dedupe_and_sort(stubs: list[FilingStub]) -> list[FilingStub]:
        return SecClient.dedupe_and_sort_stubs(stubs)

    def download_bytes(
        self, url: str, *, use_cache: bool = True, max_bytes: int | None = None
    ) -> bytes:
        return self.http.get_bytes(
            url, use_cache=use_cache, cache_ttl_seconds=None, max_bytes=max_bytes
        )

    @staticmethod
    def filename_from_url(url: str) -> str:
        path = urlparse(url).path
        name = path.rsplit("/", 1)[-1]
        return name or "document.txt"
