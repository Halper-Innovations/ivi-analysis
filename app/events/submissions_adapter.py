"""Default reader adapters over SecClient submissions JSON.

Implements the detector protocols (SubmissionsReader,
RegistrationHistoryReader, ReportingHistoryReader) with the binding
`filings.files` fallback: `filings.recent` caps at exactly 1,000 filings, so
a heavy filer's 2024 accessions roll off `recent` by 2026 — history lookups
that stop at `recent` would silently miss Item 1.03 anchors or prior
periodic filings. Page payloads carry the same parallel arrays (incl.
`items`).
"""

from __future__ import annotations

from typing import Any


class SecSubmissionsAdapter:
    def __init__(self, client) -> None:
        self.client = client
        self._roots: dict[str, dict[str, Any] | None] = {}
        self._pages: dict[str, list[dict[str, Any]]] = {}

    def _root(self, cik: str) -> dict[str, Any] | None:
        if cik not in self._roots:
            try:
                self._roots[cik] = self.client.submissions(cik)
            except Exception:
                self._roots[cik] = None
        return self._roots[cik]

    def _page_records(self, cik: str) -> list[dict[str, Any]]:
        """Paginated filings.files records, fetched lazily and memoized per run."""
        if cik in self._pages:
            return self._pages[cik]
        records: list[dict[str, Any]] = []
        root = self._root(cik)
        if root is not None:
            for entry in root.get("filings", {}).get("files", []):
                if not isinstance(entry, dict):
                    continue
                name = str(entry.get("name") or "").strip()
                if not name:
                    continue
                try:
                    payload = self.client.http.get_json(
                        f"https://data.sec.gov/submissions/{name}",
                        use_cache=True,
                        cache_ttl_seconds=3600,
                    )
                except Exception:
                    continue
                page = self.client.records_from_payload(payload)
                if page:
                    records.append(page)
        self._pages[cik] = records
        return records

    def _all_records(self, cik: str) -> list[dict[str, Any]] | None:
        root = self._root(cik)
        if root is None:
            return None
        records = []
        recent = self.client.records_from_payload(root)
        if recent:
            records.append(recent)
        records.extend(self._page_records(cik))
        return records

    @staticmethod
    def _column(records: dict[str, Any], key: str) -> list:
        value = records.get(key)
        return value if isinstance(value, list) else []

    # -- SubmissionsReader -------------------------------------------------

    def items_for(self, cik: str, accession: str) -> str | None:
        root = self._root(cik)
        if root is None:
            return None
        recent = self.client.records_from_payload(root)
        record_sets = [recent] if recent else []
        searched_pages = False
        while True:
            for records in record_sets:
                accessions = self._column(records, "accessionNumber")
                items = self._column(records, "items")
                for idx, acc in enumerate(accessions):
                    if str(acc) == accession:
                        return str(items[idx]) if idx < len(items) else ""
            if searched_pages:
                # Present in the index but absent from submissions (timing
                # lag): treat as no flagged items rather than a fetch failure.
                return ""
            record_sets = self._page_records(cik)
            searched_pages = True

    def prior_item103(self, cik: str, *, before: str) -> tuple[str, str] | None:
        return self._latest_matching(
            cik, before=before,
            match=lambda form, items: form == "8-K" and "1.03" in {
                i.strip() for i in items.split(",") if i.strip()
            },
        )

    # -- RegistrationHistoryReader ------------------------------------------

    def prior_registration(self, cik: str, *, before: str) -> tuple[str, str] | None:
        return self._latest_matching(
            cik, before=before,
            match=lambda form, items: form in {"10-12B", "10-12B/A", "10-12G", "10-12G/A"},
        )

    # -- ReportingHistoryReader ----------------------------------------------

    def has_prior_periodic_filings(self, cik: str, *, before: str) -> bool | None:
        all_records = self._all_records(cik)
        if all_records is None:
            return None
        for records in all_records:
            forms = self._column(records, "form")
            dates = self._column(records, "filingDate")
            for idx, form in enumerate(forms):
                form_str = str(form).upper()
                if not (form_str.startswith("10-K") or form_str.startswith("10-Q")):
                    continue
                if idx < len(dates) and str(dates[idx]) < before:
                    return True
        return False

    # -- filing-index enumeration (cheapness pass) -----------------------------

    def filings_window(self, cik: str, *, start: str, end: str) -> list[dict[str, str]] | None:
        """All filings in [start, end] with item codes, newest first.

        Returns None on a submissions fetch failure (distinct from an empty
        window). Scans filings.recent plus the paginated filings.files pages.
        """
        all_records = self._all_records(cik)
        if all_records is None:
            return None
        out: dict[str, dict[str, str]] = {}
        for records in all_records:
            accessions = self._column(records, "accessionNumber")
            forms = self._column(records, "form")
            dates = self._column(records, "filingDate")
            items = self._column(records, "items")
            primary_docs = self._column(records, "primaryDocument")
            for idx, acc in enumerate(accessions):
                filed = str(dates[idx]) if idx < len(dates) else ""
                if not filed or filed < start or filed > end:
                    continue
                accession = str(acc)
                if accession in out:
                    continue
                out[accession] = {
                    "accession": accession,
                    "form": str(forms[idx]).upper() if idx < len(forms) else "",
                    "filing_date": filed,
                    "items": str(items[idx]) if idx < len(items) else "",
                    "primary_document": str(primary_docs[idx]) if idx < len(primary_docs) else "",
                }
        return sorted(out.values(), key=lambda f: (f["filing_date"], f["accession"]), reverse=True)

    # -- shared --------------------------------------------------------------

    def _latest_matching(self, cik: str, *, before: str, match) -> tuple[str, str] | None:
        all_records = self._all_records(cik)
        if all_records is None:
            return None
        best: tuple[str, str] | None = None
        for records in all_records:
            accessions = self._column(records, "accessionNumber")
            forms = self._column(records, "form")
            dates = self._column(records, "filingDate")
            items = self._column(records, "items")
            for idx, acc in enumerate(accessions):
                form = str(forms[idx]).upper() if idx < len(forms) else ""
                filed = str(dates[idx]) if idx < len(dates) else ""
                item_str = str(items[idx]) if idx < len(items) else ""
                if not filed or filed >= before:
                    continue
                if not match(form, item_str):
                    continue
                if best is None or filed > best[1]:
                    best = (str(acc), filed)
        return best
