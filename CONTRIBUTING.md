# Contributing to IVI Analysis

Thanks for your interest. Bug reports, data-quality findings and pull requests
are all welcome.

## Development setup

```bash
git clone https://github.com/Halper-Innovations/ivi-analysis.git
cd ivi-analysis
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
make test
```

The web UI is optional and needs Node 20+: `make webui`.

## Tests

`make test` is the hermetic lane that CI runs. It never touches the network,
your local `data/` directory, or child processes. The test harness blocks all
three and fails the test that tries. Keep it that way:

- Use committed fixtures under `tests/fixtures/` instead of live SEC calls.
  Small, trimmed excerpts of public SEC JSON are fine.
- Assert exact values. A test that accepts "anything close" hides regressions
  in financial math.
- Tests that genuinely need the network or a populated data directory carry
  the `network`, `sec_live`, `live_data` or `live_artifact` marker and run in
  `make test-live`.

## Pull requests

- One logical change per pull request, with a test that fails before it and
  passes after.
- Keep diffs minimal and match the surrounding style (`make fmt` runs Ruff's
  formatter; line length 100).
- Never weaken a data-integrity check to make a result appear. If a number is
  unknowable from the filings, the correct output is an explicit UNKNOWN with a
  reason code, not a fallback.
- Describe user-visible behavior changes in the pull request body.

## Reporting data problems

If IVI Analysis reports a number that disagrees with a company's filing, open
an issue with the ticker, the as-of date, the line item, the value you expected
and a link to the filing. These reports are the most valuable contributions the
project gets.
