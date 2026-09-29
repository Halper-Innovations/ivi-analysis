# Test lanes
# ----------
# test               the hermetic developer loop: no network, no local data/,
#                    no child processes. This is the lane CI runs.
# test-slow          intrinsically slow tests and reviewed subprocess tests.
# test-live          read-only audits of a local installation's data/ and live
#                    SEC acceptance checks. Needs network and a populated data/.
# test-all           everything.
PY?=.venv/bin/python3
PYTEST_ARGS?=
FAST_MARKERS=not slow and not network and not sec_live and not live_data and not live_artifact and not subprocess

.PHONY: test test-fast test-slow test-live test-all lint fmt typecheck webui

test test-fast:
	$(PY) -m pytest $(PYTEST_ARGS) -m "$(FAST_MARKERS)"

test-slow:
	$(PY) -m pytest $(PYTEST_ARGS) -m "slow or subprocess"

test-live:
	$(PY) -m pytest $(PYTEST_ARGS) -m "network or sec_live or live_data or live_artifact"

test-all:
	$(PY) -m pytest $(PYTEST_ARGS)

lint:
	$(PY) -m ruff check .

fmt:
	$(PY) -m ruff format .

typecheck:
	$(PY) -m mypy app

webui:
	cd webui && npm ci && npm run build
