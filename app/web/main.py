"""The IVI web app: the read-only API plus the built SPA.

The legacy Jinja console (candidates browser, ticker pages, /legacy/ops)
was deleted in web-UI Phase 4 — every surface it served lives in the SPA
(/scans, /ops, /outcomes, /reader) on the typed API. Its URL namespaces
stay reserved so stale bookmarks get an honest 404 instead of the shell.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from app.web.api import router as api_router

app = FastAPI(title="IVI")
app.include_router(api_router)


@app.exception_handler(sqlite3.OperationalError)
async def _incomplete_schema(request: Request, exc: sqlite3.OperationalError) -> JSONResponse:
    """A missing table or column is a named precondition, not a server error.

    The read model guards the tables it can live without (an absent watchlist
    is an empty watchlist). This is the backstop for everything else: a
    database that predates the current schema answers with the same
    ``{precondition, detail}`` 503 the SPA already renders as its offline
    state, and says how to fix it. Any other database error stays a 500.
    """

    message = str(exc)
    if not message.startswith(("no such table", "no such column")):
        raise exc
    return JSONResponse(
        status_code=503,
        content={
            "detail": {
                "precondition": "engine_db_schema_incomplete",
                "detail": f"{message}. The database predates the current schema; "
                "run `ivi init-db` to complete it.",
            }
        },
    )


# --- Web UI SPA -------------------------------------------------------------
# FastAPI serves the built Vite bundle from webui/dist; node is a build-time
# dependency only. The catch-all must stay the last registered route.

WEBUI_DIST = Path(__file__).resolve().parents[2] / "webui" / "dist"

# Namespaces that must never fall through to the SPA shell: the API, plus
# the retired legacy-console URLs (deleted in Phase 4).
_RESERVED_TOP_SEGMENTS = {"api", "legacy", "candidates", "ticker", "status"}

_BOOT_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>IVI — not built yet</title>
<style>
  body { margin: 0; min-height: 100vh; display: grid; place-items: center;
         background: #060B16; color: #E8F0FA;
         font: 16px/1.6 ui-sans-serif, system-ui, sans-serif; }
  .panel { max-width: 34rem; padding: 2rem 2.5rem; border-radius: 18px;
           background: rgba(11, 22, 38, 0.55);
           border: 1px solid rgba(148, 197, 255, 0.10);
           box-shadow: 0 12px 40px rgba(0, 0, 0, 0.45),
                       inset 0 1px 0 rgba(255, 255, 255, 0.12); }
  h1 { font-size: 1.4rem; font-weight: 600; margin: 0 0 0.75rem; }
  code { font-family: ui-monospace, monospace; color: #94C5FF; }
  p { color: #8FA3BD; margin: 0.5rem 0; }
</style>
</head>
<body>
<div class="panel">
  <h1>IVI is not built yet</h1>
  <p>The API is live (<code>/api/watchlist</code>, <code>/api/runs</code>).</p>
  <p>Build the UI once: <code>cd webui &amp;&amp; npm install &amp;&amp; npm run build</code>, then reload.</p>
</div>
</body>
</html>"""


def _spa_index_response():
    index_path = WEBUI_DIST / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse(_BOOT_PAGE)


@app.get("/", include_in_schema=False)
def spa_root():
    return _spa_index_response()


@app.get("/{spa_path:path}", include_in_schema=False)
def spa_catch_all(spa_path: str):
    top_segment = spa_path.split("/", 1)[0]
    if top_segment in _RESERVED_TOP_SEGMENTS:
        raise HTTPException(status_code=404, detail="Not found")
    if WEBUI_DIST.exists():
        candidate = (WEBUI_DIST / spa_path).resolve()
        if str(candidate).startswith(str(WEBUI_DIST.resolve()) + "/") and candidate.is_file():
            return FileResponse(candidate)
    return _spa_index_response()
