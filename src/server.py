"""FastAPI dev/prod server for the HARBOR frontend.

The file serves two purposes:

1. Static-serve `runs_index.json`, per-run `run_detail.json`, and `.stock_prices/`
   files to the Vite-hosted React SPA on `http://localhost:5173` (via CORS).
2. Serve the `/api/*` mutation + batch endpoints (delete, batch run-detail fetch).

ROUTE ORDERING INVARIANT
------------------------
`StaticFiles` mounted at `/` is a catch-all. Anything registered AFTER the mount
is unreachable. Register all explicit `/api/*` routes BEFORE the mount, or they
will silently 404. There's a smoke test in `tests/test_server.py` that asserts
this — keep it green.

The mount currently only matters in prod (serving a built FE bundle from the same
process). In dev, Vite serves the FE; FastAPI only serves the JSON files under
`EXECUTION_OUTPUT_ROOT`.
"""

import json
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.evaluation.persistence import delete_runs
from src.paths import EXECUTION_OUTPUT_ROOT

app = FastAPI(title="HARBOR")

app.add_middleware(
    CORSMiddleware,
    # Both forms — Chrome surfaces either depending on how Vite advertises itself.
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# --- /api/* routes — MUST be registered BEFORE the StaticFiles mount ---


class RunDetailsRequest(BaseModel):
    run_dirs: list[str]
    fields: list[str] | None = None


# Defensive cap. 5000 run_dirs × ~80 bytes ≈ 400 KB body — comfortably under
# any sensible limit but stops a runaway client from melting the server.
_MAX_BATCH_RUN_DIRS = 5000


@app.post("/api/run_details")
def post_run_details(req: RunDetailsRequest) -> dict[str, dict[str, Any] | None]:
    """Batch read run_detail.json files.

    Returns {run_dir: detail | None}. Missing files → None (not 404 — partial
    success is expected at scale, the client handles per-key absence).
    Optional `fields` slices the response to just those top-level keys.
    """
    if len(req.run_dirs) > _MAX_BATCH_RUN_DIRS:
        raise HTTPException(
            status_code=413,
            detail=f"run_dirs exceeds cap of {_MAX_BATCH_RUN_DIRS}",
        )

    out: dict[str, dict[str, Any] | None] = {}
    for run_dir in req.run_dirs:
        path = EXECUTION_OUTPUT_ROOT / run_dir / "run_detail.json"
        # Path-traversal guard — never serve outside the execution root.
        try:
            path.resolve(strict=True).relative_to(EXECUTION_OUTPUT_ROOT.resolve())
        except (ValueError, OSError):
            out[run_dir] = None
            continue

        try:
            data: dict[str, Any] = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            out[run_dir] = None
            continue

        if req.fields is not None:
            data = {k: data[k] for k in req.fields if k in data}
        out[run_dir] = data
    return out


class DeleteRunsRequest(BaseModel):
    run_ids: list[str]


class DeleteRunsResponse(BaseModel):
    deleted: list[str]  # run_dir strings of successfully deleted runs
    not_found: list[str]  # run_ids the server didn't find
    failed: list[str]  # run_ids whose rmtree raised (entry kept in index)


@app.delete("/api/runs")
def delete_runs_route(req: DeleteRunsRequest) -> DeleteRunsResponse:
    """Bulk delete. Single-run delete is just a one-element list."""
    deleted, not_found, failed = delete_runs(req.run_ids)
    return DeleteRunsResponse(deleted=deleted, not_found=not_found, failed=failed)


# --- Static file serving — catch-all, MUST be last ---

EXECUTION_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=EXECUTION_OUTPUT_ROOT), name="static")
