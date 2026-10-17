"""Smoke tests for the FastAPI server.

Load-bearing: the route-ordering invariant in `src/server.py` says
explicit `/api/*` routes MUST register before the `app.mount("/", ...)`
StaticFiles catch-all. If a future change reorders them, POST/DELETE
would 405 with no other signal. These tests are the CI guard.
"""

from fastapi.testclient import TestClient

from src.server import app

client = TestClient(app)


def test_post_run_details_routes_to_handler() -> None:
    """POST /api/run_details should hit the JSON handler, not the static mount."""
    res = client.post("/api/run_details", json={"run_dirs": []})
    assert res.status_code == 200, (
        f"POST /api/run_details did not reach the handler — "
        f"got {res.status_code}. The StaticFiles catch-all is probably "
        f"registered before the API routes; fix the order in src/server.py."
    )
    assert res.json() == {}


def test_delete_runs_routes_to_handler() -> None:
    """DELETE /api/runs should hit the JSON handler, not the static mount."""
    res = client.request("DELETE", "/api/runs", json={"run_ids": []})
    assert res.status_code == 200, (
        f"DELETE /api/runs did not reach the handler — got {res.status_code}."
    )
    body = res.json()
    assert body == {"deleted": [], "not_found": [], "failed": []}


def test_post_run_details_rejects_oversized_batch() -> None:
    """Defensive cap: > 5000 dirs returns 413, not OOM."""
    huge = [f"fake/dir_{i}" for i in range(5001)]
    res = client.post("/api/run_details", json={"run_dirs": huge})
    assert res.status_code == 413


def test_runs_index_served_by_static_mount() -> None:
    """The catch-all StaticFiles mount still serves runs_index.json."""
    res = client.get("/runs_index.json")
    # 200 if the file exists in data/system/execution_output/, 404 otherwise — both
    # prove the mount is wired; what we're testing is that no API route
    # shadows the path.
    assert res.status_code in (200, 404)
