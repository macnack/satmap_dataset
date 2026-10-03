"""FastAPI application for satmap-web."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from satmap_dataset.web.jobs import JobRegistry
from satmap_dataset.web.service import (
    cli_deeplink,
    default_repo_root,
    list_location_files,
    location_status,
    location_summary,
    resolve_location_path,
    start_pipeline_job,
)


class RunRequest(BaseModel):
    location_id: str = Field(..., description="Location JSON stem, e.g. poznan")
    kind: Literal["index", "run"] = "run"


def _frontend_dist() -> Path | None:
    # Prefer built SPA next to the package, then repo-root web-ui/dist.
    candidates = [
        Path(__file__).resolve().parent / "static",
        default_repo_root() / "web-ui" / "dist",
    ]
    for path in candidates:
        if (path / "index.html").is_file():
            return path
    return None


def create_app(*, repo_root: Path | None = None) -> FastAPI:
    root = (repo_root or default_repo_root()).resolve()
    jobs = JobRegistry()

    app = FastAPI(
        title="satmap-web",
        description="Lightweight browser UI API for satmap_dataset",
        version="0.1.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.repo_root = root
    app.state.jobs = jobs

    @app.get("/api/health")
    def health() -> dict[str, object]:
        return {"ok": True, "repo_root": str(root), "name": "satmap-web"}

    @app.get("/api/providers")
    def providers() -> dict[str, object]:
        from satmap_dataset.studio.config_builders import PROVIDER_PRESETS

        return {
            "providers": [
                {"id": key, "srs": meta["srs"], "base_json": meta["base_json"]}
                for key, meta in PROVIDER_PRESETS.items()
            ]
        }

    @app.get("/api/locations")
    def locations() -> dict[str, object]:
        items = []
        for path in list_location_files(root):
            try:
                items.append(location_summary(path, root))
            except Exception as exc:  # noqa: BLE001 — keep listing resilient
                items.append(
                    {
                        "id": path.stem,
                        "file": str(path),
                        "location_name": path.stem,
                        "error": str(exc),
                    }
                )
        return {"locations": items, "count": len(items)}

    @app.get("/api/locations/{location_id}")
    def get_location(location_id: str) -> dict[str, object]:
        try:
            path = resolve_location_path(location_id, root)
            return location_summary(path, root)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/locations/{location_id}/status")
    def get_status(location_id: str) -> dict[str, object]:
        try:
            return location_status(location_id, repo_root=root, jobs=jobs)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/locations/{location_id}/cli")
    def get_cli(location_id: str, kind: Literal["index", "run"] = "run") -> dict[str, str]:
        try:
            return cli_deeplink(location_id, command=kind, repo_root=root)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/runs")
    def start_run(body: RunRequest) -> dict[str, object]:
        try:
            job = start_pipeline_job(
                body.location_id,
                kind=body.kind,
                jobs=jobs,
                repo_root=root,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return job.snapshot()

    @app.get("/api/runs/{job_id}")
    def get_run(job_id: str) -> dict[str, object]:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}")
        return job.snapshot()

    dist = _frontend_dist()
    if dist is not None:
        assets = dist / "assets"
        if assets.is_dir():
            app.mount("/assets", StaticFiles(directory=assets), name="assets")

        @app.get("/")
        def spa_index() -> FileResponse:
            return FileResponse(dist / "index.html")

        @app.get("/{full_path:path}")
        def spa_fallback(full_path: str) -> FileResponse:
            candidate = dist / full_path
            if candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(dist / "index.html")

    return app


app = create_app()
