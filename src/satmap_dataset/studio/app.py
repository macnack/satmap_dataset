"""satmap-studio — Streamlit UI for satmap_dataset."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import streamlit as st
from pydantic import ValidationError
from streamlit_folium import st_folium

from satmap_dataset.cli import _slugify_location_name
from satmap_dataset.models import DemAvailabilityReport, IndexManifest, YearAvailabilityReport
from satmap_dataset.pipeline import dem_availability, index_builder, location_run, raw_export
from satmap_dataset.studio.config_builders import (
    PROVIDER_PRESETS,
    build_dem_availability_config,
    build_dem_config,
    build_index_config,
    build_location_payload,
    build_osm_config,
    build_raw_export_config,
    build_run_config,
    merge_base_and_location_payload,
    resolve_base_json,
    write_location_json,
    write_merged_run_json,
)
from satmap_dataset.studio.geo import (
    bbox_corners_wgs84,
    bbox_from_center,
    estimate_output_pixels,
    nominatim_search,
    square_side_meters,
)
from satmap_dataset.studio.jobs import Job, JobStatus, migrate_job
from satmap_dataset.studio.param_hints import (
    AREA_KM2,
    CENTER_LAT,
    CENTER_LON,
    CELL_MODE,
    CONCURRENCY,
    EQUALIZE_GSD,
    LANT_API_KEY,
    LANT_PASSWORD,
    LANT_USERNAME,
    LOCATION_NAME,
    MIN_YEARS,
    MODE,
    NLS_API_KEY,
    PROFILE,
    PROVIDER,
    PX_PER_METER,
    RAW_EXPORT,
    RAW_ROOT,
    RUN_DEM,
    RUN_OSM,
    CHECK_DEM,
    CHECK_INDEX,
    RUN_FULL_STACK,
    SAVE_LOCATION_JSON,
    SEARCH_QUERY,
    SLEEP_MAX,
    SLEEP_MIN,
    STRICT_YEARS,
    VALIDATE,
    WMS_FALLBACK,
    WRITE_SECRET,
    YEAR_END,
    YEAR_START,
)

REPO_ROOT = Path.cwd().resolve()

DEFAULT_LAT = 52.4012627
DEFAULT_LON = 16.9517999


def _active_job() -> Job | None:
    job = st.session_state.get("active_job")
    if job is not None:
        migrate_job(job)
    return job


def _render_job_progress(job: Job, *, title: str | None = None) -> None:
    """Progress bar, current step label, and rolling log tail for a running or finished job."""
    migrate_job(job)
    if title:
        st.markdown(f"**{title}**")
    total = max(getattr(job.state, "progress_total", 0), 1)
    current = min(max(getattr(job.state, "progress_current", 0), 0), total)
    fraction = current / total
    st.progress(fraction, text=getattr(job.state, "progress_label", "") or job.state.message or job.name)
    progress_label = getattr(job.state, "progress_label", "")
    if progress_label and job.state.message and progress_label != job.state.message:
        st.caption(job.state.message)
    st.markdown("**Log**")
    logs = getattr(job.state, "logs", None) or []
    log_text = "\n".join(logs) if logs else "(waiting for log output…)"
    st.code(log_text, language=None)


def _poll_running_job(job: Job) -> None:
    """Auto-refresh the page while a background job is active."""
    time.sleep(0.8)
    st.rerun()


def _init_session_state() -> None:
    defaults: dict[str, Any] = {
        "provider": "geoportal",
        "location_name": "Poznan",
        "center_lat": DEFAULT_LAT,
        "center_lon": DEFAULT_LON,
        "area_km2": 4.0,
        "year_start": 2015,
        "year_end": 2025,
        "px_per_meter": 15.0,
        "profile": "reference",
        "mode": "hybrid",
        "wms_fallback_missing_years": False,
        "min_years": 3,
        "strict_years": False,
        "concurrency": 8,
        "sleep_min": 0.6,
        "sleep_max": 2.2,
        "run_dem": True,
        "run_osm": True,
        "validate": True,
        "raw_export": False,
        "cell_mode": "footprint",
        "equalize_gsd": True,
        "raw_root": "",
        "nls_api_key": "",
        "lant_api_key": "",
        "lant_username": "",
        "lant_password": "",
        "write_secret": False,
        "search_query": "",
        "index_manifest_path": None,
        "index_report": None,
        "dem_report": None,
        "last_run_artifact": None,
        "last_run_code": None,
        "active_job": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _provider_options() -> dict[str, Any]:
    provider = st.session_state["provider"]
    opts: dict[str, Any] = {}
    if provider == "nls" and st.session_state["nls_api_key"]:
        opts["api_key"] = st.session_state["nls_api_key"]
    if provider == "lantmateriet":
        if st.session_state["lant_api_key"]:
            opts["api_key"] = st.session_state["lant_api_key"]
        if st.session_state["lant_username"]:
            opts["username"] = st.session_state["lant_username"]
        if st.session_state["lant_password"]:
            opts["password"] = st.session_state["lant_password"]
    return opts


def _current_location_payload() -> dict[str, Any]:
    raw_root = st.session_state.get("raw_root") or None
    return build_location_payload(
        location_name=st.session_state["location_name"],
        center_lat=float(st.session_state["center_lat"]),
        center_lon=float(st.session_state["center_lon"]),
        area_km2=float(st.session_state["area_km2"]),
        provider=st.session_state["provider"],
        year_start=int(st.session_state["year_start"]),
        year_end=int(st.session_state["year_end"]),
        px_per_meter=float(st.session_state["px_per_meter"]),
        profile=st.session_state["profile"],
        mode=st.session_state["mode"],
        wms_fallback_missing_years=bool(st.session_state["wms_fallback_missing_years"]),
        min_years=int(st.session_state["min_years"]),
        strict_years=bool(st.session_state["strict_years"]),
        concurrency=int(st.session_state["concurrency"]),
        sleep_min=float(st.session_state["sleep_min"]),
        sleep_max=float(st.session_state["sleep_max"]),
        provider_options=_provider_options(),
        cell_mode=st.session_state["cell_mode"] if st.session_state["raw_export"] else None,
        equalize_gsd=bool(st.session_state["equalize_gsd"]) if st.session_state["raw_export"] else None,
        raw_root=raw_root,
    )


def _current_srs() -> str:
    payload = _current_location_payload()
    return str(payload["srs"])


def _make_folium_map() -> Any:
    import folium
    from folium.plugins import Draw

    lat = float(st.session_state["center_lat"])
    lon = float(st.session_state["center_lon"])
    srs = _current_srs()
    area_km2 = float(st.session_state["area_km2"])

    try:
        bbox_str, bbox_tuple = bbox_from_center(lat, lon, area_km2, srs)
        corners = bbox_corners_wgs84(bbox_tuple, srs)
    except Exception:
        bbox_str = ""
        corners = []

    m = folium.Map(location=[lat, lon], zoom_start=12, tiles="OpenStreetMap")
    folium.Marker([lat, lon], tooltip="AOI center", popup="Center").add_to(m)
    if corners:
        folium.Polygon(
            locations=corners,
            color="#e11d48",
            fill=True,
            fill_color="#fb7185",
            fill_opacity=0.2,
            weight=2,
            popup=f"AOI {area_km2} km²",
        ).add_to(m)
    Draw(export=False, draw_options={"polyline": False, "circle": False}).add_to(m)
    st.session_state["_preview_bbox"] = bbox_str
    return m


def _tab_location() -> None:
    st.subheader("Location & map")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state["provider"] = st.selectbox(
            "Provider",
            options=list(PROVIDER_PRESETS.keys()),
            index=list(PROVIDER_PRESETS.keys()).index(st.session_state["provider"]),
            format_func=lambda p: {
                "geoportal": "Poland — Geoportal (no token)",
                "lantmateriet": "Sweden — Lantmäteriet",
                "nls": "Finland — NLS (API key)",
                "sentinel2": "Sentinel-2 (no token)",
            }.get(p, p),
            help=PROVIDER,
        )
        st.session_state["location_name"] = st.text_input(
            "Location name",
            st.session_state["location_name"],
            help=LOCATION_NAME,
        )
    with col2:
        st.session_state["center_lat"] = st.number_input(
            "Center latitude (WGS84)",
            value=float(st.session_state["center_lat"]),
            format="%.7f",
            help=CENTER_LAT,
        )
        st.session_state["center_lon"] = st.number_input(
            "Center longitude (WGS84)",
            value=float(st.session_state["center_lon"]),
            format="%.7f",
            help=CENTER_LON,
        )

    provider = st.session_state["provider"]
    if provider == "nls":
        st.session_state["nls_api_key"] = st.text_input(
            "NLS API key",
            value=st.session_state["nls_api_key"],
            type="password",
            help=NLS_API_KEY,
        )
        st.session_state["write_secret"] = st.checkbox(
            "Write API key to .secret (repo root)",
            value=st.session_state["write_secret"],
            help=WRITE_SECRET,
        )
    elif provider == "lantmateriet":
        st.session_state["lant_api_key"] = st.text_input(
            "Bearer API key (optional)",
            value=st.session_state["lant_api_key"],
            type="password",
            help=LANT_API_KEY,
        )
        st.session_state["lant_username"] = st.text_input(
            "Geotorget username (optional)",
            value=st.session_state["lant_username"],
            help=LANT_USERNAME,
        )
        st.session_state["lant_password"] = st.text_input(
            "Geotorget password (optional)",
            value=st.session_state["lant_password"],
            type="password",
            help=LANT_PASSWORD,
        )

    st.session_state["area_km2"] = st.slider(
        "Area (km²)",
        min_value=1.0,
        max_value=25.0,
        value=float(st.session_state["area_km2"]),
        step=0.5,
        help=AREA_KM2,
    )
    preset_cols = st.columns(3)
    if preset_cols[0].button("4 km² (2×2 km)"):
        st.session_state["area_km2"] = 4.0
        st.rerun()
    if preset_cols[1].button("9 km² (3×3 km)"):
        st.session_state["area_km2"] = 9.0
        st.rerun()
    if preset_cols[2].button("15 km²"):
        st.session_state["area_km2"] = 15.0
        st.rerun()

    with st.expander("Search place (Nominatim)", expanded=False):
        st.session_state["search_query"] = st.text_input(
            "Search",
            st.session_state["search_query"],
            help=SEARCH_QUERY,
        )
        if st.button("Search online"):
            try:
                results = nominatim_search(st.session_state["search_query"])
                if not results:
                    st.warning("No results.")
                else:
                    for item in results:
                        label = item.get("display_name", "")
                        if st.button(label, key=f"nom_{item.get('place_id', label)}"):
                            st.session_state["center_lat"] = float(item["lat"])
                            st.session_state["center_lon"] = float(item["lon"])
                            st.rerun()
            except Exception as exc:
                st.error(f"Search failed: {exc}")

    st.caption(
        "Click the map to move the AOI center. The red rectangle shows the square AOI "
        "from area_km² in the provider CRS."
    )
    map_data = st_folium(
        _make_folium_map(),
        width=None,
        height=450,
        returned_objects=["last_clicked"],
    )
    if map_data and map_data.get("last_clicked"):
        click = map_data["last_clicked"]
        st.session_state["center_lat"] = float(click["lat"])
        st.session_state["center_lon"] = float(click["lng"])
        st.caption("Map clicked — center updated. Refresh map on next interaction.")

    srs = _current_srs()
    side_m = square_side_meters(float(st.session_state["area_km2"]))
    px_w, px_h = estimate_output_pixels(side_m, float(st.session_state["px_per_meter"]))
    bbox_preview = st.session_state.get("_preview_bbox", "")
    st.markdown(
        f"**SRS:** `{srs}` · **Bbox:** `{bbox_preview}` · "
        f"**Side:** {side_m:.0f} m · **Est. output:** {px_w}×{px_h} px"
    )


def _tab_settings() -> None:
    st.subheader("Run settings")
    c1, c2 = st.columns(2)
    with c1:
        st.session_state["year_start"] = st.number_input(
            "Year start",
            min_value=1900,
            max_value=2100,
            value=int(st.session_state["year_start"]),
            help=YEAR_START,
        )
        st.session_state["year_end"] = st.number_input(
            "Year end",
            min_value=1900,
            max_value=2100,
            value=int(st.session_state["year_end"]),
            help=YEAR_END,
        )
        st.session_state["px_per_meter"] = st.number_input(
            "Pixels per meter (output resolution)",
            min_value=0.01,
            value=float(st.session_state["px_per_meter"]),
            help=PX_PER_METER,
        )
    with c2:
        st.session_state["profile"] = st.selectbox(
            "Profile",
            options=["train", "reference"],
            index=0 if st.session_state["profile"] == "train" else 1,
            help=PROFILE,
        )
        st.session_state["run_dem"] = st.checkbox(
            "DEM layer",
            value=st.session_state["run_dem"],
            help=RUN_DEM,
        )
        st.session_state["run_osm"] = st.checkbox(
            "OSM labels layer",
            value=st.session_state["run_osm"],
            help=RUN_OSM,
        )
        st.session_state["validate"] = st.checkbox(
            "Validate RGB",
            value=st.session_state["validate"],
            help=VALIDATE,
        )

    with st.expander("Advanced"):
        st.session_state["mode"] = st.selectbox(
            "Download mode",
            options=["hybrid", "wfs_render", "wms_tiled", "stac"],
            index=["hybrid", "wfs_render", "wms_tiled", "stac"].index(st.session_state["mode"]),
            help=MODE,
        )
        st.session_state["wms_fallback_missing_years"] = st.checkbox(
            "WMS fallback for missing years",
            value=st.session_state["wms_fallback_missing_years"],
            help=WMS_FALLBACK,
        )
        st.session_state["min_years"] = st.number_input(
            "Min years required",
            min_value=1,
            value=int(st.session_state["min_years"]),
            help=MIN_YEARS,
        )
        st.session_state["strict_years"] = st.checkbox(
            "Strict years",
            value=st.session_state["strict_years"],
            help=STRICT_YEARS,
        )
        st.session_state["concurrency"] = st.number_input(
            "Concurrency",
            min_value=1,
            max_value=64,
            value=int(st.session_state["concurrency"]),
            help=CONCURRENCY,
        )
        st.session_state["sleep_min"] = st.number_input(
            "Sleep min (s)",
            min_value=0.0,
            value=float(st.session_state["sleep_min"]),
            help=SLEEP_MIN,
        )
        st.session_state["sleep_max"] = st.number_input(
            "Sleep max (s)",
            min_value=0.0,
            value=float(st.session_state["sleep_max"]),
            help=SLEEP_MAX,
        )
        if st.session_state["provider"] == "geoportal":
            st.info("Geoportal is rate-sensitive — keep sleep jitter enabled for index/download.")

    st.session_state["raw_export"] = st.checkbox(
        "Raw tile export after run (sat_roma layout)",
        value=st.session_state["raw_export"],
        help=RAW_EXPORT,
    )
    if st.session_state["raw_export"]:
        st.session_state["cell_mode"] = st.selectbox(
            "Cell mode",
            options=["footprint", "world_window"],
            index=0 if st.session_state["cell_mode"] == "footprint" else 1,
            help=CELL_MODE,
        )
        st.session_state["equalize_gsd"] = st.checkbox(
            "Equalize GSD (world_window)",
            value=st.session_state["equalize_gsd"],
            help=EQUALIZE_GSD,
        )
        st.session_state["raw_root"] = st.text_input(
            "Raw root (empty = SATMAP_RAW_ROOT or ~/Github/sat_data_raw)",
            value=st.session_state["raw_root"],
            help=RAW_ROOT,
        )


def _show_index_report(report: YearAvailabilityReport | IndexManifest) -> None:
    st.write(f"**Passed:** {report.passed}")
    if report.errors:
        for err in report.errors:
            st.error(err)
    if report.warnings:
        for warn in report.warnings:
            st.warning(warn)

    rows = []
    gsd_by_year = getattr(report, "gsd_by_year", {}) or {}
    for status in report.year_statuses:
        gsd = gsd_by_year.get(status.year)
        finest = gsd.finest if gsd else None
        coarsest = gsd.coarsest if gsd else None
        hist = gsd.histogram if gsd else {}
        rows.append(
            {
                "year": status.year,
                "status": status.status,
                "features": status.feature_count,
                "gsd_finest_m": finest,
                "gsd_coarsest_m": coarsest,
                "gsd_histogram": json.dumps(hist) if hist else "",
            }
        )
    if rows:
        st.dataframe(rows, use_container_width=True)
    else:
        st.info("No year rows in report.")

    preview_png = getattr(report, "aoi_preview_png", None)
    if preview_png and Path(preview_png).is_file():
        st.image(str(preview_png), caption="AOI preview (index)")


def _tab_availability() -> None:
    st.subheader("Availability")
    job = _active_job()

    if job and job.name in {"index", "dem_availability"} and job.is_running():
        _render_job_progress(job, title=f"Running {job.name}")
        _poll_running_job(job)
        return

    if job and job.name in {"index", "dem_availability"} and job.state.status in {
        JobStatus.SUCCESS,
        JobStatus.FAILED,
    }:
        _render_job_progress(job, title=f"Job {job.name} — {job.state.status.value}")

    if st.button("Check orthophoto availability (index)", type="primary", help=CHECK_INDEX):
        payload = _current_location_payload()
        base_json = resolve_base_json(payload["provider"], REPO_ROOT)
        if st.session_state["write_secret"] and st.session_state["nls_api_key"]:
            secret_path = REPO_ROOT / ".secret"
            secret_path.write_text(st.session_state["nls_api_key"].strip() + "\n", encoding="utf-8")
        try:
            config = build_index_config(payload, base_json)
        except (ValidationError, ValueError) as exc:
            st.error(f"Invalid config: {exc}")
            return

        def run_index() -> tuple[int, Path]:
            return index_builder.run(config)

        job = Job("index")
        st.session_state["active_job"] = job
        job.start(run_index)
        st.rerun()

    if st.button("Check DEM availability", help=CHECK_DEM):
        payload = _current_location_payload()
        base_json = resolve_base_json(payload["provider"], REPO_ROOT)
        try:
            config = build_dem_availability_config(payload, base_json)
        except (ValidationError, ValueError) as exc:
            st.error(f"Invalid config: {exc}")
            return

        def run_dem_avail() -> tuple[int, Path]:
            return dem_availability.run(config)

        job = Job("dem_availability")
        st.session_state["active_job"] = job
        job.start(run_dem_avail)
        st.rerun()

    job = _active_job()
    if job and job.state.status in {JobStatus.SUCCESS, JobStatus.FAILED}:
        st.write(job.state.message)
        if job.state.error:
            st.error(job.state.error)
        if job.name == "index" and job.state.artifact_path:
            path = Path(job.state.artifact_path)
            if path.name == "index_manifest.json":
                manifest = IndexManifest.model_validate_json(path.read_text(encoding="utf-8"))
                _show_index_report(manifest)
                avail_path = path.parent / "year_availability_report.json"
                if avail_path.is_file():
                    report = YearAvailabilityReport.model_validate_json(
                        avail_path.read_text(encoding="utf-8")
                    )
                    st.session_state["index_report"] = report
            else:
                report = YearAvailabilityReport.model_validate_json(path.read_text(encoding="utf-8"))
                _show_index_report(report)
                st.session_state["index_report"] = report
        if job.name == "dem_availability" and job.state.artifact_path:
            dem_report = DemAvailabilityReport.model_validate_json(
                Path(job.state.artifact_path).read_text(encoding="utf-8")
            )
            st.session_state["dem_report"] = dem_report
            rows = [
                {
                    "product": e.product,
                    "datum": e.datum,
                    "year": e.year,
                    "tiles": e.tile_count,
                    "coverage": e.coverage,
                }
                for e in dem_report.entries
                if e.tile_count > 0
            ]
            if rows:
                st.dataframe(rows, use_container_width=True)
        if st.button("Clear job status"):
            st.session_state["active_job"] = None
            st.rerun()

    if st.session_state.get("index_report"):
        st.markdown("### Cached index report")
        _show_index_report(st.session_state["index_report"])


def _tab_run() -> None:
    st.subheader("Run & status")
    job = _active_job()

    if job and job.name == "location_run" and job.is_running():
        _render_job_progress(job, title="Running full stack")
        _poll_running_job(job)
        return

    if job and job.name == "location_run" and job.state.status in {JobStatus.SUCCESS, JobStatus.FAILED}:
        _render_job_progress(job, title=f"Run — {job.state.status.value}")

    if st.button("Run full stack (RGB + DEM + OSM)", type="primary", help=RUN_FULL_STACK):
        payload = _current_location_payload()
        base_json = resolve_base_json(payload["provider"], REPO_ROOT)
        if st.session_state["write_secret"] and st.session_state["nls_api_key"]:
            (REPO_ROOT / ".secret").write_text(
                st.session_state["nls_api_key"].strip() + "\n",
                encoding="utf-8",
            )
        try:
            rgb_config = build_run_config(payload, base_json)
            dem_config = build_dem_config(payload, base_json) if st.session_state["run_dem"] else None
            osm_config = build_osm_config(payload, base_json) if st.session_state["run_osm"] else None
        except (ValidationError, ValueError) as exc:
            st.error(f"Invalid config: {exc}")
            return

        artifacts_dir = Path(str(rgb_config.artifacts_dir))
        run_raw = bool(st.session_state["raw_export"])

        def run_stack() -> tuple[int, Path]:
            code, artifact = location_run.run_location(
                rgb_config=rgb_config,
                dem_config=dem_config,
                osm_config=osm_config,
                artifacts_dir=artifacts_dir,
                run_dem=bool(st.session_state["run_dem"]),
                run_osm=bool(st.session_state["run_osm"]),
                validate=bool(st.session_state["validate"]),
            )
            if run_raw and code == 0:
                raw_config = build_raw_export_config(payload, base_json)
                raw_code, _ = raw_export.run(raw_config)
                code = max(code, raw_code)
            return code, artifact

        job = Job("location_run")
        st.session_state["active_job"] = job
        job.start(run_stack)
        st.rerun()

    job = _active_job()
    if job and job.name == "location_run" and job.state.status in {JobStatus.SUCCESS, JobStatus.FAILED}:
        st.write(job.state.message)
        if job.state.error:
            st.error(job.state.error)
        st.session_state["last_run_code"] = job.state.exit_code
        st.session_state["last_run_artifact"] = job.state.artifact_path
        if job.state.artifact_path:
            artifact = Path(job.state.artifact_path)
            st.code(str(artifact), language=None)
            artifacts_dir = artifact.parent
            validation = artifacts_dir / "validation_report.json"
            if validation.is_file():
                st.markdown(f"[validation_report.json]({validation})")
            preview = artifacts_dir / "aoi_preview.png"
            if preview.is_file():
                st.image(str(preview), caption="AOI preview")
        if st.button("Clear run status"):
            st.session_state["active_job"] = None
            st.rerun()

    st.markdown("### Save for CLI replay")
    slug = _slugify_location_name(st.session_state["location_name"])
    if st.button("Save location JSON + merged run JSON", help=SAVE_LOCATION_JSON):
        payload = _current_location_payload()
        base_json = resolve_base_json(payload["provider"], REPO_ROOT)
        try:
            loc_path = write_location_json(payload, REPO_ROOT, slug=slug)
            run_path = write_merged_run_json(payload, base_json, REPO_ROOT, slug=slug)
            st.success(f"Saved {loc_path} and {run_path}")
            merged = merge_base_and_location_payload(base_json, payload)
            st.code(
                f"just run-location-json location_json={loc_path}\n"
                f"python -m satmap_dataset.cli run-json {run_path}",
                language="bash",
            )
        except Exception as exc:
            st.error(str(exc))


def main() -> None:
    st.set_page_config(page_title="satmap-studio", layout="wide")
    _init_session_state()

    st.title("satmap-studio")
    st.caption("Streamlit UI for satmap_dataset — index, download, render, DEM/OSM layers")

    with st.sidebar:
        st.markdown("### Scope")
        st.markdown(
            "v1 covers single-location runs. Not included: batch `run-all`, "
            "trajectory, LROC NAC, roots management."
        )
        st.markdown("### Quick start")
        st.code("just install-studio\njust studio", language="bash")

    tab_loc, tab_set, tab_avail, tab_run = st.tabs(
        ["Location & map", "Run settings", "Availability", "Run & status"]
    )
    with tab_loc:
        _tab_location()
    with tab_set:
        _tab_settings()
    with tab_avail:
        _tab_availability()
    with tab_run:
        _tab_run()


if __name__ == "__main__":
    main()
