from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path


def _tool_path(name: str) -> str | None:
    return shutil.which(name)


def rasterize_geojson_to_file(
    geojson: dict,
    out_path: Path,
    target_bbox: tuple[float, float, float, float],
    target_width: int,
    target_height: int,
    target_srs: str = "EPSG:2180",
) -> None:
    """Reproject GeoJSON (WGS84) to target_srs and burn as uint8 GeoTIFF."""
    if not _tool_path("ogr2ogr") or not _tool_path("gdal_rasterize"):
        raise RuntimeError(
            "OSM rasterization requires GDAL CLI tools (ogr2ogr, gdal_rasterize). "
            "Install GDAL."
        )
    xmin, ymin, xmax, ymax = target_bbox
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        src_json = tmp_dir / "src.geojson"
        reproj_json = tmp_dir / "reproj.geojson"
        src_json.write_text(json.dumps(geojson), encoding="utf-8")
        try:
            subprocess.run(
                ["ogr2ogr", "-f", "GeoJSON", "-t_srs", target_srs,
                 str(reproj_json), str(src_json)],
                check=True, capture_output=True, text=True,
            )
            subprocess.run(
                [
                    "gdal_rasterize",
                    "-burn", "1",
                    "-ts", str(target_width), str(target_height),
                    "-te", str(xmin), str(ymin), str(xmax), str(ymax),
                    "-ot", "Byte",
                    "-co", "COMPRESS=DEFLATE",
                    "-a_srs", target_srs,
                    str(reproj_json), str(out_path),
                ],
                check=True, capture_output=True, text=True,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"GDAL rasterize failed: {(exc.stderr or '')[-500:]}"
            ) from exc
