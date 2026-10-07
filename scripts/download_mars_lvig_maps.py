#!/usr/bin/env python3
"""Download MARS-LVIG site basemaps from configs/run/mars_lvig/places.json.

HK places use provider=landsd_hk (trajectory cells). Armenia (AM*) uses the
Cadastre Ortho_2021 WMS (20 cm). GPS for AM comes from streams/*.npz.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import urllib.parse
import urllib.request
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satmap_dataset.config import TrajectoryConfig
from satmap_dataset.pipeline import trajectory as trajectory_stage
from satmap_dataset.providers.landsd_hk import tiles
from satmap_dataset.providers.landsd_hk.tiles import MercatorBBox, StitchResult, write_geotiff_3857
from satmap_dataset.trajectory import TrackPoint, load_track


def _expand(value: str, root: Path) -> str:
    return value.replace("${MARS_LVIG_ROOT}", str(root)).replace("$MARS_LVIG_ROOT", str(root))


def _mars_root(cfg: dict) -> Path:
    env = cfg.get("mars_lvig_root_env", "MARS_LVIG_ROOT")
    default = cfg.get("mars_lvig_root_default", "/media/maciej/fifek/mars_lvig")
    return Path(os.environ.get(env, default)).expanduser().resolve()


def _export_gps_from_npz(npz_path: Path, dest: Path, max_points: int = 8000) -> list[TrackPoint]:
    z = np.load(npz_path, allow_pickle=True)
    gps = z["gps_v"]
    step = max(1, len(gps) // max_points)
    items = []
    points: list[TrackPoint] = []
    for i, row in enumerate(gps[::step]):
        lat, lon, alt = float(row[0]), float(row[1]), float(row[2])
        items.append(
            {"image_index": i, "latitude": lat, "longitude": lon, "altitude": alt}
        )
        points.append(TrackPoint(lat=lat, lon=lon))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"items": items}, indent=2), encoding="utf-8")
    return points


def _write_geojson(path: Path, points: list[TrackPoint], place: str) -> None:
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"place": place},
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [[p.lon, p.lat] for p in points],
                        },
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _burn_overlay(tif: Path, points: list[TrackPoint], out_png: Path) -> None:
    import tifffile

    with tifffile.TiffFile(tif) as tiff:
        page = tiff.pages[0]
        arr = page.asarray()
        scale = page.tags["ModelPixelScaleTag"].value
        tie = page.tags["ModelTiepointTag"].value
    ox, oy = float(tie[3]), float(tie[4])
    sx, sy = float(scale[0]), float(scale[1])
    step = max(1, len(points) // 3000)
    xy = []
    for p in points[::step]:
        mx, my = tiles.lonlat_to_mercator(p.lon, p.lat)
        xy.append(((mx - ox) / sx, (oy - my) / sy))
    rgb = arr if arr.ndim == 2 or arr.shape[-1] == 3 else arr[..., :3]
    img = Image.fromarray(rgb).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    if len(xy) >= 2:
        draw.line(xy, fill=(255, 32, 32, 220), width=3)
    Image.alpha_composite(img, overlay).convert("RGB").save(out_png)


def _download_landsd(place: dict, mars_root: Path, out_root: Path) -> Path:
    name = place["name"]
    place_dir = out_root / name
    place_dir.mkdir(parents=True, exist_ok=True)
    track = mars_root / place["track"]
    if not track.exists():
        raise FileNotFoundError(f"track not found: {track}")
    cfg = TrajectoryConfig(
        track_path=track,
        output_dir=place_dir,
        provider="landsd_hk",
        srs="EPSG:3857",
        mode="wms_tiled",
        cell_km=float(place.get("cell_km", 1.0)),
        year_start=int(place.get("year_start", 2025)),
        year_end=int(place.get("year_end", 2025)),
        download=True,
        preview=True,
        concurrency=int(place.get("concurrency", 6)),
        sleep_min=float(place.get("sleep_min", 0.1)),
        sleep_max=float(place.get("sleep_max", 0.4)),
        provider_options=dict(place.get("provider_options") or {}),
    )
    code, manifest_path = trajectory_stage.run(cfg)
    if code != 0:
        raise RuntimeError(f"landsd_hk trajectory failed for {name}: {manifest_path}")
    points = load_track(track)
    _write_geojson(place_dir / "track.geojson", points, name)
    tifs = sorted(place_dir.rglob("*.tif"))
    if not tifs:
        raise RuntimeError(f"no GeoTIFF written for {name}")
    # Prefer the cell mosaic (largest).
    tif = max(tifs, key=lambda p: p.stat().st_size)
    _burn_overlay(tif, points, place_dir / "overlay.png")
    print(f"{name}: landsd_hk OK -> {tif}")
    return tif


def _download_armenia(place: dict, mars_root: Path, out_root: Path) -> Path:
    name = place["name"]
    place_dir = out_root / name
    place_dir.mkdir(parents=True, exist_ok=True)
    npz = mars_root / place["track_npz"]
    if not npz.exists():
        raise FileNotFoundError(f"npz not found: {npz}")
    points = _export_gps_from_npz(npz, place_dir / "gps.json")
    _write_geojson(place_dir / "track.geojson", points, name)

    pad_m = float(place.get("pad_m", 80.0))
    pad = pad_m / 111_000.0
    ll = (
        min(p.lon for p in points) - pad,
        min(p.lat for p in points) - pad,
        max(p.lon for p in points) + pad,
        max(p.lat for p in points) + pad,
    )
    merc = tiles.aoi_to_mercator_bbox(ll, "EPSG:4326")
    lat = sum(p.lat for p in points) / len(points)
    gsd = float(place.get("gsd_m", 0.25))
    max_px = int(place.get("max_px", 2048))
    ground_w = (merc.max_x - merc.min_x) * math.cos(math.radians(lat))
    ground_h = (merc.max_y - merc.min_y) * math.cos(math.radians(lat))
    width = max(512, min(max_px, int(ground_w / gsd)))
    height = max(512, min(max_px, int(ground_h / gsd)))
    aspect = ground_w / max(ground_h, 1e-6)
    if width / height > aspect:
        width = max(512, int(height * aspect))
    else:
        height = max(512, int(width / aspect))

    layer = place.get("layer", "Ortho_2021_20cm")
    base = place.get("wms_base", "https://geoportal.am/gs/Ortho_2021/wms")
    params = {
        "SERVICE": "WMS",
        "VERSION": "1.1.1",
        "REQUEST": "GetMap",
        "LAYERS": layer,
        "STYLES": "",
        "SRS": "EPSG:3857",
        "BBOX": f"{merc.min_x},{merc.min_y},{merc.max_x},{merc.max_y}",
        "WIDTH": str(width),
        "HEIGHT": str(height),
        "FORMAT": "image/png",
    }
    url = f"{base}?{urllib.parse.urlencode(params)}"
    print(f"{name}: WMS {layer} {width}x{height}")
    req = urllib.request.Request(url, headers={"User-Agent": "satmap_dataset-mars-lvig"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = resp.read()
    arr = np.asarray(Image.open(BytesIO(data)).convert("RGB"))
    if float(arr.std()) < 10:
        raise RuntimeError(f"{name}: blank WMS response for layer={layer}")
    px = (merc.max_x - merc.min_x) / arr.shape[1]
    tif = place_dir / f"{layer}.tif"
    write_geotiff_3857(tif, StitchResult(array=arr, bbox=merc, pixel_size=px))
    if place.get("write_overlay", True):
        _burn_overlay(tif, points, place_dir / "overlay.png")
    meta = {
        "place": name,
        "provider": "cadastre_am",
        "layer": layer,
        "year": place.get("year", 2021),
        "gsd_m_requested": gsd,
        "pixel_size_m": px,
        "bbox_3857": merc.as_list(),
        "bbox_wgs84": list(ll),
        "asset": str(tif),
        "sequences": place.get("sequences", []),
    }
    (place_dir / "download_manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"{name}: cadastre_am OK -> {tif}")
    return tif


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/run/mars_lvig/places.json",
        help="places.json listing the four MARS-LVIG sites",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("mars_lvig_maps"),
        help="Output root (one subdirectory per place)",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional place names to download (default: all)",
    )
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    mars_root = _mars_root(cfg)
    if not mars_root.is_dir():
        print(f"MARS_LVIG root not found: {mars_root}", file=sys.stderr)
        print(f"Set {cfg.get('mars_lvig_root_env', 'MARS_LVIG_ROOT')}", file=sys.stderr)
        return 2
    out_root = args.out.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    only = set(args.only) if args.only else None
    results = {}
    for place in cfg["places"]:
        name = place["name"]
        if only is not None and name not in only:
            continue
        provider = place["provider"]
        if provider == "landsd_hk":
            tif = _download_landsd(place, mars_root, out_root)
        elif provider == "cadastre_am":
            tif = _download_armenia(place, mars_root, out_root)
        else:
            raise ValueError(f"unknown provider {provider!r} for {name}")
        results[name] = str(tif)
    summary = {
        "mars_lvig_root": str(mars_root),
        "out": str(out_root),
        "assets": results,
        "config": str(args.config.resolve()),
    }
    (out_root / "download_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
