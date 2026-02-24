from __future__ import annotations

import io
import math
from pathlib import Path
import re
import subprocess

import httpx
from PIL import Image, ImageDraw


EPSG_2180_PROJ4 = (
    "+proj=tmerc +lat_0=0 +lon_0=19 +k=0.9993 "
    "+x_0=500000 +y_0=-5300000 +ellps=GRS80 +units=m +no_defs"
)

_PROJ_DMS_PATTERN = re.compile(
    r"""^\s*(?P<deg>-?\d+(?:\.\d+)?)d(?P<min>\d+(?:\.\d+)?)'(?P<sec>\d+(?:\.\d+)?)"?(?P<hem>[NSEW])?\s*$"""
)


def _parse_bbox(bbox: str) -> tuple[float, float, float, float]:
    parts = [part.strip() for part in bbox.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must have format xmin,ymin,xmax,ymax")
    try:
        xmin, ymin, xmax, ymax = (float(part) for part in parts)
    except ValueError as exc:
        raise ValueError("bbox coordinates must be numeric") from exc
    if xmin >= xmax or ymin >= ymax:
        raise ValueError("bbox must satisfy xmin<xmax and ymin<ymax")
    return xmin, ymin, xmax, ymax


def write_osm_preview_html(*, bbox: str, srs: str, output_path: Path) -> Path:
    if srs.upper() != "EPSG:2180":
        raise ValueError("AOI preview supports only EPSG:2180.")
    xmin, ymin, xmax, ymax = _parse_bbox(bbox)
    center_x = (xmin + xmax) / 2.0
    center_y = (ymin + ymax) / 2.0

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AOI Preview</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <style>
    html, body {{
      margin: 0;
      padding: 0;
      height: 100%;
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f6f7f9;
      color: #1f2937;
    }}
    .layout {{
      display: grid;
      grid-template-rows: auto 1fr;
      height: 100%;
    }}
    .meta {{
      padding: 10px 14px;
      font-size: 13px;
      border-bottom: 1px solid #d1d5db;
      background: #ffffff;
    }}
    #map {{
      width: 100%;
      height: 100%;
    }}
    code {{
      font-size: 12px;
      background: #eef2ff;
      padding: 2px 6px;
      border-radius: 4px;
    }}
  </style>
</head>
<body>
  <div class="layout">
    <div class="meta">
      <div><strong>AOI Preview</strong></div>
      <div>Input bbox (<code>EPSG:2180</code>): <code>{xmin},{ymin},{xmax},{ymax}</code></div>
      <div id="center"></div>
    </div>
    <div id="map"></div>
  </div>

  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/proj4@2.19.10/dist/proj4.min.js"></script>
  <script>
    const bbox2180 = {{
      xmin: {xmin},
      ymin: {ymin},
      xmax: {xmax},
      ymax: {ymax},
    }};
    const center2180 = {{
      x: {center_x},
      y: {center_y},
    }};

    proj4.defs("EPSG:2180", "{EPSG_2180_PROJ4}");

    function toLatLon(point) {{
      const out = proj4("EPSG:2180", "EPSG:4326", [point.x, point.y]);
      return {{ lon: out[0], lat: out[1] }};
    }}

    const corners = [
      toLatLon({{ x: bbox2180.xmin, y: bbox2180.ymin }}),
      toLatLon({{ x: bbox2180.xmin, y: bbox2180.ymax }}),
      toLatLon({{ x: bbox2180.xmax, y: bbox2180.ymin }}),
      toLatLon({{ x: bbox2180.xmax, y: bbox2180.ymax }}),
    ];

    const lats = corners.map((p) => p.lat);
    const lons = corners.map((p) => p.lon);
    const sw = [Math.min(...lats), Math.min(...lons)];
    const ne = [Math.max(...lats), Math.max(...lons)];
    const center = toLatLon(center2180);

    const map = L.map("map");
    L.tileLayer("https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png", {{
      maxZoom: 20,
      attribution: '&copy; OpenStreetMap contributors',
    }}).addTo(map);

    const rect = L.rectangle([sw, ne], {{
      color: "#e11d48",
      weight: 2,
      fillColor: "#fb7185",
      fillOpacity: 0.15,
    }}).addTo(map);
    L.circleMarker([center.lat, center.lon], {{
      radius: 6,
      color: "#1d4ed8",
      fillColor: "#2563eb",
      fillOpacity: 1.0,
      weight: 2,
    }}).addTo(map).bindPopup("AOI center");

    map.fitBounds(rect.getBounds(), {{ padding: [24, 24] }});
    const centerEl = document.getElementById("center");
    centerEl.textContent = `Center lat/lon: ${{center.lat.toFixed(7)}}, ${{center.lon.toFixed(7)}}`;
  </script>
</body>
</html>
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def _epsg2180_to_latlon(x: float, y: float) -> tuple[float, float]:
    try:
        from pyproj import Transformer

        transformer = Transformer.from_crs("EPSG:2180", "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(x, y)
        return float(lat), float(lon)
    except Exception:
        proj_cmd = [
            "proj",
            "-I",
            "-f",
            "%.10f",
            "+proj=tmerc",
            "+lat_0=0",
            "+lon_0=19",
            "+k=0.9993",
            "+x_0=500000",
            "+y_0=-5300000",
            "+ellps=GRS80",
            "+units=m",
            "+no_defs",
        ]
        completed = subprocess.run(
            proj_cmd,
            input=f"{x} {y}\n",
            text=True,
            capture_output=True,
            check=True,
        )
        values = completed.stdout.strip().split()
        if len(values) < 2:
            raise RuntimeError("Failed to parse PROJ output for AOI preview conversion.")
        lon = _parse_proj_angle(values[0])
        lat = _parse_proj_angle(values[1])
        return lat, lon


def _parse_proj_angle(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        match = _PROJ_DMS_PATTERN.match(value)
        if not match:
            raise
        deg = float(match.group("deg"))
        minutes = float(match.group("min"))
        seconds = float(match.group("sec"))
        hemisphere = match.group("hem")
        magnitude = abs(deg) + (minutes / 60.0) + (seconds / 3600.0)
        sign = -1.0 if deg < 0 else 1.0
        if hemisphere in {"S", "W"}:
            sign = -1.0
        if hemisphere in {"N", "E"}:
            sign = 1.0
        return sign * magnitude


def _clip_lat(lat: float) -> float:
    return max(min(lat, 85.05112878), -85.05112878)


def _lonlat_to_world_pixels(lon: float, lat: float, zoom: int) -> tuple[float, float]:
    n = 2**zoom
    x = (lon + 180.0) / 360.0 * 256.0 * n
    lat_rad = math.radians(_clip_lat(lat))
    y = (
        (1.0 - math.log(math.tan(lat_rad) + (1.0 / math.cos(lat_rad))) / math.pi)
        / 2.0
        * 256.0
        * n
    )
    return x, y


def _fit_zoom(sw_lat: float, sw_lon: float, ne_lat: float, ne_lon: float, width: int, height: int) -> int:
    lon_span = abs(ne_lon - sw_lon)
    if lon_span > 180.0:
        lon_span = 360.0 - lon_span
    for zoom in range(19, -1, -1):
        x1, y1 = _lonlat_to_world_pixels(sw_lon, sw_lat, zoom)
        x2, y2 = _lonlat_to_world_pixels(ne_lon, ne_lat, zoom)
        span_x = abs(x2 - x1)
        span_y = abs(y2 - y1)
        if span_x <= width * 0.8 and span_y <= height * 0.8:
            return zoom
    return 0


def _write_schematic_png(
    *,
    output_path: Path,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    center_x: float,
    center_y: float,
    reason: str,
) -> Path:
    width = 1200
    height = 800
    panel_h = 140
    pad = 80

    image = Image.new("RGB", (width, height), "#f6f7f9")
    draw = ImageDraw.Draw(image)

    draw.rectangle((0, 0, width, panel_h), fill="#ffffff", outline="#d1d5db", width=1)
    draw.text((24, 18), "AOI Preview (fallback schematic)", fill="#111827")
    draw.text(
        (24, 50),
        f"BBOX: {xmin:.3f}, {ymin:.3f}, {xmax:.3f}, {ymax:.3f}",
        fill="#374151",
    )
    draw.text((24, 82), f"Reason: {reason[:120]}", fill="#b91c1c")

    map_top = panel_h + 16
    map_bottom = height - 24
    map_left = 24
    map_right = width - 24
    draw.rectangle((map_left, map_top, map_right, map_bottom), fill="#eef2ff", outline="#c7d2fe", width=2)

    aoi_left = map_left + pad
    aoi_top = map_top + pad
    aoi_right = map_right - pad
    aoi_bottom = map_bottom - pad
    draw.rectangle((aoi_left, aoi_top, aoi_right, aoi_bottom), fill="#fda4af", outline="#e11d48", width=4)

    cx = (aoi_left + aoi_right) / 2
    cy = (aoi_top + aoi_bottom) / 2
    r = 8
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill="#2563eb", outline="#1d4ed8", width=2)
    draw.text((map_left + 12, map_top + 12), "Schematic AOI rectangle", fill="#4338ca")
    draw.text((map_left + 12, map_bottom - 28), f"Center XY: {center_x:.3f}, {center_y:.3f}", fill="#1d4ed8")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG", optimize=True)
    return output_path


def write_aoi_preview_png(*, bbox: str, srs: str, output_path: Path) -> Path:
    if srs.upper() != "EPSG:2180":
        raise ValueError("AOI preview supports only EPSG:2180.")
    xmin, ymin, xmax, ymax = _parse_bbox(bbox)
    center_x = (xmin + xmax) / 2.0
    center_y = (ymin + ymax) / 2.0

    try:
        corners = [
            _epsg2180_to_latlon(xmin, ymin),
            _epsg2180_to_latlon(xmin, ymax),
            _epsg2180_to_latlon(xmax, ymin),
            _epsg2180_to_latlon(xmax, ymax),
        ]
        lats = [lat for lat, _ in corners]
        lons = [lon for _, lon in corners]
        sw_lat = min(lats)
        ne_lat = max(lats)
        sw_lon = min(lons)
        ne_lon = max(lons)

        center_lat, center_lon = _epsg2180_to_latlon(center_x, center_y)

        width = 1200
        height = 800
        panel_h = 120
        map_w = width
        map_h = height - panel_h
        zoom = _fit_zoom(sw_lat, sw_lon, ne_lat, ne_lon, map_w, map_h)

        n = 2**zoom
        center_wx, center_wy = _lonlat_to_world_pixels(center_lon, center_lat, zoom)
        top_left_x = center_wx - (map_w / 2.0)
        top_left_y = center_wy - (map_h / 2.0)
        tile_size = 256
        tile_x0 = int(math.floor(top_left_x / tile_size))
        tile_y0 = int(math.floor(top_left_y / tile_size))
        tile_x1 = int(math.floor((top_left_x + map_w - 1) / tile_size))
        tile_y1 = int(math.floor((top_left_y + map_h - 1) / tile_size))

        image = Image.new("RGB", (width, height), "#f6f7f9")
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, width, panel_h), fill="#ffffff", outline="#d1d5db", width=1)
        draw.text((16, 12), f"AOI Preview OSM (z={zoom})", fill="#111827")
        draw.text((16, 40), f"BBOX EPSG:2180: {xmin:.3f}, {ymin:.3f}, {xmax:.3f}, {ymax:.3f}", fill="#374151")
        draw.text((16, 68), f"Center lat/lon: {center_lat:.6f}, {center_lon:.6f}", fill="#374151")

        map_img = Image.new("RGB", (map_w, map_h), "#e5e7eb")
        fetched_tiles = 0
        with httpx.Client(
            timeout=httpx.Timeout(4.0, connect=2.0),
            headers={"User-Agent": "satmap_dataset/0.1 AOI preview"},
        ) as client:
            for tx in range(tile_x0, tile_x1 + 1):
                for ty in range(tile_y0, tile_y1 + 1):
                    if ty < 0 or ty >= n:
                        continue
                    wrapped_tx = tx % n
                    url = f"https://tile.openstreetmap.org/{zoom}/{wrapped_tx}/{ty}.png"
                    response = client.get(url)
                    if response.status_code != 200:
                        continue
                    tile = Image.open(io.BytesIO(response.content)).convert("RGB")
                    px = int(tx * tile_size - top_left_x)
                    py = int(ty * tile_size - top_left_y)
                    map_img.paste(tile, (px, py))
                    fetched_tiles += 1

        if fetched_tiles == 0:
            return _write_schematic_png(
                output_path=output_path,
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax,
                center_x=center_x,
                center_y=center_y,
                reason="Could not download OSM tiles",
            )

        polygon_latlon = [
            _epsg2180_to_latlon(xmin, ymin),
            _epsg2180_to_latlon(xmin, ymax),
            _epsg2180_to_latlon(xmax, ymax),
            _epsg2180_to_latlon(xmax, ymin),
        ]
        polygon_px: list[tuple[float, float]] = []
        for plat, plon in polygon_latlon:
            pwx, pwy = _lonlat_to_world_pixels(plon, plat, zoom)
            polygon_px.append((pwx - top_left_x, pwy - top_left_y))
        draw_map = ImageDraw.Draw(map_img)
        draw_map.polygon(polygon_px, outline="#e11d48", width=4)
        cwx, cwy = _lonlat_to_world_pixels(center_lon, center_lat, zoom)
        cx = cwx - top_left_x
        cy = cwy - top_left_y
        draw_map.ellipse((cx - 7, cy - 7, cx + 7, cy + 7), fill="#2563eb", outline="#1d4ed8", width=2)

        image.paste(map_img, (0, panel_h))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, format="PNG", optimize=True)
        return output_path
    except Exception as exc:
        return _write_schematic_png(
            output_path=output_path,
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
            center_x=center_x,
            center_y=center_y,
            reason=str(exc),
        )
