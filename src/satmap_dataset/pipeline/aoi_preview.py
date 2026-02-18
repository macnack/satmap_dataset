from __future__ import annotations

from pathlib import Path


EPSG_2180_PROJ4 = (
    "+proj=tmerc +lat_0=0 +lon_0=19 +k=0.9993 "
    "+x_0=500000 +y_0=-5300000 +ellps=GRS80 +units=m +no_defs"
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
