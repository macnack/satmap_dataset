"""CRS helpers for the LROC NAC provider.

ODE expects planetocentric lon/lat degrees. For the geographic lunar CRS
(`IAU_2015:30100`) the request bbox is already lon/lat and only needs
reordering. A projected lunar CRS (equirectangular/sinusoidal/polar) is
converted corner-wise via pyproj's IAU_2015 authority.
"""

from __future__ import annotations

_GEOGRAPHIC_LUNAR = "IAU_2015:30100"


def _is_lunar(srs: str) -> bool:
    return srs.upper().startswith("IAU_2015:301")


def normalize_bbox_to_ode(bbox: str, srs: str) -> tuple[float, float, float, float]:
    """Return (westlon, eastlon, minlat, maxlat) in degrees for ODE."""
    if not _is_lunar(srs):
        raise ValueError(
            f"lroc_nac requires a lunar IAU_2015:301xx CRS; got srs={srs!r}."
        )
    parts = [float(p.strip()) for p in bbox.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must have format xmin,ymin,xmax,ymax")
    xmin, ymin, xmax, ymax = parts
    if srs.upper() == _GEOGRAPHIC_LUNAR:
        return (xmin, ymin, xmax, ymax)
    # Projected lunar CRS: convert the four corners to geographic lon/lat.
    from pyproj import Transformer

    transformer = Transformer.from_crs(srs, _GEOGRAPHIC_LUNAR, always_xy=True)
    lons: list[float] = []
    lats: list[float] = []
    for x, y in ((xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax)):
        lon, lat = transformer.transform(x, y)
        lons.append(float(lon))
        lats.append(float(lat))
    return (min(lons), max(lons), min(lats), max(lats))
