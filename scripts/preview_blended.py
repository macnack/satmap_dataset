#!/usr/bin/env python3
"""Alpha-blend two rendered orthophoto TIFFs into a single PNG.

Useful for visually verifying co-registration between two same-AOI renders
(e.g. a Lantmäteriet summer mosaic and a Sentinel-2 winter scene). If the
blended image shows clean edges along roads / buildings rather than ghost
doubles, the layers align.

    python scripts/preview_blended.py \\
        rendered_kisa/year_2017.tiff \\
        rendered_kisa_winter/year_2024.tiff \\
        -o previews/kisa_blend.png \\
        --alpha 0.5
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _downsample_to_png(src: Path, dst: Path, size: int) -> None:
    cmd = [
        "gdal_translate",
        "-of", "PNG",
        "-outsize", str(size), str(size),
        "-q",
        str(src),
        str(dst),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("base", type=Path, help="Base TIFF (alpha=0 returns this, e.g. summer / map).")
    parser.add_argument("overlay", type=Path, help="Overlay TIFF (alpha=1 returns this, e.g. winter / current).")
    parser.add_argument("-o", "--out", type=Path, required=True, help="Output PNG path.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Blend weight for the overlay (0.0..1.0, default 0.5).")
    parser.add_argument("--size", type=int, default=400, help="Pixel side length for the preview (default 400).")
    args = parser.parse_args()

    if not 0.0 <= args.alpha <= 1.0:
        sys.stderr.write("error: --alpha must be in [0.0, 1.0]\n")
        return 2
    if shutil.which("gdal_translate") is None:
        sys.stderr.write("error: gdal_translate not found on PATH; install GDAL.\n")
        return 2
    for src in (args.base, args.overlay):
        if not src.exists():
            sys.stderr.write(f"error: {src} not found\n")
            return 2

    try:
        from PIL import Image
    except ImportError:
        sys.stderr.write("error: Pillow is required; pip install Pillow\n")
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        a_png = tmp_dir / "base.png"
        b_png = tmp_dir / "overlay.png"
        _downsample_to_png(args.base, a_png, args.size)
        _downsample_to_png(args.overlay, b_png, args.size)
        a = Image.open(a_png).convert("RGB")
        b = Image.open(b_png).convert("RGB").resize(a.size, Image.LANCZOS)
        blended = Image.blend(a, b, args.alpha)
        blended.save(args.out)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
