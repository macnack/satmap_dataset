#!/usr/bin/env python3
"""Build a side-by-side PNG preview of two rendered orthophoto TIFFs.

Typical use:

    python scripts/preview_side_by_side.py \\
        rendered_kisa/year_2017.tiff \\
        rendered_kisa_winter/year_2024.tiff \\
        -o previews/kisa_2017_vs_2024.png

Both inputs are downsampled to ``--size`` pixels via gdal_translate, then
composited side-by-side with optional labels above each panel. PIL is the
only Python dependency; gdal_translate must be on PATH.
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


def _label_for(path: Path, override: str | None) -> str:
    if override is not None:
        return override
    # year_2017.tiff → "2017"; otherwise stem.
    stem = path.stem
    if stem.startswith("year_"):
        return stem[len("year_"):]
    return stem


def _annotate(image, text: str) -> None:
    """Burn a label into the top-left corner using PIL."""
    from PIL import ImageDraw, ImageFont

    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=max(14, image.height // 24))
    except OSError:
        font = ImageFont.load_default()
    pad = 6
    bbox = draw.textbbox((pad, pad), text, font=font)
    draw.rectangle((bbox[0] - 4, bbox[1] - 2, bbox[2] + 4, bbox[3] + 2), fill=(0, 0, 0, 200))
    draw.text((pad, pad), text, fill=(255, 255, 255), font=font)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("left", type=Path, help="First TIFF (commonly the summer / map source).")
    parser.add_argument("right", type=Path, help="Second TIFF (commonly the winter / current photo).")
    parser.add_argument("-o", "--out", type=Path, required=True, help="Output PNG path.")
    parser.add_argument("--size", type=int, default=400, help="Pixel side length for each panel (default 400).")
    parser.add_argument("--gap", type=int, default=10, help="White gap (px) between panels (default 10).")
    parser.add_argument("--left-label", default=None, help="Override the auto-derived label for the left panel.")
    parser.add_argument("--right-label", default=None, help="Override the auto-derived label for the right panel.")
    parser.add_argument("--no-labels", action="store_true", help="Skip burning labels onto the panels.")
    args = parser.parse_args()

    if shutil.which("gdal_translate") is None:
        sys.stderr.write("error: gdal_translate not found on PATH; install GDAL.\n")
        return 2
    for src in (args.left, args.right):
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
        left_png = tmp_dir / "left.png"
        right_png = tmp_dir / "right.png"
        _downsample_to_png(args.left, left_png, args.size)
        _downsample_to_png(args.right, right_png, args.size)
        left_image = Image.open(left_png).convert("RGB")
        right_image = Image.open(right_png).convert("RGB").resize(left_image.size, Image.LANCZOS)

    if not args.no_labels:
        _annotate(left_image, _label_for(args.left, args.left_label))
        _annotate(right_image, _label_for(args.right, args.right_label))

    composite = Image.new("RGB", (left_image.width + args.gap + right_image.width, left_image.height), "white")
    composite.paste(left_image, (0, 0))
    composite.paste(right_image, (left_image.width + args.gap, 0))
    composite.save(args.out)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
