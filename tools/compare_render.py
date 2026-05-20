from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


def _load_rgba(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGBA"), dtype=np.int16)


def _render_svg(svg_path: Path, png_path: Path, width: int, height: int) -> None:
    if shutil.which("rsvg-convert") is None:
        raise RuntimeError(
            "rsvg-convert is required to render SVGs; install librsvg2-bin or an equivalent package"
        )
    subprocess.run(
        [
            "rsvg-convert",
            "--format=png",
            f"--width={width}",
            f"--height={height}",
            "--output",
            str(png_path),
            str(svg_path),
        ],
        check=True,
    )


def compare(
    original_path: Path, svg_path: Path, render_path: Path | None = None
) -> dict[str, float]:
    original = _load_rgba(original_path)
    height, width = original.shape[:2]

    if render_path is None:
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            _render_svg(svg_path, Path(tmp.name), width, height)
            rendered = _load_rgba(Path(tmp.name))
    else:
        _render_svg(svg_path, render_path, width, height)
        rendered = _load_rgba(render_path)

    if rendered.shape != original.shape:
        raise ValueError(
            f"rendered shape {rendered.shape} does not match original {original.shape}"
        )

    diff = rendered - original
    rgb_diff = diff[:, :, :3]
    alpha_diff = diff[:, :, 3]

    return {
        "rgb_mae": float(np.mean(np.abs(rgb_diff))),
        "rgba_mae": float(np.mean(np.abs(diff))),
        "rgb_rmse": float(np.sqrt(np.mean(np.square(rgb_diff.astype(np.float64))))),
        "alpha_mae": float(np.mean(np.abs(alpha_diff))),
        "max_abs": float(np.max(np.abs(diff))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render an SVG with rsvg-convert and compare it to an original PNG."
    )
    parser.add_argument("original", type=Path)
    parser.add_argument("svg", type=Path)
    parser.add_argument("--render", type=Path, help="Optional path for the rendered PNG")
    args = parser.parse_args()

    print(json.dumps(compare(args.original, args.svg, args.render), sort_keys=True))


if __name__ == "__main__":
    main()
