from __future__ import annotations

import argparse
from pathlib import Path

from .trace import TraceOptions, trace_image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="open-vectorizer",
        description="Convert low-color raster artwork into grouped, smooth SVG paths.",
    )
    parser.add_argument("input", type=Path, help="Input raster image")
    parser.add_argument("output", type=Path, help="Output SVG path")
    parser.add_argument("--groups", type=int, default=2, help="Number of foreground color groups")
    parser.add_argument("--resize", type=int, default=1200, help="Longest side used for tracing")
    parser.add_argument("--padding", type=int, default=32, help="Transparent viewBox padding")
    parser.add_argument("--threshold", type=float, default=8.0, help="Background distance threshold")
    parser.add_argument("--simplify", type=float, default=3.2, help="Contour simplification in pixels")
    parser.add_argument("--min-area", type=float, default=18.0, help="Small component removal threshold")
    parser.add_argument("--no-crop", action="store_true", help="Keep the original image viewBox")
    parser.add_argument(
        "--palette",
        default=None,
        help="Comma-separated fill colors. Example: '#36d7d4,#111111'",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    palette = None
    if args.palette:
        palette = [color.strip() for color in args.palette.split(",") if color.strip()]

    svg = trace_image(
        args.input,
        TraceOptions(
            groups=args.groups,
            resize_long_side=args.resize,
            crop=not args.no_crop,
            padding=args.padding,
            background_threshold=args.threshold,
            simplify=args.simplify,
            min_area=args.min_area,
            palette=palette,
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg, encoding="utf-8")


if __name__ == "__main__":
    main()

