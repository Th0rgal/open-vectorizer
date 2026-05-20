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
    parser.add_argument(
        "--alpha-threshold",
        type=float,
        default=8.0,
        help="Alpha threshold used to detect foreground in transparent images",
    )
    parser.add_argument("--simplify", type=float, default=3.2, help="Contour simplification in pixels")
    parser.add_argument(
        "--contour-smooth",
        type=int,
        default=15,
        help="Odd-sized smoothing window applied to traced contours before simplification",
    )
    parser.add_argument(
        "--curve-spacing",
        type=float,
        default=0.0,
        help="Resample smoothed contours at this pixel spacing before cubic fitting; 0 disables",
    )
    parser.add_argument(
        "--corner-angle",
        type=float,
        default=0.0,
        help="Angles at or below this value keep crisp handles; 0 disables corner clamping",
    )
    parser.add_argument(
        "--corner-radius",
        type=float,
        default=0.0,
        help="Optional radius in pixels used to round simplified contour corners",
    )
    parser.add_argument(
        "--corner-rounding",
        type=int,
        default=1,
        help="Chaikin corner-cut iterations before spline fitting",
    )
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
            alpha_threshold=args.alpha_threshold,
            simplify=args.simplify,
            contour_smooth=args.contour_smooth,
            curve_spacing=args.curve_spacing,
            corner_angle=args.corner_angle,
            corner_radius=args.corner_radius,
            corner_rounding=args.corner_rounding,
            min_area=args.min_area,
            palette=palette,
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg, encoding="utf-8")


if __name__ == "__main__":
    main()
