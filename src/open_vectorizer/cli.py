from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from . import __version__
from .trace import TraceOptions, _normalize_hex_color, trace_image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="open-vectorizer",
        description="Convert low-color raster artwork into grouped, smooth SVG paths.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument("input", type=Path, help="Input raster image")
    parser.add_argument("output", type=Path, help="Output SVG path, or '-' for stdout")
    parser.add_argument(
        "--groups",
        type=_int_at_least("groups", 1),
        default=2,
        help="Number of foreground color groups",
    )
    parser.add_argument(
        "--resize",
        type=_int_at_least("resize", 0),
        default=1200,
        help="Longest side used for tracing",
    )
    parser.add_argument(
        "--padding",
        type=_int_at_least("padding", 0),
        default=32,
        help="Transparent viewBox padding",
    )
    parser.add_argument(
        "--threshold",
        type=_float_at_least("threshold", 0.0),
        default=8.0,
        help="Background distance threshold",
    )
    parser.add_argument(
        "--background",
        type=_parse_color,
        default=None,
        help="Optional background hex color override. Example: '#ffffff'",
    )
    parser.add_argument(
        "--alpha-threshold",
        type=_float_between("alpha-threshold", 0.0, 255.0),
        default=8.0,
        help="Alpha threshold used to detect foreground in transparent images",
    )
    parser.add_argument(
        "--mask-blur",
        type=_float_at_least("mask-blur", 0.0),
        default=0.0,
        help="Gaussian sigma used to fair binary color masks before contour extraction",
    )
    parser.add_argument(
        "--simplify",
        type=_float_at_least("simplify", 0.0),
        default=3.2,
        help="Contour simplification in pixels",
    )
    parser.add_argument(
        "--contour-smooth",
        type=_int_at_least("contour-smooth", 0),
        default=15,
        help="Odd-sized smoothing window applied to traced contours before simplification",
    )
    parser.add_argument(
        "--curve-spacing",
        type=_float_at_least("curve-spacing", 0.0),
        default=0.0,
        help="Resample smoothed contours at this pixel spacing before cubic fitting; 0 disables",
    )
    parser.add_argument(
        "--corner-angle",
        type=_float_between("corner-angle", 0.0, 180.0),
        default=0.0,
        help="Angles at or below this value keep crisp handles; 0 disables corner clamping",
    )
    parser.add_argument(
        "--corner-radius",
        type=_float_at_least("corner-radius", 0.0),
        default=0.0,
        help="Optional radius in pixels used to round simplified contour corners",
    )
    parser.add_argument(
        "--corner-rounding",
        type=_int_at_least("corner-rounding", 0),
        default=1,
        help="Chaikin corner-cut iterations before spline fitting",
    )
    parser.add_argument(
        "--curve-fit-error",
        type=_float_at_least("curve-fit-error", 0.0),
        default=1.2,
        help="Approximate contours with recursive least-squares cubic Beziers at this error",
    )
    parser.add_argument(
        "--min-area",
        type=_float_at_least("min-area", 0.0),
        default=18.0,
        help="Small component removal threshold",
    )
    parser.add_argument("--no-crop", action="store_true", help="Keep the original image viewBox")
    parser.add_argument(
        "--palette",
        type=_parse_palette,
        default=None,
        help="Comma-separated fill colors. Example: '#36d7d4,#111111'",
    )
    parser.add_argument(
        "--dark-palette",
        type=_parse_palette,
        default=None,
        help=(
            "Comma-separated fill colors used inside @media (prefers-color-scheme: dark). "
            "Example: '#8edbc8,#f4ead8'"
        ),
    )
    return parser


def _parse_palette(value: str) -> list[str]:
    colors = [color.strip() for color in value.split(",") if color.strip()]
    try:
        return [_normalize_hex_color(color, "palette") for color in colors]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _parse_color(value: str) -> str:
    try:
        return _normalize_hex_color(value, "background")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _int_at_least(name: str, minimum: int) -> Callable[[str], int]:
    def parse(value: str) -> int:
        try:
            parsed = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{name} must be an integer") from exc
        if parsed < minimum:
            raise argparse.ArgumentTypeError(f"{name} must be at least {minimum}")
        return parsed

    return parse


def _float_at_least(name: str, minimum: float) -> Callable[[str], float]:
    def parse(value: str) -> float:
        parsed = _parse_float(name, value)
        if parsed < minimum:
            raise argparse.ArgumentTypeError(f"{name} must be at least {minimum}")
        return parsed

    return parse


def _float_between(name: str, minimum: float, maximum: float) -> Callable[[str], float]:
    def parse(value: str) -> float:
        parsed = _parse_float(name, value)
        if parsed < minimum or parsed > maximum:
            raise argparse.ArgumentTypeError(f"{name} must be between {minimum} and {maximum}")
        return parsed

    return parse


def _parse_float(name: str, value: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be a number") from exc


def main() -> None:
    args = build_parser().parse_args()
    svg = trace_image(
        args.input,
        TraceOptions(
            groups=args.groups,
            resize_long_side=args.resize,
            crop=not args.no_crop,
            padding=args.padding,
            background_threshold=args.threshold,
            background_color=args.background,
            alpha_threshold=args.alpha_threshold,
            mask_blur=args.mask_blur,
            simplify=args.simplify,
            contour_smooth=args.contour_smooth,
            curve_spacing=args.curve_spacing,
            corner_angle=args.corner_angle,
            corner_radius=args.corner_radius,
            corner_rounding=args.corner_rounding,
            curve_fit_error=args.curve_fit_error,
            min_area=args.min_area,
            palette=args.palette,
            dark_palette=args.dark_palette,
        ),
    )
    if str(args.output) == "-":
        print(svg, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(svg, encoding="utf-8")


if __name__ == "__main__":
    main()
