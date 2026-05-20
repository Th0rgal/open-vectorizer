from importlib.resources import files
from pathlib import Path

import cv2
import numpy as np
import pytest

from open_vectorizer import TraceOptions, __version__, trace_image
from open_vectorizer.cli import build_parser, main
from open_vectorizer.trace import (
    _closed_bezier_fit_path,
    _closed_catmull_rom_path,
    _contour_smooth_window,
    _resample_closed_points,
    _rounded_closed_path,
    _rounded_closed_points,
    _smooth_closed_contour,
)


def test_trace_outputs_grouped_svg(tmp_path: Path) -> None:
    source = Path("examples/keel-compressed.jpg")
    svg = trace_image(
        source,
        TraceOptions(
            groups=2,
            resize_long_side=600,
            palette=["#36d7d4", "#111111"],
            simplify=3.0,
        ),
    )

    assert '<g id="shape-group-1" fill="#36d7d4" fill-rule="evenodd">' in svg
    assert '<g id="shape-group-2" fill="#111111" fill-rule="evenodd">' in svg
    assert svg.count("<path") >= 2
    assert "C 309 407 309 407 309 407" not in svg


def test_trace_can_embed_light_and_dark_theme_palettes() -> None:
    source = Path("examples/keel-compressed.jpg")
    svg = trace_image(
        source,
        TraceOptions(
            groups=2,
            resize_long_side=600,
            palette=["#c77832", "#171717"],
            dark_palette=["#f0a45b", "#f4ead8"],
            simplify=3.0,
        ),
    )

    assert "@media (prefers-color-scheme: dark)" in svg
    assert "--ov-group-1: #c77832;" in svg
    assert "--ov-group-2: #171717;" in svg
    assert "--ov-group-1: #f0a45b;" in svg
    assert "--ov-group-2: #f4ead8;" in svg
    assert ".shape-group-1 { fill: var(--ov-group-1, #c77832); }" in svg
    assert 'class="shape-group shape-group-1"' in svg


def test_trace_uses_accessible_svg_title() -> None:
    source = Path("examples/keel-compressed.jpg")
    svg = trace_image(
        source,
        TraceOptions(
            groups=2,
            resize_long_side=600,
            palette=["#c77832", "#171717"],
            title='Keel & Sail "Mark"',
            simplify=3.0,
        ),
    )

    assert '<title>Keel &amp; Sail "Mark"</title>' in svg
    assert "aria-label='Keel &amp; Sail \"Mark\"'" in svg


def test_trace_is_reproducible_with_default_seed() -> None:
    source = Path("examples/keel-compressed.jpg")
    options = TraceOptions(
        groups=2,
        resize_long_side=600,
        palette=["#c77832", "#171717"],
        simplify=3.0,
    )

    assert trace_image(source, options) == trace_image(source, options)


def test_trace_normalizes_short_hex_palette_colors() -> None:
    source = Path("examples/keel-compressed.jpg")
    svg = trace_image(
        source,
        TraceOptions(
            groups=2,
            resize_long_side=600,
            palette=["#ABC", "#111111"],
            dark_palette=["#DEF", "#F4EAD8"],
            simplify=3.0,
        ),
    )

    assert "--ov-group-1: #aabbcc;" in svg
    assert "--ov-group-1: #ddeeff;" in svg
    assert 'fill="#ABC"' not in svg


def test_trace_rejects_non_hex_palette_colors() -> None:
    source = Path("examples/keel-compressed.jpg")

    with pytest.raises(ValueError, match="palette colors must be hex values"):
        trace_image(
            source,
            TraceOptions(
                groups=2,
                resize_long_side=600,
                palette=["#36d7d4", "url(#paint)"],
                simplify=3.0,
            ),
        )


def test_trace_can_override_estimated_background_color(tmp_path: Path) -> None:
    image_path = tmp_path / "border-touching.png"
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    image[:, :] = (54, 215, 212)
    cv2.rectangle(image, (28, 28), (52, 52), (255, 255, 255), -1)
    cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    estimated = trace_image(
        image_path,
        TraceOptions(
            groups=1,
            resize_long_side=80,
            palette=["#111111"],
            padding=0,
            simplify=1.0,
            min_area=20,
        ),
    )
    overridden = trace_image(
        image_path,
        TraceOptions(
            groups=1,
            resize_long_side=80,
            palette=["#36d7d4"],
            background_color="#fff",
            padding=0,
            simplify=1.0,
            min_area=20,
        ),
    )

    assert 'viewBox="0 0 25 25"' in estimated
    assert 'viewBox="0 0 80 80"' in overridden
    assert 'fill="#36d7d4"' in overridden


def test_cli_parser_rejects_non_hex_palette_colors() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["input.png", "output.svg", "--palette", "#36d7d4,not-a-color"])


def test_cli_parser_rejects_non_hex_background_color() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["input.png", "output.svg", "--background", "white"])


def test_cli_parser_reports_installed_version(capsys: pytest.CaptureFixture[str]) -> None:
    parser = build_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--version"])

    assert exc_info.value.code == 0
    assert capsys.readouterr().out.startswith("open-vectorizer 0.1.0")


def test_package_exposes_installed_version() -> None:
    assert __version__ == "0.1.0"


def test_package_includes_typed_marker() -> None:
    assert files("open_vectorizer").joinpath("py.typed").is_file()


def test_cli_main_writes_svg_with_normalized_palette(tmp_path: Path) -> None:
    output = tmp_path / "keel.svg"
    main(
        [
            "examples/keel-compressed.jpg",
            str(output),
            "--groups",
            "2",
            "--resize",
            "600",
            "--palette",
            "#ABC,#111111",
            "--dark-palette",
            "#DEF,#F4EAD8",
            "--background",
            "#000",
            "--title",
            "Keel mark",
            "--simplify",
            "3",
            "--min-area",
            "1000",
        ],
    )

    svg = output.read_text(encoding="utf-8")
    assert output.exists()
    assert "--ov-group-1: #aabbcc;" in svg
    assert "--ov-group-1: #ddeeff;" in svg
    assert "<title>Keel mark</title>" in svg
    assert "<svg" in svg


def test_cli_main_writes_svg_to_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    main(
        [
            "examples/keel-compressed.jpg",
            "-",
            "--groups",
            "2",
            "--resize",
            "600",
            "--palette",
            "#ABC,#111111",
            "--simplify",
            "3",
            "--min-area",
            "1000",
        ],
    )

    stdout = capsys.readouterr().out
    assert stdout.startswith("<svg")
    assert 'fill="#aabbcc"' in stdout


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--groups", "0"),
        ("--resize", "-1"),
        ("--padding", "-1"),
        ("--threshold", "-1"),
        ("--alpha-threshold", "256"),
        ("--mask-blur", "-0.1"),
        ("--simplify", "-0.1"),
        ("--contour-smooth", "-1"),
        ("--curve-spacing", "-0.1"),
        ("--corner-angle", "181"),
        ("--corner-radius", "-0.1"),
        ("--corner-rounding", "-1"),
        ("--curve-fit-error", "-0.1"),
        ("--min-area", "-0.1"),
        ("--seed", "-1"),
        ("--seed", "2147483648"),
    ],
)
def test_cli_parser_rejects_invalid_numeric_ranges(flag: str, value: str) -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["input.png", "output.svg", flag, value])


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("groups", 0, "groups must be at least 1"),
        ("resize_long_side", -1, "resize_long_side must be at least 0"),
        ("padding", -1, "padding must be at least 0"),
        ("background_threshold", -1.0, "background_threshold must be at least 0.0"),
        ("alpha_threshold", 256.0, "alpha_threshold must be between 0.0 and 255.0"),
        ("mask_blur", -0.1, "mask_blur must be at least 0.0"),
        ("simplify", -0.1, "simplify must be at least 0.0"),
        ("contour_smooth", -1, "contour_smooth must be at least 0"),
        ("curve_spacing", -0.1, "curve_spacing must be at least 0.0"),
        ("corner_angle", 181.0, "corner_angle must be between 0.0 and 180.0"),
        ("corner_radius", -0.1, "corner_radius must be at least 0.0"),
        ("corner_rounding", -1, "corner_rounding must be at least 0"),
        ("curve_fit_error", -0.1, "curve_fit_error must be at least 0.0"),
        ("min_area", -0.1, "min_area must be at least 0.0"),
        ("seed", -1, "seed must be between 0 and 2147483647"),
        ("seed", 2147483648, "seed must be between 0 and 2147483647"),
    ],
)
def test_trace_rejects_invalid_option_ranges(option: str, value: float, message: str) -> None:
    source = Path("examples/keel-compressed.jpg")
    options = TraceOptions(**{option: value})

    with pytest.raises(ValueError, match=message):
        trace_image(source, options)


def test_smooth_closed_contour_dampens_boundary_jitter() -> None:
    mask = np.zeros((80, 80), dtype=np.uint8)
    points = np.array(
        [
            [15, 12],
            [18, 24],
            [14, 36],
            [18, 48],
            [15, 64],
            [58, 64],
            [58, 12],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(mask, [points], 255)
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)

    raw = contour.reshape(-1, 2).astype(np.float32)
    smoothed = _smooth_closed_contour(contour, 15).reshape(-1, 2)

    assert len(smoothed) == len(raw)
    assert _total_turn(smoothed) < _total_turn(raw)


def test_thin_contours_use_smaller_smoothing_window() -> None:
    mask = np.zeros((80, 80), dtype=np.uint8)
    cv2.rectangle(mask, (10, 36), (70, 40), 255, -1)
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    window = _contour_smooth_window(contours[0], 21)

    assert 3 <= window < 21
    assert window % 2 == 1


def test_broad_contours_keep_requested_smoothing_window() -> None:
    mask = np.zeros((80, 80), dtype=np.uint8)
    cv2.rectangle(mask, (10, 10), (70, 70), 255, -1)
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    assert _contour_smooth_window(contours[0], 21) == 21


def test_shallow_bends_keep_curve_handles() -> None:
    points = np.array(
        [
            [0.0, 0.0],
            [30.0, 6.0],
            [45.0, 34.0],
            [18.0, 60.0],
            [-8.0, 28.0],
        ]
    )

    path = _closed_catmull_rom_path(points, corner_angle=100.0)

    assert "C 30 6 30 6 30 6" not in path


def test_rounded_path_offsets_corners() -> None:
    points = np.array(
        [
            [0.0, 0.0],
            [60.0, 0.0],
            [60.0, 60.0],
            [0.0, 60.0],
        ]
    )

    path = _rounded_closed_path(points, radius=10.0)

    assert "Q 60 0 60 7.07" in path
    assert "Q 60 60 52.93 60" in path
    assert "L 52.93 0" in path


def test_rounded_points_feed_cubic_spline_without_line_segments() -> None:
    points = np.array(
        [
            [0.0, 0.0],
            [60.0, 0.0],
            [60.0, 60.0],
            [0.0, 60.0],
        ]
    )

    rounded = _rounded_closed_points(points, radius=10.0)
    path = _closed_catmull_rom_path(rounded, corner_angle=0.0)

    assert len(rounded) == 8
    assert "C " in path
    assert "L " not in path
    assert "Q " not in path


def test_resample_closed_points_adds_even_curve_anchors() -> None:
    points = np.array(
        [
            [0.0, 0.0],
            [80.0, 0.0],
            [80.0, 20.0],
            [0.0, 20.0],
        ]
    )

    resampled = _resample_closed_points(points, spacing=10.0)
    lengths = np.linalg.norm(np.roll(resampled, -1, axis=0) - resampled, axis=1)

    assert len(resampled) == 20
    assert np.max(lengths) <= 10.5


def test_bezier_fit_emits_only_cubic_segments_for_smooth_contours() -> None:
    angles = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    points = np.column_stack(
        [
            50.0 + np.cos(angles) * (25.0 + np.sin(angles * 5.0) * 0.5),
            50.0 + np.sin(angles) * 18.0,
        ]
    )

    path = _closed_bezier_fit_path(points, max_error=1.2)

    assert path.startswith("M ")
    assert "C " in path
    assert "L " not in path
    assert "Q " not in path


def test_trace_preserves_background_holes(tmp_path: Path) -> None:
    image_path = tmp_path / "ring.png"
    mask = np.zeros((120, 120, 3), dtype=np.uint8)
    cv2.circle(mask, (60, 60), 42, (54, 215, 212), -1)
    cv2.circle(mask, (60, 60), 18, (0, 0, 0), -1)
    cv2.imwrite(str(image_path), cv2.cvtColor(mask, cv2.COLOR_RGB2BGR))

    svg = trace_image(
        image_path,
        TraceOptions(
            groups=1,
            resize_long_side=120,
            palette=["#36d7d4"],
            simplify=2.0,
            min_area=50,
        ),
    )

    assert 'fill-rule="evenodd"' in svg
    assert svg.count("M ") >= 2


def test_trace_uses_alpha_foreground_for_transparent_black_artwork(tmp_path: Path) -> None:
    image_path = tmp_path / "transparent-black.png"
    image = np.zeros((80, 80, 4), dtype=np.uint8)
    cv2.rectangle(image, (20, 18), (60, 62), (0, 0, 0, 255), -1)
    cv2.circle(image, (24, 24), 10, (54, 215, 212, 255), -1)
    cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA))

    svg = trace_image(
        image_path,
        TraceOptions(
            groups=2,
            resize_long_side=80,
            palette=["#36d7d4", "#111111"],
            simplify=1.0,
            min_area=50,
        ),
    )

    assert '<g id="shape-group-1" fill="#36d7d4" fill-rule="evenodd">' in svg
    assert '<g id="shape-group-2" fill="#111111" fill-rule="evenodd">' in svg
    assert svg.count("<path") == 2


def _total_turn(points: np.ndarray) -> float:
    previous = np.roll(points, 1, axis=0)
    current = points
    following = np.roll(points, -1, axis=0)
    return float(np.abs(previous - (2 * current) + following).sum())
