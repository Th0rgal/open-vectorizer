from pathlib import Path

import cv2
import numpy as np

from open_vectorizer import TraceOptions, trace_image
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
