from pathlib import Path

import cv2
import numpy as np

from open_vectorizer import TraceOptions, trace_image
from open_vectorizer.trace import _closed_catmull_rom_path, _rounded_closed_path, _smooth_closed_contour


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


def _total_turn(points: np.ndarray) -> float:
    previous = np.roll(points, 1, axis=0)
    current = points
    following = np.roll(points, -1, axis=0)
    return float(np.abs(previous - (2 * current) + following).sum())
