from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class TraceOptions:
    groups: int = 2
    resize_long_side: int = 1200
    crop: bool = True
    padding: int = 32
    background_threshold: float = 8.0
    simplify: float = 3.2
    contour_smooth: int = 15
    corner_angle: float = 0.0
    corner_radius: float = 0.0
    corner_rounding: int = 1
    min_area: float = 18.0
    palette: list[str] | None = None
    alpha_threshold: float = 8.0


def trace_image(path: str | Path, options: TraceOptions | None = None) -> str:
    opts = options or TraceOptions()
    rgb, alpha = _load_rgba(path, opts.resize_long_side)
    background = _estimate_background(rgb)
    has_alpha = bool(np.any(alpha < 255))
    foreground = _foreground_mask(
        rgb, background, opts.background_threshold, alpha, opts.alpha_threshold
    )
    masks, colors = _cluster_foreground(
        rgb,
        foreground,
        background,
        opts.background_threshold,
        opts.groups,
        opts.palette,
        has_alpha,
    )

    if opts.crop:
        x0, y0, x1, y1 = _mask_bbox(foreground, opts.padding, rgb.shape[1], rgb.shape[0])
    else:
        x0, y0, x1, y1 = 0, 0, rgb.shape[1], rgb.shape[0]

    width, height = x1 - x0, y1 - y0
    paths_by_color: list[tuple[str, list[str]]] = []
    for mask, color in zip(masks, colors, strict=True):
        clipped = mask[y0:y1, x0:x1]
        paths = _mask_to_paths(
            clipped,
            opts.simplify,
            opts.min_area,
            opts.contour_smooth,
            opts.corner_angle,
            opts.corner_radius,
            opts.corner_rounding,
        )
        if paths:
            paths_by_color.append((color, paths))

    return _svg(width, height, paths_by_color)


def _load_rgb(path: str | Path, resize_long_side: int) -> np.ndarray:
    rgb, _alpha = _load_rgba(path, resize_long_side)
    return rgb


def _load_rgba(path: str | Path, resize_long_side: int) -> tuple[np.ndarray, np.ndarray]:
    image = Image.open(path).convert("RGBA")
    if resize_long_side and max(image.size) > resize_long_side:
        scale = resize_long_side / max(image.size)
        image = image.resize(
            (round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS
        )
    rgba = np.asarray(image)
    return rgba[:, :, :3], rgba[:, :, 3]


def _estimate_background(rgb: np.ndarray) -> np.ndarray:
    border = np.concatenate(
        [
            rgb[0, :, :],
            rgb[-1, :, :],
            rgb[:, 0, :],
            rgb[:, -1, :],
        ],
        axis=0,
    )
    return np.median(border.astype(np.float32), axis=0)


def _foreground_mask(
    rgb: np.ndarray,
    background: np.ndarray,
    threshold: float,
    alpha: np.ndarray | None = None,
    alpha_threshold: float = 8.0,
) -> np.ndarray:
    if alpha is not None and np.any(alpha < 255):
        mask = (alpha.astype(np.float32) > alpha_threshold).astype(np.uint8) * 255
    else:
        distance = np.linalg.norm(rgb.astype(np.float32) - background, axis=2)
        mask = (distance > threshold).astype(np.uint8) * 255
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def _cluster_foreground(
    rgb: np.ndarray,
    foreground: np.ndarray,
    background: np.ndarray,
    background_threshold: float,
    groups: int,
    palette: list[str] | None,
    alpha_foreground: bool = False,
) -> tuple[list[np.ndarray], list[str]]:
    pixels = rgb[foreground > 0].reshape(-1, 3).astype(np.float32)
    if len(pixels) == 0:
        return [], []

    groups = max(1, min(groups, len(pixels)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 120, 0.1)
    _compactness, labels, centers = cv2.kmeans(
        pixels, groups, None, criteria, 16, cv2.KMEANS_PP_CENTERS
    )

    full = np.full(rgb.shape[:2], -1, dtype=np.int32)
    full[foreground > 0] = labels.reshape(-1)

    # Saturated color groups first, dark neutral groups second. This gives stable SVG layering
    # for logo artwork where dark strokes sit over or near chromatic fills.
    hsv_centers = cv2.cvtColor(centers.reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_RGB2HSV)[0]
    order = sorted(
        range(groups),
        key=lambda i: (int(hsv_centers[i, 1] < 40), int(centers[i].mean())),
    )

    masks: list[np.ndarray] = []
    colors: list[str] = []
    for cluster_index in order:
        if (
            not alpha_foreground
            and groups > 1
            and _rgb_distance(centers[cluster_index], background)
            <= max(background_threshold * 4.0, background_threshold + 12.0)
        ):
            continue
        mask = (full == cluster_index).astype(np.uint8) * 255
        mask = _clean_mask(mask)
        if np.count_nonzero(mask) == 0:
            continue
        masks.append(mask)
        if palette and len(colors) < len(palette):
            colors.append(palette[len(colors)])
        else:
            colors.append(_hex(centers[cluster_index]))
    return masks, colors


def _rgb_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left.astype(np.float32) - right.astype(np.float32)))


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def _mask_bbox(mask: np.ndarray, padding: int, width: int, height: int) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0, 0, width, height
    return (
        max(0, int(xs.min()) - padding),
        max(0, int(ys.min()) - padding),
        min(width, int(xs.max()) + padding + 1),
        min(height, int(ys.max()) + padding + 1),
    )


def _mask_to_paths(
    mask: np.ndarray,
    simplify: float,
    min_area: float,
    contour_smooth: int,
    corner_angle: float,
    corner_radius: float,
    corner_rounding: int,
) -> list[str]:
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        return []

    contour_hierarchy = hierarchy[0]
    paths: list[str] = []
    external_indices = [
        index for index, contour_info in enumerate(contour_hierarchy) if contour_info[3] == -1
    ]
    for contour_index in sorted(
        external_indices, key=lambda index: cv2.contourArea(contours[index]), reverse=True
    ):
        contour = contours[contour_index]
        area = cv2.contourArea(contour)
        if area < min_area or len(contour) < 8:
            continue
        subpaths = [
            _contour_to_path(
                contour,
                simplify,
                contour_smooth,
                corner_angle,
                corner_radius,
                corner_rounding,
            )
        ]
        child_index = contour_hierarchy[contour_index][2]
        while child_index != -1:
            child = contours[child_index]
            if cv2.contourArea(child) >= min_area and len(child) >= 8:
                subpaths.append(
                    _contour_to_path(
                        child,
                        simplify,
                        contour_smooth,
                        corner_angle,
                        corner_radius,
                        corner_rounding,
                    )
                )
            child_index = contour_hierarchy[child_index][0]
        subpaths = [path for path in subpaths if path]
        if not subpaths:
            continue
        paths.append(" ".join(subpaths))
    return paths


def _contour_to_path(
    contour: np.ndarray,
    simplify: float,
    contour_smooth: int,
    corner_angle: float,
    corner_radius: float,
    corner_rounding: int,
) -> str:
    contour = _smooth_closed_contour(
        contour, _contour_smooth_window(contour, contour_smooth), corner_angle
    )
    approx = cv2.approxPolyDP(contour, simplify, closed=True).reshape(-1, 2).astype(float)
    if len(approx) < 4:
        return ""
    if corner_radius > 0:
        approx = _rounded_closed_points(approx, corner_radius)
    approx = _chaikin_closed(approx, corner_rounding)
    return _closed_catmull_rom_path(approx, corner_angle)


def _contour_smooth_window(contour: np.ndarray, requested: int) -> int:
    if requested <= 1 or len(contour) < 8:
        return requested

    perimeter = cv2.arcLength(contour, closed=True)
    if perimeter == 0.0:
        return requested

    stroke_width = 2.0 * cv2.contourArea(contour) / perimeter
    if stroke_width >= requested:
        return requested

    capped = max(3, int(round(stroke_width * 0.5)))
    if capped % 2 == 0:
        capped += 1
    return min(requested, capped)


def _smooth_closed_contour(
    contour: np.ndarray, window: int, corner_angle: float = 100.0
) -> np.ndarray:
    points = contour.reshape(-1, 2).astype(np.float32)
    if window <= 1 or len(points) < 8:
        return points.reshape(-1, 1, 2)

    if window % 2 == 0:
        window += 1
    window = min(window, len(points) - 1 if len(points) % 2 == 0 else len(points))
    if window < 3:
        return points.reshape(-1, 1, 2)

    pad = window // 2
    kernel = np.hanning(window).astype(np.float32)
    if float(kernel.sum()) == 0.0:
        kernel = np.ones(window, dtype=np.float32)
    kernel /= kernel.sum()

    wrapped = np.vstack([points[-pad:], points, points[:pad]])
    xs = np.convolve(wrapped[:, 0], kernel, mode="valid")
    ys = np.convolve(wrapped[:, 1], kernel, mode="valid")
    smoothed = np.column_stack([xs, ys]).astype(np.float32)

    corner_indices = _sharp_corner_indices(contour, window, corner_angle)
    if corner_indices:
        smoothed = _restore_corner_neighborhoods(points, smoothed, corner_indices, pad)

    return smoothed.reshape(-1, 1, 2)


def _sharp_corner_indices(contour: np.ndarray, window: int, corner_angle: float) -> list[int]:
    epsilon = max(2.0, window / 3.0)
    approx = cv2.approxPolyDP(contour, epsilon, closed=True).reshape(-1, 2).astype(np.float32)
    if len(approx) < 3:
        return []

    points = contour.reshape(-1, 2).astype(np.float32)
    corner_indices: list[int] = []
    for index, current in enumerate(approx):
        previous = approx[(index - 1) % len(approx)]
        following = approx[(index + 1) % len(approx)]
        if _corner_angle(previous, current, following) <= corner_angle:
            distances = np.linalg.norm(points - current, axis=1)
            corner_indices.append(int(np.argmin(distances)))
    return corner_indices


def _corner_angle(previous: np.ndarray, current: np.ndarray, following: np.ndarray) -> float:
    left = previous - current
    right = following - current
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return 180.0
    cosine = float(np.dot(left, right) / denominator)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _restore_corner_neighborhoods(
    original: np.ndarray, smoothed: np.ndarray, corner_indices: list[int], radius: int
) -> np.ndarray:
    if radius <= 0:
        return smoothed

    result = smoothed.copy()
    count = len(original)
    indices = np.arange(count)
    distances = np.full(count, count, dtype=np.float32)
    for corner_index in corner_indices:
        cyclic = np.abs(indices - corner_index)
        distances = np.minimum(distances, np.minimum(cyclic, count - cyclic))

    affected = distances < radius
    weights = np.ones(count, dtype=np.float32)
    weights[affected] = distances[affected] / radius
    result[affected] = (
        original[affected] * (1.0 - weights[affected, None])
        + smoothed[affected] * weights[affected, None]
    )
    return result


def _closed_catmull_rom_path(points: np.ndarray, corner_angle: float) -> str:
    p = points
    parts = [f"M {_fmt(p[0, 0])} {_fmt(p[0, 1])}"]
    n = len(p)
    tension = 0.7
    sharp = [
        _corner_angle(p[(index - 1) % n], p[index], p[(index + 1) % n]) <= corner_angle
        for index in range(n)
    ]
    for i in range(n):
        p0 = p[(i - 1) % n]
        p1 = p[i]
        p2 = p[(i + 1) % n]
        p3 = p[(i + 2) % n]
        c1 = p1 + (p2 - p0) * (tension / 6.0)
        c2 = p2 - (p3 - p1) * (tension / 6.0)
        if sharp[i]:
            c1 = p1
        if sharp[(i + 1) % n]:
            c2 = p2
        parts.append(
            "C "
            f"{_fmt(c1[0])} {_fmt(c1[1])} "
            f"{_fmt(c2[0])} {_fmt(c2[1])} "
            f"{_fmt(p2[0])} {_fmt(p2[1])}"
        )
    parts.append("Z")
    return " ".join(parts)


def _chaikin_closed(points: np.ndarray, iterations: int, ratio: float = 0.22) -> np.ndarray:
    if iterations <= 0 or len(points) < 4:
        return points

    result = points.astype(float)
    for _ in range(iterations):
        rounded: list[np.ndarray] = []
        for index in range(len(result)):
            current = result[index]
            following = result[(index + 1) % len(result)]
            rounded.append(current * (1.0 - ratio) + following * ratio)
            rounded.append(current * ratio + following * (1.0 - ratio))
        result = np.array(rounded, dtype=float)
    return result


def _rounded_closed_points(points: np.ndarray, radius: float) -> np.ndarray:
    p = points.astype(float)
    if radius <= 0 or len(p) < 4:
        return p

    rounded: list[np.ndarray] = []
    n = len(p)
    for index in range(n):
        previous = p[(index - 1) % n]
        current = p[index]
        following = p[(index + 1) % n]

        incoming = previous - current
        outgoing = following - current
        incoming_length = float(np.linalg.norm(incoming))
        outgoing_length = float(np.linalg.norm(outgoing))
        if incoming_length == 0.0 or outgoing_length == 0.0:
            rounded.append(current)
            continue

        angle = _corner_angle(previous, current, following)
        if angle > 175.0:
            rounded.append(current)
            continue

        angle_scale = max(0.35, min(1.0, np.sin(np.radians(angle / 2.0))))
        cut = min(radius * angle_scale, incoming_length * 0.45, outgoing_length * 0.45)
        if cut < 0.5:
            rounded.append(current)
            continue

        rounded.append(current + (incoming / incoming_length) * cut)
        rounded.append(current + (outgoing / outgoing_length) * cut)

    return np.array(rounded, dtype=float)


def _rounded_closed_path(points: np.ndarray, radius: float) -> str:
    p = points.astype(float)
    n = len(p)
    entries = np.empty_like(p)
    exits = np.empty_like(p)

    for index in range(n):
        previous = p[(index - 1) % n]
        current = p[index]
        following = p[(index + 1) % n]

        incoming = previous - current
        outgoing = following - current
        incoming_length = float(np.linalg.norm(incoming))
        outgoing_length = float(np.linalg.norm(outgoing))
        if incoming_length == 0.0 or outgoing_length == 0.0:
            entries[index] = current
            exits[index] = current
            continue

        angle = _corner_angle(previous, current, following)
        if angle > 175.0:
            entries[index] = current
            exits[index] = current
            continue

        # Clip the radius to local edge lengths so small features are rounded without
        # swallowing the shape. Sharper angles get a slightly shorter cut.
        angle_scale = max(0.35, min(1.0, np.sin(np.radians(angle / 2.0))))
        cut = min(radius * angle_scale, incoming_length * 0.45, outgoing_length * 0.45)
        if cut < 0.5:
            entries[index] = current
            exits[index] = current
            continue

        entries[index] = current + (incoming / incoming_length) * cut
        exits[index] = current + (outgoing / outgoing_length) * cut

    parts = [f"M {_fmt(exits[0, 0])} {_fmt(exits[0, 1])}"]
    for index in range(n):
        following_index = (index + 1) % n
        entry = entries[following_index]
        corner = p[following_index]
        exit_point = exits[following_index]
        if np.linalg.norm(entry - p[index]) > 0.1:
            parts.append(f"L {_fmt(entry[0])} {_fmt(entry[1])}")
        parts.append(
            "Q "
            f"{_fmt(corner[0])} {_fmt(corner[1])} "
            f"{_fmt(exit_point[0])} {_fmt(exit_point[1])}"
        )
    parts.append("Z")
    return " ".join(parts)


def _svg(width: int, height: int, paths_by_color: list[tuple[str, list[str]]]) -> str:
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        'role="img" aria-label="Vectorized artwork">',
        '  <title>Vectorized artwork</title>',
    ]
    for index, (color, paths) in enumerate(paths_by_color, start=1):
        lines.append(f'  <g id="shape-group-{index}" fill="{escape(color)}" fill-rule="evenodd">')
        for path in paths:
            lines.append(f'    <path d="{path}"/>')
        lines.append("  </g>")
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _hex(rgb: np.ndarray) -> str:
    r, g, b = np.clip(np.round(rgb), 0, 255).astype(int)
    return f"#{r:02x}{g:02x}{b:02x}"


def _fmt(value: float) -> str:
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return text if text != "-0" else "0"
