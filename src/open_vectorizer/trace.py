from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from xml.sax.saxutils import escape, quoteattr

import cv2
import numpy as np
from PIL import Image

_HEX_COLOR_RE = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")


@dataclass(frozen=True)
class TraceOptions:
    groups: int = 2
    resize_long_side: int = 1200
    crop: bool = True
    padding: int = 32
    background_threshold: float = 8.0
    simplify: float = 3.2
    contour_smooth: int = 15
    curve_spacing: float = 0.0
    corner_angle: float = 0.0
    corner_radius: float = 0.0
    corner_rounding: int = 1
    curve_fit_error: float = 1.2
    min_area: float = 18.0
    palette: list[str] | None = None
    dark_palette: list[str] | None = None
    background_color: str | None = None
    title: str = "Vectorized artwork"
    alpha_threshold: float = 8.0
    mask_blur: float = 0.0


def trace_image(path: str | Path, options: TraceOptions | None = None) -> str:
    opts = options or TraceOptions()
    _validate_options(opts)
    palette = _normalize_palette(opts.palette, "palette")
    dark_palette = _normalize_palette(opts.dark_palette, "dark_palette")
    background_color = _normalize_optional_color(opts.background_color, "background_color")
    rgb, alpha = _load_rgba(path, opts.resize_long_side)
    background = _hex_to_rgb(background_color) if background_color else _estimate_background(rgb)
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
        palette,
        has_alpha,
        opts.mask_blur,
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
            opts.curve_spacing,
            opts.corner_angle,
            opts.corner_radius,
            opts.corner_rounding,
            opts.curve_fit_error,
        )
        if paths:
            paths_by_color.append((color, paths))

    return _svg(width, height, paths_by_color, dark_palette, opts.title)


def _normalize_palette(palette: list[str] | None, name: str) -> list[str] | None:
    if palette is None:
        return None
    return [_normalize_hex_color(color, name) for color in palette]


def _normalize_optional_color(color: str | None, name: str) -> str | None:
    if color is None:
        return None
    return _normalize_hex_color(color, name)


def _normalize_hex_color(color: str, name: str) -> str:
    cleaned = color.strip()
    if not _HEX_COLOR_RE.fullmatch(cleaned):
        raise ValueError(f"{name} colors must be hex values like #111 or #111111: {color!r}")
    if len(cleaned) == 4:
        cleaned = "#" + "".join(channel * 2 for channel in cleaned[1:])
    return cleaned.lower()


def _hex_to_rgb(color: str) -> np.ndarray:
    return np.array(
        [int(color[index : index + 2], 16) for index in range(1, 7, 2)],
        dtype=np.float32,
    )


def _validate_options(options: TraceOptions) -> None:
    _require_at_least("groups", options.groups, 1)
    _require_at_least("resize_long_side", options.resize_long_side, 0)
    _require_at_least("padding", options.padding, 0)
    _require_at_least("background_threshold", options.background_threshold, 0.0)
    _require_at_least("simplify", options.simplify, 0.0)
    _require_at_least("contour_smooth", options.contour_smooth, 0)
    _require_at_least("curve_spacing", options.curve_spacing, 0.0)
    _require_at_least("corner_radius", options.corner_radius, 0.0)
    _require_at_least("corner_rounding", options.corner_rounding, 0)
    _require_at_least("curve_fit_error", options.curve_fit_error, 0.0)
    _require_at_least("min_area", options.min_area, 0.0)
    _require_between("corner_angle", options.corner_angle, 0.0, 180.0)
    _require_between("alpha_threshold", options.alpha_threshold, 0.0, 255.0)
    _require_at_least("mask_blur", options.mask_blur, 0.0)


def _require_at_least(name: str, value: float, minimum: float) -> None:
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")


def _require_between(name: str, value: float, minimum: float, maximum: float) -> None:
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")


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
    mask_blur: float = 0.0,
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
        mask = _clean_mask(mask, mask_blur)
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


def _clean_mask(mask: np.ndarray, blur_sigma: float = 0.0) -> np.ndarray:
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    if blur_sigma > 0:
        blurred = cv2.GaussianBlur(mask, (0, 0), blur_sigma)
        _threshold, mask = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    return mask


def _mask_bbox(
    mask: np.ndarray, padding: int, width: int, height: int
) -> tuple[int, int, int, int]:
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
    curve_spacing: float,
    corner_angle: float,
    corner_radius: float,
    corner_rounding: int,
    curve_fit_error: float,
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
                curve_spacing,
                corner_angle,
                corner_radius,
                corner_rounding,
                curve_fit_error,
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
                        curve_spacing,
                        corner_angle,
                        corner_radius,
                        corner_rounding,
                        curve_fit_error,
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
    curve_spacing: float,
    corner_angle: float,
    corner_radius: float,
    corner_rounding: int,
    curve_fit_error: float,
) -> str:
    contour = _smooth_closed_contour(
        contour, _contour_smooth_window(contour, contour_smooth), corner_angle
    )
    if curve_spacing > 0:
        approx = _resample_closed_points(contour.reshape(-1, 2).astype(float), curve_spacing)
    else:
        approx = cv2.approxPolyDP(contour, simplify, closed=True).reshape(-1, 2).astype(float)
    if len(approx) < 4:
        return ""
    if corner_radius > 0:
        approx = _rounded_closed_points(approx, corner_radius)
    approx = _chaikin_closed(approx, corner_rounding)
    if curve_fit_error > 0:
        return _closed_bezier_fit_path(approx, curve_fit_error)
    return _closed_catmull_rom_path(approx, corner_angle)


def _resample_closed_points(points: np.ndarray, spacing: float) -> np.ndarray:
    p = points.astype(float)
    if spacing <= 0 or len(p) < 4:
        return p

    following = np.roll(p, -1, axis=0)
    lengths = np.linalg.norm(following - p, axis=1)
    perimeter = float(lengths.sum())
    if perimeter == 0.0:
        return p

    sample_count = max(4, int(round(perimeter / spacing)))
    targets = np.linspace(0.0, perimeter, sample_count, endpoint=False)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    result: list[np.ndarray] = []
    segment_index = 0
    for target in targets:
        while segment_index < len(lengths) - 1 and cumulative[segment_index + 1] <= target:
            segment_index += 1
        length = lengths[segment_index]
        if length == 0.0:
            result.append(p[segment_index])
            continue
        ratio = (target - cumulative[segment_index]) / length
        result.append(p[segment_index] * (1.0 - ratio) + following[segment_index] * ratio)
    return np.array(result, dtype=float)


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
            f"C {_fmt(c1[0])} {_fmt(c1[1])} {_fmt(c2[0])} {_fmt(c2[1])} {_fmt(p2[0])} {_fmt(p2[1])}"
        )
    parts.append("Z")
    return " ".join(parts)


def _closed_bezier_fit_path(points: np.ndarray, max_error: float) -> str:
    p = _remove_near_duplicate_points(points.astype(float), min_distance=0.25)
    if len(p) < 4:
        return ""

    start = _lowest_curvature_index(p)
    p = np.roll(p, -start, axis=0)
    open_points = np.vstack([p, p[0]])
    tangent = _unit_vector(open_points[1] - open_points[-2])
    if np.linalg.norm(tangent) == 0.0:
        tangent = _unit_vector(open_points[1] - open_points[0])

    curves = _fit_cubic_sequence(open_points, max_error, tangent, tangent, depth=0)
    parts = [f"M {_fmt(open_points[0, 0])} {_fmt(open_points[0, 1])}"]
    for curve in curves:
        parts.append(
            "C "
            f"{_fmt(curve[1, 0])} {_fmt(curve[1, 1])} "
            f"{_fmt(curve[2, 0])} {_fmt(curve[2, 1])} "
            f"{_fmt(curve[3, 0])} {_fmt(curve[3, 1])}"
        )
    parts.append("Z")
    return " ".join(parts)


def _remove_near_duplicate_points(points: np.ndarray, min_distance: float) -> np.ndarray:
    if len(points) < 2:
        return points

    kept = [points[0]]
    for point in points[1:]:
        if np.linalg.norm(point - kept[-1]) >= min_distance:
            kept.append(point)
    if len(kept) > 1 and np.linalg.norm(kept[0] - kept[-1]) < min_distance:
        kept.pop()
    return np.array(kept, dtype=float)


def _lowest_curvature_index(points: np.ndarray) -> int:
    if len(points) < 4:
        return 0

    best_index = 0
    best_score = float("inf")
    for index in range(len(points)):
        previous = points[(index - 1) % len(points)]
        current = points[index]
        following = points[(index + 1) % len(points)]
        turn = abs(180.0 - _corner_angle(previous, current, following))
        span = np.linalg.norm(following - previous)
        score = turn / max(float(span), 1.0)
        if score < best_score:
            best_score = score
            best_index = index
    return best_index


def _fit_cubic_sequence(
    points: np.ndarray,
    max_error: float,
    left_tangent: np.ndarray,
    right_tangent: np.ndarray,
    depth: int,
) -> list[np.ndarray]:
    if len(points) == 2:
        return [_fallback_cubic(points[0], points[1], left_tangent, right_tangent)]

    parameters = _chord_parameters(points)
    curve = _generate_cubic(points, parameters, left_tangent, right_tangent)
    split_index, error = _max_bezier_error(points, curve, parameters)
    if error <= max_error**2 or depth >= 24:
        return [curve]

    center_tangent = _center_tangent(points, split_index)
    left = _fit_cubic_sequence(
        points[: split_index + 1],
        max_error,
        left_tangent,
        center_tangent,
        depth + 1,
    )
    right = _fit_cubic_sequence(
        points[split_index:],
        max_error,
        -center_tangent,
        right_tangent,
        depth + 1,
    )
    return left + right


def _generate_cubic(
    points: np.ndarray,
    parameters: np.ndarray,
    left_tangent: np.ndarray,
    right_tangent: np.ndarray,
) -> np.ndarray:
    p0 = points[0]
    p3 = points[-1]
    c = np.zeros((2, 2), dtype=float)
    x = np.zeros(2, dtype=float)

    for point, u in zip(points, parameters, strict=True):
        b0, b1, b2, b3 = _bernstein3(float(u))
        a1 = left_tangent * b1
        a2 = right_tangent * b2
        tmp = point - ((p0 * (b0 + b1)) + (p3 * (b2 + b3)))
        c[0, 0] += float(np.dot(a1, a1))
        c[0, 1] += float(np.dot(a1, a2))
        c[1, 0] = c[0, 1]
        c[1, 1] += float(np.dot(a2, a2))
        x[0] += float(np.dot(a1, tmp))
        x[1] += float(np.dot(a2, tmp))

    det = (c[0, 0] * c[1, 1]) - (c[0, 1] * c[1, 0])
    segment_length = float(np.linalg.norm(p3 - p0))
    epsilon = 1.0e-6 * segment_length
    if abs(det) > 1.0e-12:
        alpha_left = ((x[0] * c[1, 1]) - (x[1] * c[0, 1])) / det
        alpha_right = ((c[0, 0] * x[1]) - (c[1, 0] * x[0])) / det
    else:
        alpha_left = alpha_right = segment_length / 3.0

    max_alpha = max(segment_length * 1.75, epsilon)
    if (
        alpha_left < epsilon
        or alpha_right < epsilon
        or alpha_left > max_alpha
        or alpha_right > max_alpha
    ):
        return _fallback_cubic(p0, p3, left_tangent, right_tangent)

    return np.array(
        [
            p0,
            p0 + (left_tangent * alpha_left),
            p3 + (right_tangent * alpha_right),
            p3,
        ],
        dtype=float,
    )


def _fallback_cubic(
    p0: np.ndarray, p3: np.ndarray, left_tangent: np.ndarray, right_tangent: np.ndarray
) -> np.ndarray:
    distance = float(np.linalg.norm(p3 - p0)) / 3.0
    return np.array([p0, p0 + left_tangent * distance, p3 + right_tangent * distance, p3])


def _chord_parameters(points: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total = float(lengths.sum())
    if total == 0.0:
        return np.linspace(0.0, 1.0, len(points))
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    return cumulative / total


def _max_bezier_error(
    points: np.ndarray, curve: np.ndarray, parameters: np.ndarray
) -> tuple[int, float]:
    split_index = max(1, len(points) // 2)
    max_error = -1.0
    for index in range(1, len(points) - 1):
        projected = _bezier_point(curve, float(parameters[index]))
        error = float(np.sum((projected - points[index]) ** 2))
        if error > max_error:
            max_error = error
            split_index = index
    return split_index, max_error


def _center_tangent(points: np.ndarray, center_index: int) -> np.ndarray:
    previous = points[max(0, center_index - 1)]
    following = points[min(len(points) - 1, center_index + 1)]
    tangent = _unit_vector(following - previous)
    if np.linalg.norm(tangent) == 0.0:
        tangent = _unit_vector(points[-1] - points[0])
    return tangent


def _unit_vector(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length == 0.0:
        return np.zeros(2, dtype=float)
    return vector / length


def _bernstein3(u: float) -> tuple[float, float, float, float]:
    inverse = 1.0 - u
    return inverse**3, 3.0 * u * inverse**2, 3.0 * u**2 * inverse, u**3


def _bezier_point(curve: np.ndarray, u: float) -> np.ndarray:
    b0, b1, b2, b3 = _bernstein3(u)
    return (curve[0] * b0) + (curve[1] * b1) + (curve[2] * b2) + (curve[3] * b3)


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
            f"Q {_fmt(corner[0])} {_fmt(corner[1])} {_fmt(exit_point[0])} {_fmt(exit_point[1])}"
        )
    parts.append("Z")
    return " ".join(parts)


def _svg(
    width: int,
    height: int,
    paths_by_color: list[tuple[str, list[str]]],
    dark_palette: list[str] | None = None,
    title: str = "Vectorized artwork",
) -> str:
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'role="img" aria-label={quoteattr(title)}>',
        f"  <title>{escape(title)}</title>",
    ]
    if dark_palette:
        lines.extend(_theme_style(paths_by_color, dark_palette))
    for index, (color, paths) in enumerate(paths_by_color, start=1):
        if dark_palette:
            lines.append(
                f'  <g id="shape-group-{index}" class="shape-group shape-group-{index}" '
                'fill-rule="evenodd">'
            )
        else:
            lines.append(
                f'  <g id="shape-group-{index}" fill="{escape(color)}" fill-rule="evenodd">'
            )
        for path in paths:
            lines.append(f'    <path d="{path}"/>')
        lines.append("  </g>")
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _theme_style(paths_by_color: list[tuple[str, list[str]]], dark_palette: list[str]) -> list[str]:
    lines = ["  <style>"]
    lines.append("    :root {")
    for index, (color, _paths) in enumerate(paths_by_color, start=1):
        lines.append(f"      --ov-group-{index}: {color};")
    lines.append("    }")
    for index, (color, _paths) in enumerate(paths_by_color, start=1):
        lines.append(f"    .shape-group-{index} {{ fill: var(--ov-group-{index}, {color}); }}")
    lines.append("    @media (prefers-color-scheme: dark) {")
    lines.append("      :root {")
    for index, color in enumerate(dark_palette[: len(paths_by_color)], start=1):
        lines.append(f"        --ov-group-{index}: {color};")
    lines.append("      }")
    lines.append("    }")
    lines.append("  </style>")
    return lines


def _hex(rgb: np.ndarray) -> str:
    r, g, b = np.clip(np.round(rgb), 0, 255).astype(int)
    return f"#{r:02x}{g:02x}{b:02x}"


def _fmt(value: float) -> str:
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return text if text != "-0" else "0"
