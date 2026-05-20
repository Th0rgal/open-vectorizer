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
    min_area: float = 18.0
    palette: list[str] | None = None


def trace_image(path: str | Path, options: TraceOptions | None = None) -> str:
    opts = options or TraceOptions()
    rgb = _load_rgb(path, opts.resize_long_side)
    background = _estimate_background(rgb)
    foreground = _foreground_mask(rgb, background, opts.background_threshold)
    masks, colors = _cluster_foreground(rgb, foreground, opts.groups, opts.palette)

    if opts.crop:
        x0, y0, x1, y1 = _mask_bbox(foreground, opts.padding, rgb.shape[1], rgb.shape[0])
    else:
        x0, y0, x1, y1 = 0, 0, rgb.shape[1], rgb.shape[0]

    width, height = x1 - x0, y1 - y0
    paths_by_color: list[tuple[str, list[str]]] = []
    for mask, color in zip(masks, colors, strict=True):
        clipped = mask[y0:y1, x0:x1]
        paths = _mask_to_paths(clipped, opts.simplify, opts.min_area)
        if paths:
            paths_by_color.append((color, paths))

    return _svg(width, height, paths_by_color)


def _load_rgb(path: str | Path, resize_long_side: int) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if resize_long_side and max(image.size) > resize_long_side:
        scale = resize_long_side / max(image.size)
        image = image.resize(
            (round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS
        )
    return np.asarray(image)


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


def _foreground_mask(rgb: np.ndarray, background: np.ndarray, threshold: float) -> np.ndarray:
    distance = np.linalg.norm(rgb.astype(np.float32) - background, axis=2)
    mask = (distance > threshold).astype(np.uint8) * 255
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def _cluster_foreground(
    rgb: np.ndarray, foreground: np.ndarray, groups: int, palette: list[str] | None
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
    for out_index, cluster_index in enumerate(order):
        mask = (full == cluster_index).astype(np.uint8) * 255
        mask = _clean_mask(mask)
        if np.count_nonzero(mask) == 0:
            continue
        masks.append(mask)
        if palette and out_index < len(palette):
            colors.append(palette[out_index])
        else:
            colors.append(_hex(centers[cluster_index]))
    return masks, colors


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


def _mask_to_paths(mask: np.ndarray, simplify: float, min_area: float) -> list[str]:
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    paths: list[str] = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(contour)
        if area < min_area or len(contour) < 8:
            continue
        approx = cv2.approxPolyDP(contour, simplify, closed=True).reshape(-1, 2).astype(float)
        if len(approx) < 4:
            continue
        paths.append(_closed_catmull_rom_path(approx))
    return paths


def _closed_catmull_rom_path(points: np.ndarray) -> str:
    p = points
    parts = [f"M {_fmt(p[0, 0])} {_fmt(p[0, 1])}"]
    n = len(p)
    tension = 0.42
    for i in range(n):
        p0 = p[(i - 1) % n]
        p1 = p[i]
        p2 = p[(i + 1) % n]
        p3 = p[(i + 2) % n]
        c1 = p1 + (p2 - p0) * (tension / 6.0)
        c2 = p2 - (p3 - p1) * (tension / 6.0)
        parts.append(
            "C "
            f"{_fmt(c1[0])} {_fmt(c1[1])} "
            f"{_fmt(c2[0])} {_fmt(c2[1])} "
            f"{_fmt(p2[0])} {_fmt(p2[1])}"
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
        lines.append(f'  <g id="shape-group-{index}" fill="{escape(color)}">')
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
