# Algorithm Notes

Vector Magic publicly describes a high-quality tracing flow for logo artwork: choose settings for
the image type, constrain colors when needed, place edges through anti-aliased boundaries, fit
smooth curves, and provide editing controls for cleanup. The exact implementation is proprietary,
so Open Vectorizer follows an open, reproducible pipeline with comparable stages for low-color
logos.

## Current Pipeline

1. Load and optionally resize the raster image.
2. Estimate the background color from border pixels.
3. Build a foreground mask from alpha for transparent inputs, otherwise by measuring RGB distance
   from the background.
4. Cluster foreground pixels into semantic color groups with k-means.
5. Clean each group mask with morphological close/open passes.
6. Optionally fair binary masks with a light Gaussian blur and re-threshold. This removes
   compression stair-steps before they become real curve bends.
7. Extract contour hierarchies for each group, including background-colored holes.
8. Smooth each closed contour with a wrapped Hann window to reduce JPEG stair-steps before fitting
   curves, capping the window on thin contours so narrow strokes keep their coverage.
9. Remove small contours with `--min-area`.
10. Simplify contours with Ramer-Douglas-Peucker, or resample smoothed contours at even arc-length
   spacing when the artwork has long shallow curves.
11. Offset simplified/resampled contour corners into rounded point pairs so sharp raster vertices get real
    radius without creating straight line SVG segments.
12. Apply optional Chaikin smoothing to the rounded points.
13. Convert the rounded contours to closed cubic Bezier paths. The default fitter recursively
    approximates point runs with least-squares cubic Beziers, which keeps smooth logo curves from
    inheriting every small raster jitter as a visible bend. Catmull-Rom fitting is still available
    by setting `--curve-fit-error 0`.
14. Emit one SVG `<g>` per color group with even-odd fill for compound paths. When a dark palette
    is supplied, the SVG embeds CSS custom properties and a `prefers-color-scheme: dark` media
    query so one asset can adapt to light and dark UI themes.

## Keel Experiment

The uploaded image has a pure black background, two real foreground colors, and a small amount of
JPEG compression/anti-alias noise. The cleanest result came from keeping only larger connected
components:

```bash
open-vectorizer examples/keel-compressed.jpg examples/keel.svg \
  --groups 2 \
  --palette '#c77832,#171717' \
  --dark-palette '#f0a45b,#f4ead8' \
  --resize 1200 \
  --mask-blur 1.0 \
  --simplify 1.98 \
  --contour-smooth 25 \
  --curve-spacing 16 \
  --corner-angle 60 \
  --corner-radius 3.25 \
  --corner-rounding 1 \
  --curve-fit-error 1.5 \
  --threshold 8 \
  --min-area 1000
```

The output has two shape groups and three total paths:

- `shape-group-1`: copper blade in light mode, amber blade in dark mode
- `shape-group-2`: soft black mast and horizontal stroke in light mode, warm ivory in dark mode

The keel's left edge is especially sensitive to JPEG stair-step noise because the contour is long,
thin, and shallowly curved. Tracing the raw binary boundary can leave visible segment-to-segment
changes after cubic fitting. The mask fairing and contour smoothing passes dampen those one-pixel
wiggles before fitting, while the adaptive smoothing cap avoids eroding thin black strokes. A naive
corner-radius pass made the corners round but left straight `L` segments between them. The current
approach instead resamples the smoothed contour, rounds raster corners, then fits approximating
cubic Beziers. That gives long shallow edges continuous SVG curvature without forcing the path
through every noisy contour point.

## Future Work

- Palette locking from user-picked swatches.
- Hole-aware contour hierarchy.
- Centerline tracing for calligraphic strokes.
- Alternative curve fitting based on Potrace's polygon optimization.
- Browser UI with live preview and per-group controls.
- Optional ML segmentation for noisy photos and scans.
