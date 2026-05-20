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
6. Extract contour hierarchies for each group, including background-colored holes.
7. Smooth each closed contour with a wrapped Hann window to reduce JPEG stair-steps before fitting
   curves, capping the window on thin contours so narrow strokes keep their coverage.
8. Remove small contours with `--min-area`.
9. Simplify contours with Ramer-Douglas-Peucker.
10. Offset simplified polygon corners into rounded point pairs so sharp raster vertices get real
    radius without creating straight line SVG segments.
11. Apply optional Chaikin smoothing to the rounded points.
12. Convert the rounded contours to closed cubic Bezier paths using Catmull-Rom control points.
13. Emit one SVG `<g>` per color group with even-odd fill for compound paths.

## Keel Experiment

The uploaded image has a pure black background, two real foreground colors, and a small amount of
JPEG compression/anti-alias noise. The cleanest result came from keeping only larger connected
components:

```bash
open-vectorizer examples/keel-compressed.jpg examples/keel.svg \
  --groups 2 \
  --palette '#36d7d4,#111111' \
  --resize 1200 \
  --simplify 1.98 \
  --contour-smooth 15 \
  --corner-angle 60 \
  --corner-radius 3.25 \
  --corner-rounding 1 \
  --threshold 8 \
  --min-area 1000
```

The output has two shape groups and three total paths:

- `shape-group-1`: teal blade
- `shape-group-2`: black mast and horizontal stroke

The keel's left edge is especially sensitive to JPEG stair-step noise because the contour is long,
thin, and shallowly curved. Tracing the raw binary boundary can leave visible segment-to-segment
changes after cubic fitting. The contour smoothing pass dampens those one-pixel wiggles before
simplification, while the adaptive smoothing cap avoids eroding thin black strokes. A naive
corner-radius pass made the corners round but left straight `L` segments between them. The current
approach instead offsets each corner into rounded points and then fits cubic Bezier curves around
the whole contour, so the broad teal edges use SVG curvature instead of straight chords.

## Future Work

- Palette locking from user-picked swatches.
- Hole-aware contour hierarchy.
- Centerline tracing for calligraphic strokes.
- Alternative curve fitting based on Potrace's polygon optimization.
- Browser UI with live preview and per-group controls.
- Optional ML segmentation for noisy photos and scans.
