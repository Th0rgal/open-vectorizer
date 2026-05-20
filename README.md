# Open Vectorizer

Open Vectorizer is an open source raster-to-SVG tracer for low-color artwork, logos, and icons.
The Python package ships a `py.typed` marker, so type checkers can read its inline annotations.
It is built around the same practical stages used by high-quality commercial tracers:

1. estimate and remove the background from border pixels,
2. cluster foreground pixels into a small number of color groups,
3. clean each group with morphology,
4. optionally fair binary masks to remove compression stair-steps,
5. extract contours,
6. smooth and optionally simplify boundaries,
7. resample long outlines at even arc-length spacing,
8. offset corner vertices into rounded control points,
9. fit smooth cubic SVG paths around the entire contour.

The first target is clean logo reconstruction: a small number of semantic shape groups with elegant
curves instead of thousands of pixel-like fragments.

## Install

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .
```

For development tools:

```bash
pip install -e '.[dev]'
pytest
ruff check .
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for contributor workflow details.

Check the installed CLI version with:

```bash
open-vectorizer --version
```

Python callers can read the same package metadata from `open_vectorizer.__version__`.
Use `-` as the output path to write SVG to stdout for shell pipelines.

## Example

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

The same tracer can be called from Python:

```python
from pathlib import Path

from open_vectorizer import TraceOptions, trace_image

svg = trace_image(
    "examples/keel-compressed.jpg",
    TraceOptions(
        groups=2,
        palette=["#c77832", "#171717"],
        dark_palette=["#f0a45b", "#f4ead8"],
        title="Keel mark",
    ),
)
Path("examples/keel.svg").write_text(svg, encoding="utf-8")
```

This produces two SVG groups with a warm light palette and an embedded dark-mode palette. The SVG
uses CSS custom properties plus `@media (prefers-color-scheme: dark)`, so the same file can adapt
from copper and soft black on light backgrounds to amber and ivory on dark backgrounds.
For the included `keel-compressed.jpg` example, the result is three total paths: one warm blade and
two black strokes. The generated paths use cubic Bezier segments throughout, including the broad
blade edges.

Palette entries must be hex colors in `#rgb` or `#rrggbb` form. Short colors are expanded and
all palette values are normalized to lowercase before they are written into SVG attributes or theme
CSS.

## CLI Options

The tracing controls are:

- `--groups`: expected foreground color groups before SVG emission.
- `--resize`: longest side used for tracing; use `0` to keep the original size.
- `--padding`: transparent crop padding around the foreground bounding box.
- `--no-crop`: keep the original image viewBox instead of cropping to foreground.
- `--palette`: comma-separated light-mode hex fills, assigned in traced group order.
- `--dark-palette`: optional dark-mode hex fills emitted through CSS custom properties.
- `--background`: optional hex background override when the image border is not reliable.
- `--title`: accessible SVG `<title>` and `aria-label` text.
- `--seed`: OpenCV k-means RNG seed for reproducible color grouping.
- `--threshold`: RGB distance from the estimated border background required for foreground.
- `--alpha-threshold`: alpha cutoff used when tracing transparent artwork.
- `--mask-blur`: Gaussian sigma for fairing binary masks before contour extraction.
- `--simplify`: Ramer-Douglas-Peucker contour simplification in pixels.
- `--contour-smooth`: wrapped smoothing window for dampening raster stair-steps.
- `--curve-spacing`: arc-length spacing for resampling long contours before fitting.
- `--corner-radius`: optional rounded-corner offset before cubic fitting.
- `--corner-angle`: angles at or below this value keep crisp curve handles.
- `--corner-rounding`: Chaikin corner-cut iterations before spline fitting.
- `--curve-fit-error`: recursive cubic Bezier fit tolerance; set `0` for Catmull-Rom output.
- `--min-area`: connected-component area floor for ignoring specks and compression debris.

Numeric options are validated before tracing starts. Counts, distances, areas, and smoothing
settings must be non-negative, `--groups` must be at least `1`, `--alpha-threshold` must be between
`0` and `255`, `--corner-angle` must be between `0` and `180`, and `--seed` must fit in OpenCV's
signed 32-bit RNG seed range.

## Render Comparison

`tools/compare_render.py` can render an SVG with `rsvg-convert` and compare it to a source image:

```bash
python tools/compare_render.py examples/keel-compressed.jpg examples/keel.svg \
  --render /tmp/keel-render.png
```

It reports RGB/RGBA mean absolute error, RGB RMSE, alpha error, and max absolute channel error as
JSON. Install `librsvg2-bin` or an equivalent package first so `rsvg-convert` is available.

## Why This Approach

Vector Magic appears to use an automatic full-color tracing pipeline with image-type detection,
color reduction, anti-alias-aware edge placement, curve fitting, and manual cleanup tools. Public
documentation describes the user-visible behavior, but not the proprietary implementation details.

Open Vectorizer starts with the strongest open building blocks for this class of image:

- color quantization for semantic groups,
- contour tracing over cleaned masks,
- Ramer-Douglas-Peucker simplification,
- arc-length resampling so long design curves get enough spline anchors,
- radius-based corner offsetting before curve fitting,
- Chaikin corner cutting for additional smoothing,
- recursive least-squares cubic Bezier fitting for compact, editable SVG paths.

Future work should add optional centerline tracing, better topology handling for holes, palette
locking, and a browser UI with live threshold/simplification controls.
See [docs/LIMITATIONS.md](docs/LIMITATIONS.md) for current fit, weak spots, and tuning tips.

## License

MIT
