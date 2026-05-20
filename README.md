# Open Vectorizer

Open Vectorizer is an open source raster-to-SVG tracer for low-color artwork, logos, and icons.
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

## Example

```bash
open-vectorizer examples/keel-compressed.jpg examples/keel.svg \
  --groups 2 \
  --palette '#36d7d4,#111111' \
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

This produces two SVG groups: one teal group and one black group, on a transparent background.
For the included `keel-compressed.jpg` example, the result is three total paths: one teal blade and
two black strokes. The generated paths use cubic Bezier segments throughout, including the broad
teal blade edges.

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

## License

MIT
