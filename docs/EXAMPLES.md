# Example Recipes

## Logo With Locked Brand Colors

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

## Artwork Touching The Border

Border-based background estimation can fail when the foreground reaches every edge. Supply the
known background color:

```bash
open-vectorizer badge.png badge.svg \
  --groups 1 \
  --background '#ffffff' \
  --palette '#36d7d4' \
  --padding 0
```

## Transparent PNG Artwork

Transparent inputs use alpha as the foreground mask. This works well for icons exported from design
tools:

```bash
open-vectorizer transparent-icon.png transparent-icon.svg \
  --groups 3 \
  --alpha-threshold 16 \
  --palette '#36d7d4,#171717,#c77832'
```

## Shell Pipelines

Use `-` as the output path when another command should consume the SVG:

```bash
open-vectorizer icon.png - --groups 2 --palette '#36d7d4,#111111' > icon.svg
```
