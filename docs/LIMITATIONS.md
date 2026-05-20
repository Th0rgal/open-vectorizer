# Limitations

Open Vectorizer is currently tuned for low-color artwork, logos, icons, and simple illustrations.
It is not a general photo vectorizer.

## Works Best

- Flat-color or low-color raster artwork.
- Logos and marks with clear foreground/background separation.
- Transparent PNG artwork where alpha already defines the foreground.
- Inputs where a small number of semantic color groups is known ahead of time.

## Current Weak Spots

- Photographs and textured artwork create too many color regions for the current k-means pipeline.
- Heavy shadows, gradients, and antialiasing may need manual palette and threshold tuning.
- Deeply nested holes or overlapping groups can expose contour-topology edge cases.
- Thin calligraphic strokes are traced as filled outlines, not centerlines.
- Automatic palette selection is still basic; user-supplied palettes produce more predictable output.

## Practical Tips

- Use `--background` when artwork touches the image border and border-based background estimation is
  unreliable.
- Use `--palette` and `--dark-palette` to lock brand colors instead of accepting k-means centers.
- Increase `--min-area` to remove compression specks.
- Use `--mask-blur` and `--contour-smooth` to reduce JPEG stair-step noise before curve fitting.
- Use `--curve-spacing` for long shallow curves that need more spline anchors.
