# Changelog

## Unreleased

- Added palette validation and normalization for light and dark SVG palettes.
- Added range validation for tracer and CLI numeric options.
- Added explicit background color overrides for border-touching artwork.
- Added custom SVG titles and aria labels.
- Added deterministic color clustering with a bounded OpenCV RNG seed.
- Added stdout SVG output with `open-vectorizer input.png -`.
- Added package version reporting through `open-vectorizer --version` and
  `open_vectorizer.__version__`.
- Added GitHub Actions CI on Python 3.10 and 3.12.
- Added development extras, package metadata, pytest configuration, and a `py.typed` marker.
- Improved render comparison errors when `rsvg-convert` is missing.
- Expanded README, contributor, algorithm, and limitations documentation.
