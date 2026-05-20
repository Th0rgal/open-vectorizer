"""Open raster-to-SVG vectorization tools."""

from importlib.metadata import PackageNotFoundError, version

from .trace import TraceOptions, trace_image

try:
    __version__ = version("open-vectorizer")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = ["TraceOptions", "__version__", "trace_image"]
