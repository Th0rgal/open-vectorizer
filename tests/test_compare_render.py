from pathlib import Path

import pytest

from tools.compare_render import _render_svg


def test_render_svg_reports_missing_rsvg_convert(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", "")

    with pytest.raises(RuntimeError, match="rsvg-convert is required"):
        _render_svg(Path("input.svg"), Path("output.png"), 16, 16)
