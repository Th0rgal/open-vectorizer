from pathlib import Path

import pytest
from PIL import Image

from tools.compare_render import _render_svg, main


def test_render_svg_reports_missing_rsvg_convert(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", "")

    with pytest.raises(RuntimeError, match="rsvg-convert is required"):
        _render_svg(Path("input.svg"), Path("output.png"), 16, 16)


def test_compare_render_cli_reports_missing_rsvg_convert(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    original = tmp_path / "original.png"
    svg = tmp_path / "trace.svg"
    Image.new("RGBA", (1, 1), (255, 255, 255, 255)).save(original)
    svg.write_text("<svg/>", encoding="utf-8")
    monkeypatch.setenv("PATH", "")

    with pytest.raises(SystemExit) as exc_info:
        main([str(original), str(svg)])

    assert exc_info.value.code == 2
    assert "rsvg-convert is required" in capsys.readouterr().err
