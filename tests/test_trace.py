from pathlib import Path

from open_vectorizer import TraceOptions, trace_image


def test_trace_outputs_grouped_svg(tmp_path: Path) -> None:
    source = Path("examples/keel-compressed.jpg")
    svg = trace_image(
        source,
        TraceOptions(
            groups=2,
            resize_long_side=600,
            palette=["#36d7d4", "#111111"],
            simplify=3.0,
        ),
    )

    assert '<g id="shape-group-1" fill="#36d7d4">' in svg
    assert '<g id="shape-group-2" fill="#111111">' in svg
    assert svg.count("<path") >= 2

