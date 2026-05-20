# Contributing

## Setup

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
```

## Checks

Run the same local gates used by CI:

```bash
pytest
ruff check .
```

CI runs those checks on Python 3.10 and 3.12 for every push to `main` and every pull request.

## Tracing Changes

When changing trace behavior, add or update tests that cover the specific raster condition. Prefer
small synthetic images in `tmp_path` for edge cases, and keep the `examples/keel-compressed.jpg`
fixture for regression checks against the main logo workflow.

For CLI changes, cover both parser validation and `main(argv=...)` behavior so console entry-point
logic remains testable without mutating process-global `sys.argv`.

## Generated Files

Do not commit virtual environments, caches, build directories, or local render outputs. The
`.gitignore` already excludes the common paths.
