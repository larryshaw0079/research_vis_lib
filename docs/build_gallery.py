"""Regenerate ``docs/gallery`` from the current templates.

Run with ``python docs/build_gallery.py``. Each template contributes palette 1 of
its reference figure, scaled to a common width so the README table stays tidy.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")

from rvl.registry import load_registry
from rvl.render import save_figure

GALLERY = REPO_ROOT / "docs" / "gallery"
TARGET_WIDTH_PIXELS = 900


def main() -> int:
    registry = load_registry()
    GALLERY.mkdir(parents=True, exist_ok=True)
    for spec in registry:
        module = registry.module(spec.template_id)
        figure = module.create_figure(palette=module.PALETTES[0])
        dpi = max(40, round(TARGET_WIDTH_PIXELS / figure.get_size_inches()[0]))
        written = save_figure(
            figure,
            GALLERY,
            spec.template_id.replace("-", "_"),
            formats=("png",),
            dpi=dpi,
        )
        for path in written:
            print(f"{path.relative_to(REPO_ROOT)}  ({dpi} dpi)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
