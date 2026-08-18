from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt

from radial_line import (
    DEFAULT_DATA,
    FEATURES,
    FEATURE_GROUPS,
    FEATURE_SLOTS,
    PALETTES,
    create_figure,
    palette_from_selector,
    render_palette,
)


class RadialLineTests(unittest.TestCase):
    def test_reference_structure_is_complete(self) -> None:
        DEFAULT_DATA.validate()
        self.assertEqual(len(FEATURES), 64)
        self.assertEqual(sum(len(group.features) for group in FEATURE_GROUPS), 64)
        self.assertEqual(len(FEATURE_SLOTS), 64)
        self.assertEqual(len(set(FEATURE_SLOTS)), 64)
        self.assertEqual(len(PALETTES), 18)
        self.assertEqual(len({palette.colors for palette in PALETTES}), 18)

    def test_palette_selector_accepts_number_and_name(self) -> None:
        self.assertEqual(palette_from_selector("1"), (1, PALETTES[0]))
        self.assertEqual(
            palette_from_selector("category20b_red_purple"),
            (18, PALETTES[17]),
        )

    def test_create_figure_uses_reference_canvas(self) -> None:
        figure = create_figure(PALETTES[0])
        try:
            self.assertEqual(tuple(figure.get_size_inches()), (10.84, 10.8))
            self.assertEqual(len(figure.axes), 1)
            self.assertFalse(figure.axes[0].axison)
        finally:
            plt.close(figure)

    def test_render_palette_writes_raster_and_vector_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = render_palette(
                1,
                PALETTES[0],
                Path(directory),
                formats=("png", "svg"),
                dpi=20,
            )
            self.assertEqual([path.suffix for path in paths], [".png", ".svg"])
            self.assertTrue(all(path.stat().st_size > 0 for path in paths))


if __name__ == "__main__":
    unittest.main()
