from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.pie_3d import (
    CATEGORIES,
    DEFAULT_DATA,
    DEFAULT_STYLE,
    PALETTES,
    create_figure,
    palette_from_selector,
    render_palette,
    _slice_angles,
)


class Pie3DTests(unittest.TestCase):
    def test_reference_structure_is_complete(self) -> None:
        DEFAULT_DATA.validate()
        DEFAULT_STYLE.validate()
        self.assertEqual(len(CATEGORIES), 15)
        self.assertEqual(len(PALETTES), 18)
        self.assertEqual(len({palette.colors for palette in PALETTES}), 18)
        self.assertEqual(len({palette.name for palette in PALETTES}), 18)
        self.assertAlmostEqual(sum(DEFAULT_DATA.percentages.values()), 100.2, places=5)
        for category in CATEGORIES:
            self.assertIn(category, DEFAULT_DATA.percentages)
        self.assertEqual(len(PALETTES[0].colors), 15)

    def test_palette_selector_accepts_number_and_name(self) -> None:
        self.assertEqual(palette_from_selector("1"), (1, PALETTES[0]))
        self.assertEqual(palette_from_selector(18), (18, PALETTES[17]))
        self.assertEqual(palette_from_selector("spectral_rainbow"), (7, PALETTES[6]))

    def test_slices_start_at_noon_and_run_counterclockwise(self) -> None:
        edges = _slice_angles(DEFAULT_DATA.values())
        self.assertAlmostEqual(float(edges[0]), 0.5 * np.pi)
        self.assertAlmostEqual(float(edges[-1]), 2.5 * np.pi)
        self.assertTrue(np.all(np.diff(edges) > 0))
        widest = CATEGORIES[int(np.argmax(DEFAULT_DATA.values()))]
        self.assertEqual(widest, "NDWI")

    def test_create_figure_uses_reference_canvas(self) -> None:
        figure = create_figure(PALETTES[0])
        try:
            self.assertEqual(tuple(figure.get_size_inches()), (10.095, 6.75))
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
