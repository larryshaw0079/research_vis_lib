from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.cartoon_stacked_bar import (
    BAR_IDS,
    DEFAULT_DATA,
    DEFAULT_STYLE,
    GROUPS,
    PALETTES,
    SERIES,
    TARGET_TOTALS,
    create_figure,
    palette_from_selector,
    render_palette,
)


class CartoonStackedBarTests(unittest.TestCase):
    def test_reference_structure_is_complete(self) -> None:
        DEFAULT_DATA.validate()
        DEFAULT_STYLE.validate()
        self.assertEqual(len(SERIES), 9)
        self.assertEqual(len(BAR_IDS), 9)
        self.assertEqual(len(GROUPS), 3)
        self.assertEqual(len(PALETTES), 16)
        self.assertEqual(len({palette.colors for palette in PALETTES}), 16)
        self.assertEqual(len({palette.name for palette in PALETTES}), 16)
        for bar_id, total in TARGET_TOTALS.items():
            self.assertAlmostEqual(DEFAULT_DATA.labelled_total(bar_id), total, places=6)

    def test_palette_selector_accepts_number_and_name(self) -> None:
        self.assertEqual(palette_from_selector("1"), (1, PALETTES[0]))
        self.assertEqual(palette_from_selector(16), (16, PALETTES[15]))
        self.assertEqual(palette_from_selector("pacific_sunset"), (11, PALETTES[10]))

    def test_additive_baselines_match_the_labelled_totals(self) -> None:
        a = DEFAULT_DATA.labelled_total("policy_a")
        b = DEFAULT_DATA.labelled_total("policy_b")
        c = DEFAULT_DATA.labelled_total("policy_c")
        self.assertEqual(a, 48)
        self.assertEqual(b, 54)
        self.assertEqual(c, 60)
        self.assertEqual(DEFAULT_DATA.labelled_total("dual_unrelated"), 78)
        self.assertEqual(DEFAULT_DATA.labelled_total("dual_synergies"), 86)
        self.assertEqual(DEFAULT_DATA.labelled_total("multi_unrelated"), 114)
        self.assertEqual(DEFAULT_DATA.labelled_total("multi_synergies"), 130)
        self.assertEqual(a + b + c, 162)
        self.assertIn("overlap_cyan", DEFAULT_DATA.bars["dual_synergies"])
        self.assertNotIn("overlap_cyan", DEFAULT_DATA.bars["dual_unrelated"])
        self.assertIn("tradeoffs_missing", DEFAULT_DATA.bars["multi_tradeoffs"])
        self.assertEqual(DEFAULT_DATA.bars["multi_tradeoffs"]["tradeoffs_missing"], 24)

    def test_create_figure_uses_reference_canvas(self) -> None:
        figure = create_figure(PALETTES[0])
        try:
            self.assertEqual(tuple(figure.get_size_inches()), (12.774, 7.2))
            self.assertEqual(len(figure.axes), 1)
            ax = figure.axes[0]
            np.testing.assert_allclose(ax.get_xlim(), (-0.85, 11.45))
            np.testing.assert_allclose(ax.get_ylim(), (-18.0, 168.0))
            self.assertEqual(ax.get_ylabel(), "Effect Score")
            self.assertEqual(list(ax.get_yticks()), [0.0, 30.0, 60.0, 90.0, 120.0, 150.0])
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
