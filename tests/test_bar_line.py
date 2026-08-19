from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.bar_line import (
    DATES,
    DEFAULT_DATA,
    DEFAULT_STYLE,
    GROUPS,
    PALETTES,
    PERIOD_BOUNDARIES,
    TARGET_FLUX,
    TARGET_REPLICATES,
    create_figure,
    naive_cumulative,
    palette_from_selector,
    render_palette,
    replicate_sem,
)


class BarLineTests(unittest.TestCase):
    def test_reference_structure_is_complete(self) -> None:
        DEFAULT_DATA.validate()
        DEFAULT_STYLE.validate()
        self.assertEqual(len(GROUPS), 2)
        self.assertEqual(len(DATES), 11)
        self.assertEqual(len(PALETTES), 18)
        self.assertEqual(len({palette.colors for palette in PALETTES}), 18)
        self.assertEqual(len({palette.name for palette in PALETTES}), 18)
        self.assertEqual(PERIOD_BOUNDARIES, (4.0, 6.0))
        for group in GROUPS:
            flux = DEFAULT_DATA.flux_series(group)
            mean, sem = DEFAULT_DATA.bar_stats(group)
            self.assertEqual(flux.shape, (11,))
            self.assertEqual(tuple(flux), TARGET_FLUX[group])
            self.assertAlmostEqual(mean, naive_cumulative(flux), places=6)
            self.assertAlmostEqual(sem, replicate_sem(TARGET_REPLICATES[group]), places=6)
            self.assertGreater(mean, 0.0)
            self.assertGreater(sem, 0.0)
        self.assertGreater(
            DEFAULT_DATA.bar_stats("HF")[0],
            DEFAULT_DATA.bar_stats("LF")[0],
        )
        self.assertGreater(
            float(DEFAULT_DATA.flux_series("HF")[8]),
            float(DEFAULT_DATA.flux_series("LF")[8]),
        )

    def test_palette_selector_accepts_number_and_name(self) -> None:
        self.assertEqual(palette_from_selector("1"), (1, PALETTES[0]))
        self.assertEqual(palette_from_selector(18), (18, PALETTES[17]))
        self.assertEqual(palette_from_selector("navy_amber"), (11, PALETTES[10]))

    def test_naive_cumulative_matches_the_source_shortcut(self) -> None:
        self.assertEqual(naive_cumulative(TARGET_FLUX["LF"]), 303.0)
        self.assertEqual(naive_cumulative(TARGET_FLUX["HF"]), 467.0)

    def test_create_figure_uses_reference_canvas(self) -> None:
        figure = create_figure(PALETTES[0])
        try:
            self.assertEqual(tuple(figure.get_size_inches()), (8.4, 7.2))
            self.assertEqual(len(figure.axes), 2)
            ax_bar, ax_line = figure.axes
            np.testing.assert_allclose(ax_bar.get_xlim(), (0.0, 640.0))
            np.testing.assert_allclose(ax_line.get_ylim(), (0.0, 165.0))
            self.assertIn("cumulative emission", ax_bar.get_xlabel())
            self.assertIn("emission flux", ax_line.get_ylabel())
            self.assertEqual([label.get_text() for label in ax_bar.get_yticklabels()], ["LF", "HF"])
            self.assertEqual(len(ax_line.get_xticklabels()), 11)
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
