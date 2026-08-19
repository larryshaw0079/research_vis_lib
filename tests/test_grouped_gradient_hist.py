from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.grouped_gradient_hist import (
    BIN_EDGES,
    BIN_LABELS,
    DEFAULT_DATA,
    DEFAULT_STYLE,
    GROUPS,
    GROUP_SIZES,
    PALETTES,
    TARGET_COUNTS,
    TARGET_STATS,
    create_figure,
    format_legend_label,
    histogram_counts,
    palette_from_selector,
    render_palette,
)


class GroupedGradientHistTests(unittest.TestCase):
    def test_reference_structure_is_complete(self) -> None:
        DEFAULT_DATA.validate()
        DEFAULT_STYLE.validate()
        self.assertEqual(len(GROUPS), 3)
        self.assertEqual(len(BIN_EDGES), 15)
        self.assertEqual(len(BIN_LABELS), 14)
        self.assertEqual(len(PALETTES), 18)
        self.assertEqual(len({palette.colors for palette in PALETTES}), 18)
        self.assertEqual(len({palette.name for palette in PALETTES}), 18)
        for group in GROUPS:
            mean, std = DEFAULT_DATA.stats(group)
            target_mean, target_std = TARGET_STATS[group]
            self.assertEqual(DEFAULT_DATA.sample(group).size, GROUP_SIZES[group])
            self.assertEqual(tuple(histogram_counts(DEFAULT_DATA.sample(group))), TARGET_COUNTS[group])
            self.assertAlmostEqual(mean, target_mean, places=6)
            self.assertAlmostEqual(std, target_std, places=6)

    def test_palette_selector_accepts_number_and_name(self) -> None:
        self.assertEqual(palette_from_selector("1"), (1, PALETTES[0]))
        self.assertEqual(palette_from_selector(18), (18, PALETTES[17]))
        self.assertEqual(
            palette_from_selector("crimson_navy_amber"),
            (4, PALETTES[3]),
        )

    def test_histogram_uses_the_published_interval_edges(self) -> None:
        counts = histogram_counts(DEFAULT_DATA.sample("Zhang et al."))
        self.assertEqual(counts.shape, (14,))
        self.assertGreater(int(counts[6]), int(counts[0]))
        self.assertEqual(BIN_LABELS[6], "[-15.3, 0.0)")
        self.assertEqual(BIN_LABELS[-1], "[91.6, 106.8)")

    def test_legend_label_prints_mean_and_std(self) -> None:
        label = format_legend_label("Xue et al.", -3.3, 10.6)
        self.assertIn("Xue et al.", label)
        self.assertIn("-3.3", label)
        self.assertIn("10.6", label)

    def test_create_figure_uses_reference_canvas(self) -> None:
        figure = create_figure(PALETTES[0])
        try:
            self.assertEqual(tuple(figure.get_size_inches()), (8.484, 7.048))
            self.assertEqual(len(figure.axes), 1)
            ax = figure.axes[0]
            self.assertTrue(ax.axison)
            np.testing.assert_allclose(ax.get_ylim(), (0.0, 50.0))
            self.assertIn("Residual Group", ax.get_xlabel())
            self.assertEqual(ax.get_ylabel(), "Frequency")
            self.assertEqual(len(ax.get_xticklabels()), 14)
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
