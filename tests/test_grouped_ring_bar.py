from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.grouped_ring_bar import (
    DATASETS,
    DEFAULT_DATA,
    DEFAULT_STYLE,
    MODELS,
    PALETTES,
    create_figure,
    palette_from_selector,
    render_palette,
    _inverted_outers,
)


class GroupedRingBarTests(unittest.TestCase):
    def test_reference_structure_is_complete(self) -> None:
        DEFAULT_DATA.validate()
        DEFAULT_STYLE.validate()
        self.assertEqual(len(DATASETS), 8)
        self.assertEqual(len(MODELS), 5)
        self.assertEqual(len(PALETTES), 18)
        self.assertEqual(len({palette.colors for palette in PALETTES}), 18)
        self.assertEqual(len({palette.name for palette in PALETTES}), 18)
        for dataset in DATASETS:
            self.assertEqual(set(DEFAULT_DATA.mse[dataset]), set(MODELS))

    def test_palette_selector_accepts_number_and_name(self) -> None:
        self.assertEqual(palette_from_selector("1"), (1, PALETTES[0]))
        self.assertEqual(palette_from_selector(10), (10, PALETTES[9]))
        self.assertEqual(
            palette_from_selector("dusk_sunset"),
            (18, PALETTES[17]),
        )

    def test_lower_scores_become_longer_bars_within_each_dataset(self) -> None:
        for dataset in DATASETS:
            values = [DEFAULT_DATA.mse[dataset][model] for model in MODELS]
            outers = _inverted_outers(values, DEFAULT_STYLE)
            self.assertAlmostEqual(float(outers.max()), DEFAULT_STYLE.max_outer_radius)
            self.assertAlmostEqual(float(outers.min()), DEFAULT_STYLE.min_outer_radius)
            order = np.argsort(values)
            self.assertGreaterEqual(outers[order[0]], outers[order[-1]])
            self.assertAlmostEqual(
                float(outers[order[0]]), DEFAULT_STYLE.max_outer_radius
            )
            self.assertAlmostEqual(
                float(outers[order[-1]]), DEFAULT_STYLE.min_outer_radius
            )

    def test_create_figure_uses_reference_canvas(self) -> None:
        figure = create_figure(PALETTES[0])
        try:
            self.assertEqual(tuple(figure.get_size_inches()), (10.404, 10.404))
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
