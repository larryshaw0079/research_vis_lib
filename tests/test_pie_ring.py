from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.pie_ring import (
    DEFAULT_DATA,
    DEFAULT_STYLE,
    FUELS,
    PALETTES,
    PIE_ORDER,
    SECTORS,
    create_figure,
    palette_from_selector,
    render_palette,
)


class PieRingTests(unittest.TestCase):
    def test_reference_structure_is_complete(self) -> None:
        DEFAULT_DATA.validate()
        DEFAULT_STYLE.validate()
        self.assertEqual(len(FUELS), 7)
        self.assertEqual(len(SECTORS), 6)
        self.assertEqual(len(PIE_ORDER), 7)
        self.assertEqual(len(PALETTES), 18)
        self.assertEqual(len({palette.colors for palette in PALETTES}), 18)
        self.assertEqual(len({palette.name for palette in PALETTES}), 18)
        self.assertAlmostEqual(sum(DEFAULT_DATA.pie_percentages.values()), 100.0)
        self.assertEqual(DEFAULT_DATA.sector_total("Electricity"), 8.0)
        self.assertEqual(DEFAULT_DATA.sector_total("Industrial"), 6.0)
        self.assertEqual(DEFAULT_DATA.sector_total("Road"), 4.2)
        self.assertAlmostEqual(DEFAULT_DATA.sector_total("Buildings"), 2.7)
        self.assertAlmostEqual(DEFAULT_DATA.sector_total("Residential fuel"), 0.7)
        self.assertAlmostEqual(DEFAULT_DATA.sector_total("Non-road"), 0.1)
        self.assertEqual(DEFAULT_DATA.pie_percentages["Coal"], 52.0)
        self.assertEqual(DEFAULT_DATA.pie_values()[0], 52.0)

    def test_palette_selector_accepts_number_and_name(self) -> None:
        self.assertEqual(palette_from_selector("1"), (1, PALETTES[0]))
        self.assertEqual(palette_from_selector(18), (18, PALETTES[17]))
        self.assertEqual(palette_from_selector("gold_navy"), (7, PALETTES[6]))

    def test_energy_angles_run_clockwise_from_noon(self) -> None:
        self.assertAlmostEqual(DEFAULT_STYLE.energy_to_theta(0.0), 90.0)
        self.assertAlmostEqual(DEFAULT_STYLE.energy_to_theta(8.0), -180.0)
        self.assertGreater(
            DEFAULT_STYLE.energy_to_theta(0.0),
            DEFAULT_STYLE.energy_to_theta(2.0),
        )

    def test_stacked_segments_follow_legend_order(self) -> None:
        fuels = [fuel for fuel, _value in DEFAULT_DATA.stacked("Electricity")]
        self.assertEqual(
            fuels,
            ["Gas", "Coal", "Biomass", "Wind", "Solar", "Electricity"],
        )

    def test_create_figure_uses_reference_canvas(self) -> None:
        figure = create_figure(PALETTES[0])
        try:
            self.assertEqual(tuple(figure.get_size_inches()), (11.85, 12.45))
            self.assertEqual(len(figure.axes), 1)
            self.assertFalse(figure.axes[0].axison)
            np.testing.assert_allclose(figure.axes[0].get_xlim(), (-1185.0, 1185.0))
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
