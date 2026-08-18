from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.grouped_lm_marginal import (
    DEFAULT_DATA,
    DEFAULT_STYLE,
    GROUPS,
    PALETTES,
    TARGET_FITS,
    create_figure,
    fit_ols,
    format_fit_line,
    palette_from_selector,
    render_palette,
)


class GroupedLMMarginalTests(unittest.TestCase):
    def test_reference_structure_is_complete(self) -> None:
        DEFAULT_DATA.validate()
        DEFAULT_STYLE.validate()
        self.assertEqual(GROUPS, ("Grass", "Land", "Water", "Urban"))
        self.assertEqual(len(PALETTES), 16)
        self.assertEqual(len({palette.colors for palette in PALETTES}), 16)
        self.assertEqual(len({palette.name for palette in PALETTES}), 16)
        for group in GROUPS:
            self.assertEqual(DEFAULT_DATA.gst[group].shape, (25,))
            self.assertEqual(DEFAULT_DATA.lst[group].shape, (25,))

    def test_palette_selector_accepts_number_and_name(self) -> None:
        self.assertEqual(palette_from_selector("1"), (1, PALETTES[0]))
        self.assertEqual(palette_from_selector(16), (16, PALETTES[15]))
        self.assertEqual(
            palette_from_selector("navy_cyan_rose_gold"),
            (16, PALETTES[15]),
        )

    def test_default_fits_match_printed_annotations(self) -> None:
        for group, (slope, intercept, r_squared) in TARGET_FITS.items():
            fit = fit_ols(DEFAULT_DATA.gst[group], DEFAULT_DATA.lst[group])
            self.assertAlmostEqual(fit.slope, slope, places=6)
            self.assertAlmostEqual(fit.intercept, intercept, places=6)
            self.assertAlmostEqual(fit.r_squared, r_squared, places=6)
            line = format_fit_line(group, fit)
            self.assertIn(f"y = {slope:.2f}x + {intercept:.2f}", line)
            self.assertIn(f"$R^2$ = {r_squared:.3f}", line)
            if group == "Urban":
                self.assertIn("p = 0.011", line)
            else:
                self.assertIn("p < 0.001", line)

    def test_create_figure_uses_reference_canvas(self) -> None:
        figure = create_figure(PALETTES[0])
        try:
            self.assertEqual(tuple(figure.get_size_inches()), (8.528, 7.848))
            self.assertEqual(len(figure.axes), 3)
            joint, marg_x, marg_y = figure.axes
            self.assertTrue(joint.axison)
            np.testing.assert_allclose(joint.get_xlim(), (2.0, 28.0))
            np.testing.assert_allclose(joint.get_ylim(), (7.0, 29.0))
            self.assertEqual(joint.get_xlabel(), "GST")
            self.assertEqual(joint.get_ylabel(), "LST")
            self.assertFalse(any(spine.get_visible() for spine in marg_x.spines.values()))
            self.assertFalse(any(spine.get_visible() for spine in marg_y.spines.values()))
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
