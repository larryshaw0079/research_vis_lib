from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.flower_plot import (
    DEFAULT_DATA,
    DEFAULT_STYLE,
    GROUPS,
    PALETTES,
    create_figure,
    palette_from_selector,
    render_palette,
    rounded_petal_vertices,
    _contrasting_text_color,
    _label_rotation,
    _petal_center_degrees,
)


class FlowerPlotTests(unittest.TestCase):
    def test_reference_structure_is_complete(self) -> None:
        DEFAULT_DATA.validate()
        DEFAULT_STYLE.validate()
        self.assertEqual(len(GROUPS), 10)
        self.assertEqual(len(PALETTES), 18)
        self.assertEqual(len({palette.colors for palette in PALETTES}), 18)
        self.assertEqual(len({palette.name for palette in PALETTES}), 18)
        self.assertEqual(DEFAULT_DATA.core, 936)
        for group in GROUPS:
            self.assertIn(group, DEFAULT_DATA.totals)
            self.assertIn(group, DEFAULT_DATA.uniques)
            self.assertLessEqual(DEFAULT_DATA.uniques[group], DEFAULT_DATA.totals[group])
        self.assertEqual(len(PALETTES[0].colors), 10)

    def test_palette_selector_accepts_number_and_name(self) -> None:
        self.assertEqual(palette_from_selector("1"), (1, PALETTES[0]))
        self.assertEqual(palette_from_selector(18), (18, PALETTES[17]))
        self.assertEqual(palette_from_selector("tol_muted"), (11, PALETTES[10]))

    def test_petals_run_clockwise_from_north(self) -> None:
        self.assertAlmostEqual(_petal_center_degrees(0), 90.0)
        self.assertAlmostEqual(_petal_center_degrees(1), 54.0)
        self.assertAlmostEqual(_petal_center_degrees(5), 270.0)

    def test_outer_labels_stay_right_side_up(self) -> None:
        self.assertAlmostEqual(_label_rotation(90.0), 0.0)
        self.assertAlmostEqual(_label_rotation(54.0), -36.0)
        self.assertAlmostEqual(_label_rotation(270.0), 0.0)
        self.assertAlmostEqual(_label_rotation(126.0), 36.0)

    def test_text_color_follows_petal_luminance(self) -> None:
        self.assertEqual(_contrasting_text_color("#1F77B4", 0.22), "#FFFFFF")
        self.assertEqual(_contrasting_text_color("#FFFFCC", 0.22), "#000000")
        self.assertEqual(_contrasting_text_color("#EC655D", 0.22), "#000000")

    def test_petal_outline_is_closed_and_rounded(self) -> None:
        vertices = rounded_petal_vertices(
            90.0,
            DEFAULT_STYLE.petal_width_degrees,
            DEFAULT_STYLE.inner_radius,
            DEFAULT_STYLE.outer_radius,
            DEFAULT_STYLE.inner_corner_radius,
            DEFAULT_STYLE.outer_corner_radius,
        )
        radii = np.hypot(vertices[:, 0], vertices[:, 1])
        angles = np.degrees(np.arctan2(vertices[:, 1], vertices[:, 0]))
        half_width = 0.5 * DEFAULT_STYLE.petal_width_degrees
        mid = (radii > DEFAULT_STYLE.inner_radius + 0.04) & (
            radii < DEFAULT_STYLE.outer_radius - 0.08
        )
        self.assertGreater(len(vertices), 80)
        self.assertGreater(float(radii.max()), DEFAULT_STYLE.outer_radius - 0.01)
        self.assertLess(float(radii.min()), DEFAULT_STYLE.inner_radius + 0.02)
        self.assertLess(
            float(np.linalg.norm(vertices[0] - vertices[-1])),
            DEFAULT_STYLE.outer_radius - DEFAULT_STYLE.inner_radius,
        )
        self.assertLess(DEFAULT_STYLE.outer_corner_radius, 0.08)
        self.assertGreater(DEFAULT_STYLE.outer_corner_radius, 0.04)
        if np.any(mid):
            self.assertLess(float(np.abs(angles[mid] - 90.0).max()), half_width + 0.4)

    def test_create_figure_uses_reference_canvas(self) -> None:
        figure = create_figure(PALETTES[0])
        try:
            self.assertEqual(tuple(figure.get_size_inches()), (10.24, 10.24))
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
