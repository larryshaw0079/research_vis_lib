from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.smooth_radar import (
    CATEGORIES,
    DEFAULT_DATA,
    DEFAULT_STYLE,
    MODELS,
    PALETTES,
    TASKS,
    TASK_CATEGORY,
    create_figure,
    palette_from_selector,
    render_palette,
    split_task_label,
    _angles,
    _smooth_closed_curve,
)


class SmoothRadarTests(unittest.TestCase):
    def test_reference_structure_is_complete(self) -> None:
        DEFAULT_DATA.validate()
        self.assertEqual(len(TASKS), 31)
        self.assertEqual(len(MODELS), 5)
        self.assertEqual(len(CATEGORIES), 3)
        self.assertEqual(len(PALETTES), 18)
        self.assertEqual(len({palette.models for palette in PALETTES}), 18)
        self.assertEqual(len({palette.name for palette in PALETTES}), 18)
        self.assertEqual(set(TASK_CATEGORY), set(TASKS))
        self.assertEqual(set(TASK_CATEGORY.values()), set(CATEGORIES))
        self.assertEqual(len(PALETTES[0].models), 5)
        self.assertEqual(len(PALETTES[0].categories), 3)
        for model in MODELS:
            self.assertEqual(len(DEFAULT_DATA.auroc[model]), 31)

    def test_palette_selector_accepts_number_and_name(self) -> None:
        self.assertEqual(palette_from_selector("1"), (1, PALETTES[0]))
        self.assertEqual(palette_from_selector(18), (18, PALETTES[17]))
        self.assertEqual(palette_from_selector("tol_muted"), (11, PALETTES[10]))

    def test_task_labels_split_into_cohort_and_endpoint(self) -> None:
        self.assertEqual(split_task_label("CPTAC CRC KRAS"), ("CPTAC CRC", "KRAS"))
        self.assertEqual(
            split_task_label("CPTAC NSCLC Subtyping"), ("CPTAC NSCLC", "Subtyping")
        )
        self.assertEqual(
            split_task_label("BERN STAD N-STATUS"), ("BERN STAD", "N-STATUS")
        )

    def test_axes_run_clockwise_from_north(self) -> None:
        angles = _angles(31)
        self.assertAlmostEqual(angles[0], np.pi / 2)
        self.assertLess(angles[1], angles[0])
        self.assertAlmostEqual(angles[1], np.pi / 2 - 2 * np.pi / 31)

    def test_smooth_curve_is_closed_and_periodic(self) -> None:
        values = DEFAULT_DATA.auroc["EAGLE"]
        theta, radii = _smooth_closed_curve(values, 16)
        self.assertGreater(len(theta), 31)
        self.assertAlmostEqual(float(radii[0]), float(radii[-1]), places=6)
        self.assertAlmostEqual(float(theta[0] - theta[-1]), 2 * np.pi, places=6)

    def test_create_figure_uses_reference_canvas(self) -> None:
        figure = create_figure(PALETTES[0])
        try:
            self.assertEqual(
                tuple(round(side, 3) for side in figure.get_size_inches()),
                (10.24, 8.972),
            )
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
