from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.concordance_upset import (
    DEFAULT_DATA,
    DEFAULT_STYLE,
    GROUPS,
    PALETTES,
    VARIANT_ROWS,
    Combination,
    ConcordanceUpsetData,
    create_figure,
    group_label_color,
    palette_from_selector,
    render_palette,
)


class ConcordanceUpsetTests(unittest.TestCase):
    def test_reference_structure_is_complete(self) -> None:
        DEFAULT_DATA.validate()
        DEFAULT_STYLE.validate()
        self.assertEqual(len(GROUPS), 3)
        self.assertEqual(GROUPS[0], "Disconcordant")
        self.assertEqual(len(VARIANT_ROWS), 3)
        self.assertEqual(len(DEFAULT_DATA.combinations), 8)
        self.assertEqual(len(PALETTES), 18)
        self.assertEqual(len({palette.colors for palette in PALETTES}), 18)
        self.assertEqual(len({palette.name for palette in PALETTES}), 18)
        self.assertEqual(DEFAULT_DATA.total_patients(), 1111)
        self.assertEqual(DEFAULT_DATA.counts(), (420, 7, 76, 8, 42, 99, 209, 250))
        self.assertEqual(len(PALETTES[0].colors), 3)

    def test_membership_matrix_matches_the_carousel(self) -> None:
        matrix = DEFAULT_DATA.membership_matrix()
        self.assertEqual(matrix.shape, (3, 8))
        expected = np.array(
            [
                [False, True, True, False, False, True, False, True],
                [True, False, True, False, False, False, True, True],
                [False, False, False, False, True, True, True, True],
            ],
            dtype=bool,
        )
        np.testing.assert_array_equal(matrix, expected)

    def test_palette_selector_accepts_number_and_name(self) -> None:
        self.assertEqual(palette_from_selector("1"), (1, PALETTES[0]))
        self.assertEqual(palette_from_selector(18), (18, PALETTES[17]))
        self.assertEqual(palette_from_selector("tol_muted"), (17, PALETTES[16]))

    def test_group_label_darkens_near_white_fills(self) -> None:
        self.assertEqual(group_label_color("#1F7598"), "#1F7598")
        self.assertNotEqual(group_label_color("#FFFFFF"), "#FFFFFF")

    def test_validate_rejects_duplicate_or_reordered_groups(self) -> None:
        broken = ConcordanceUpsetData(
            combinations=DEFAULT_DATA.combinations[:-1]
            + (Combination(True, True, True, 250, GROUPS[0]),)
        )
        with self.assertRaises(ValueError):
            broken.validate()

    def test_create_figure_uses_reference_canvas(self) -> None:
        figure = create_figure(PALETTES[0])
        try:
            self.assertEqual(tuple(figure.get_size_inches()), (9.724, 6.412))
            self.assertEqual(len(figure.axes), 2)
            ax_bar, ax_mat = figure.axes
            self.assertTrue(ax_bar.axison)
            self.assertEqual(ax_bar.get_title(), "Patient-Level Concordance")
            self.assertEqual(ax_bar.get_ylabel(), "Number of patients")
            np.testing.assert_allclose(ax_bar.get_ylim(), (0.0, 478.0))
            self.assertFalse(ax_mat.axison)
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
