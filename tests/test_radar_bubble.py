from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt

from radar_bubble import (
    DEFAULT_DATA,
    PALETTES,
    create_figure,
    palette_from_selector,
    render_palette,
)


class RadarBubbleTests(unittest.TestCase):
    def test_reference_data_and_palettes_are_complete(self) -> None:
        DEFAULT_DATA.validate()
        self.assertEqual(len(PALETTES), 18)
        self.assertEqual(len({palette.colors for palette in PALETTES}), 18)

    def test_palette_selector_accepts_number_and_name(self) -> None:
        self.assertEqual(palette_from_selector("1"), (1, PALETTES[0]))
        self.assertEqual(
            palette_from_selector("violet_magenta_cyan"), (4, PALETTES[3])
        )

    def test_create_figure_uses_reference_canvas_ratio(self) -> None:
        figure = create_figure(PALETTES[0])
        try:
            self.assertEqual(tuple(figure.get_size_inches()), (11.96, 10.8))
            self.assertEqual(len(figure.axes), 1)
        finally:
            plt.close(figure)

    def test_render_palette_writes_requested_formats(self) -> None:
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
