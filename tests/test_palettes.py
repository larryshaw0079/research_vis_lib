from __future__ import annotations

import unittest

from rvl.palettes import (
    Palette,
    darken_if_pale,
    palette_from_selector,
    readable_text_color,
    relative_luminance,
    selected_palettes,
)
from rvl.render import nice_step, normalize_formats, padded_range, resolve_limits

PALETTES = (
    Palette("first-pair", ("#FA882F", "#338227")),
    Palette("second-pair", ("#DA5D47", "#538CB9")),
    Palette("third-pair", ("#CD89A0", "#539780")),
)


class PaletteTests(unittest.TestCase):
    def test_colours_cycle_and_stay_distinct(self) -> None:
        palette = PALETTES[0]
        taken = palette.take(6)
        self.assertEqual(taken[:2], palette.colors)
        self.assertEqual(len(set(taken)), 6)

    def test_take_zero_is_empty(self) -> None:
        self.assertEqual(PALETTES[0].take(0), ())

    def test_negative_index_is_rejected(self) -> None:
        with self.assertRaises(IndexError):
            PALETTES[0].color(-1)

    def test_invalid_colour_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Palette("bad", ("#not-a-colour",))

    def test_empty_palette_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Palette("empty", ())

    def test_name_must_be_lowercase(self) -> None:
        with self.assertRaises(ValueError):
            Palette("Mixed-Case", ("#000000",))

    def test_reversed_keeps_the_colours(self) -> None:
        reversed_palette = PALETTES[0].reversed()
        self.assertEqual(reversed_palette.colors, tuple(reversed(PALETTES[0].colors)))


class SelectorTests(unittest.TestCase):
    def test_number_and_name_both_resolve(self) -> None:
        self.assertEqual(palette_from_selector(PALETTES, "1"), (1, PALETTES[0]))
        self.assertEqual(palette_from_selector(PALETTES, 3), (3, PALETTES[2]))
        self.assertEqual(
            palette_from_selector(PALETTES, "second_pair"), (2, PALETTES[1])
        )

    def test_out_of_range_number_reports_the_range(self) -> None:
        with self.assertRaises(ValueError) as caught:
            palette_from_selector(PALETTES, 9)
        self.assertIn("between 1 and 3", str(caught.exception))

    def test_unknown_name_lists_the_choices(self) -> None:
        with self.assertRaises(ValueError) as caught:
            palette_from_selector(PALETTES, "chartreuse")
        self.assertIn("first-pair", str(caught.exception))

    def test_all_selects_every_palette(self) -> None:
        self.assertEqual(len(selected_palettes(PALETTES, "all")), 3)
        self.assertEqual(len(selected_palettes(PALETTES, "2")), 1)


class ContrastTests(unittest.TestCase):
    def test_luminance_ordering(self) -> None:
        self.assertGreater(relative_luminance("#FFFFFF"), relative_luminance("#000000"))

    def test_text_colour_flips_with_background(self) -> None:
        self.assertEqual(readable_text_color("#FFFFFF"), "#111111")
        self.assertEqual(readable_text_color("#101010"), "#FFFFFF")

    def test_pale_fills_are_darkened_for_use_as_ink(self) -> None:
        self.assertNotEqual(darken_if_pale("#FFFFFF"), "#ffffff")
        self.assertEqual(darken_if_pale("#336C99"), "#336C99")


class ScaleTests(unittest.TestCase):
    def test_nice_step_is_a_round_number(self) -> None:
        self.assertIn(nice_step(100.0), {10.0, 20.0, 25.0, 50.0})
        self.assertGreater(nice_step(0.4), 0.0)

    def test_padded_range_includes_zero_by_default(self) -> None:
        low, high = padded_range(4.0, 9.0)
        self.assertLessEqual(low, 0.0)
        self.assertGreaterEqual(high, 9.0)

    def test_padded_range_handles_a_flat_series(self) -> None:
        low, high = padded_range(5.0, 5.0)
        self.assertLess(low, high)

    def test_pinned_limits_are_kept_while_the_data_fits(self) -> None:
        self.assertEqual(resolve_limits((0.0, 165.0), 6.0, 152.0), (0.0, 165.0))

    def test_pinned_limits_are_dropped_when_data_would_clip(self) -> None:
        low, high = resolve_limits((0.0, 165.0), 6.0, 900.0)
        self.assertGreaterEqual(high, 900.0)

    def test_pinned_limits_are_dropped_when_data_would_be_a_sliver(self) -> None:
        """Counts of a few dozen must not be drawn on an axis tuned for hundreds."""

        low, high = resolve_limits((0.0, 478.0), 0.0, 31.0)
        self.assertLess(high, 100.0)

    def test_min_fill_threshold_is_adjustable(self) -> None:
        self.assertEqual(resolve_limits((0.0, 100.0), 0.0, 40.0), (0.0, 100.0))
        self.assertLess(resolve_limits((0.0, 100.0), 0.0, 40.0, min_fill=0.9)[1], 100.0)

    def test_a_degenerate_pin_is_still_returned(self) -> None:
        self.assertEqual(resolve_limits((5.0, 5.0), 5.0, 5.0), (5.0, 5.0))

    def test_no_pin_auto_fits(self) -> None:
        low, high = resolve_limits(None, 1.0, 3.0)
        self.assertGreaterEqual(high, 3.0)


class FormatTests(unittest.TestCase):
    def test_formats_are_normalised_and_deduplicated(self) -> None:
        self.assertEqual(normalize_formats([".PNG", "png", "svg"]), ("png", "svg"))

    def test_unsupported_format_lists_the_options(self) -> None:
        with self.assertRaises(ValueError) as caught:
            normalize_formats(["gif"])
        self.assertIn("png", str(caught.exception))

    def test_empty_format_list_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalize_formats([])


if __name__ == "__main__":
    unittest.main()
