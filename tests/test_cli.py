from __future__ import annotations

import io
import unittest
from contextlib import redirect_stderr, redirect_stdout

from src.cli import main


class CliTests(unittest.TestCase):
    def test_help_lists_chart_commands(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            status = main(["--help"])
        text = buffer.getvalue()
        self.assertEqual(status, 0)
        self.assertIn("bar-line", text)
        self.assertIn("concordance-upset", text)
        self.assertIn("flower-plot", text)
        self.assertIn("grouped-gradient-hist", text)
        self.assertIn("grouped-lm-marginal", text)
        self.assertIn("grouped-ring-bar", text)
        self.assertIn("pie-3d", text)
        self.assertIn("radar-bubble", text)
        self.assertIn("radial-line", text)
        self.assertIn("smooth-radar", text)

    def test_unknown_chart_returns_usage_error(self) -> None:
        buffer = io.StringIO()
        with redirect_stderr(buffer):
            status = main(["not-a-chart"])
        self.assertEqual(status, 2)
        self.assertIn("unknown chart", buffer.getvalue())

    def test_chart_command_reaches_the_selected_renderer(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            status = main(["radar-bubble", "--list-palettes"])
        self.assertEqual(status, 0)
        self.assertIn("slate-coral-apricot", buffer.getvalue())

    def test_grouped_gradient_hist_command_lists_palettes(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            status = main(["grouped-gradient-hist", "--list-palettes"])
        self.assertEqual(status, 0)
        self.assertIn("coral-navy-teal", buffer.getvalue())
        self.assertIn("wine-azure-emerald", buffer.getvalue())

    def test_pie_3d_command_lists_palettes(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            status = main(["pie-3d", "--list-palettes"])
        self.assertEqual(status, 0)
        self.assertIn("navy-peach-maroon", buffer.getvalue())
        self.assertIn("spectral-rainbow", buffer.getvalue())

    def test_flower_plot_command_lists_palettes(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            status = main(["flower-plot", "--list-palettes"])
        self.assertEqual(status, 0)
        self.assertIn("gray-peach-orchid", buffer.getvalue())
        self.assertIn("tol-muted", buffer.getvalue())

    def test_bar_line_command_lists_palettes(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            status = main(["bar-line", "--list-palettes"])
        self.assertEqual(status, 0)
        self.assertIn("orange-green", buffer.getvalue())
        self.assertIn("rose-sage", buffer.getvalue())

    def test_concordance_upset_command_lists_palettes(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            status = main(["concordance-upset", "--list-palettes"])
        self.assertEqual(status, 0)
        self.assertIn("teal-coral-sand", buffer.getvalue())
        self.assertIn("set2-bloom", buffer.getvalue())

    def test_smooth_radar_command_lists_palettes(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            status = main(["smooth-radar", "--list-palettes"])
        self.assertEqual(status, 0)
        self.assertIn("crimson-peach-ice", buffer.getvalue())
        self.assertIn("tol-muted", buffer.getvalue())
