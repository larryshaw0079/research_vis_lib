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
        self.assertIn("grouped-lm-marginal", text)
        self.assertIn("grouped-ring-bar", text)
        self.assertIn("radar-bubble", text)
        self.assertIn("radial-line", text)

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
