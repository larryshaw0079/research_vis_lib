from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from rvl.cli import main

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def run(argv: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


class CliTests(unittest.TestCase):
    def test_templates_lists_every_template(self) -> None:
        code, out, _ = run(["templates"])
        self.assertEqual(code, 0)
        self.assertIn("grouped-ring-bar", out)
        self.assertIn("chart templates", out)

    def test_templates_json_is_machine_readable(self) -> None:
        import json

        code, out, _ = run(["templates", "--json"])
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertGreaterEqual(len(payload), 12)
        self.assertIn("template_id", payload[0])

    def test_templates_verbose_includes_guidance(self) -> None:
        code, out, _ = run(["templates", "-v"])
        self.assertEqual(code, 0)
        self.assertIn("good for:", out)
        self.assertIn("avoid:", out)

    def test_profile_describes_the_readings(self) -> None:
        code, out, _ = run(["profile", str(EXAMPLES / "forecasting_mse.xlsx")])
        self.assertEqual(code, 0)
        self.assertIn("readings:", out)
        self.assertIn("matrix", out)

    def test_recommend_ranks_templates(self) -> None:
        code, out, _ = run(
            ["recommend", str(EXAMPLES / "forecasting_mse.xlsx"), "--table", "MSE"]
        )
        self.assertEqual(code, 0)
        self.assertIn("ranked templates:", out)
        self.assertIn("score", out)

    def test_generate_writes_a_script(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "figure.py"
            code, out, _ = run(
                [
                    "generate",
                    str(EXAMPLES / "forecasting_mse.xlsx"),
                    "--table",
                    "MSE",
                    "-o",
                    str(target),
                ]
            )
            self.assertEqual(code, 0)
            self.assertTrue(target.exists())
            self.assertIn("template:", out)

    def test_generate_prints_to_stdout_without_output_flag(self) -> None:
        code, out, _ = run(
            ["generate", str(EXAMPLES / "variance_contributions.json")]
        )
        self.assertEqual(code, 0)
        self.assertIn("create_figure", out)

    def test_generate_rejects_an_inapplicable_template(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            run(
                [
                    "generate",
                    str(EXAMPLES / "variant_concordance.csv"),
                    "--template",
                    "pie-3d",
                ]
            )
        self.assertIn("cannot draw this data", str(caught.exception))

    def test_verify_passes_for_a_generated_script(self) -> None:
        source = EXAMPLES / "forecasting_mse.xlsx"
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "figure.py"
            run(
                [
                    "generate",
                    str(source),
                    "--table",
                    "MSE",
                    "--dpi",
                    "40",
                    "--figure-dir",
                    str(Path(directory) / "out"),
                    "-o",
                    str(target),
                ]
            )
            code, out, _ = run(["verify", str(target), "--source", str(source)])
        self.assertEqual(code, 0, out)
        self.assertIn("RESULT: pass", out)

    def test_missing_data_file_is_a_clean_error(self) -> None:
        code, _, err = run(["profile", "no/such/file.csv"])
        self.assertEqual(code, 2)
        self.assertIn("error:", err)

    def test_unknown_subcommand_exits_nonzero(self) -> None:
        with self.assertRaises(SystemExit):
            run(["frobnicate"])

    def test_render_forwards_flags_to_the_template(self) -> None:
        code, out, _ = run(["render", "grouped-ring-bar", "--list-palettes"])
        self.assertEqual(code, 0)
        self.assertIn("rose-steel-indigo", out)

    def test_render_rejects_an_unknown_template(self) -> None:
        with self.assertRaises(SystemExit):
            run(["render", "no-such-chart", "--list-palettes"])


if __name__ == "__main__":
    unittest.main()
