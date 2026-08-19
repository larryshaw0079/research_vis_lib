from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rvl.ingest import (
    ColumnKind,
    column_letter,
    is_missing,
    parse_boolean,
    parse_measurement,
    read_table,
    read_tables,
)

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


class MeasurementParsingTests(unittest.TestCase):
    def test_plain_numbers(self) -> None:
        self.assertEqual(parse_measurement("0.447").value, 0.447)
        self.assertEqual(parse_measurement(12).value, 12.0)
        self.assertEqual(parse_measurement("-3.5e2").value, -350.0)

    def test_thousands_separator(self) -> None:
        self.assertEqual(parse_measurement("1,842").value, 1842.0)

    def test_percentages_are_flagged_not_rescaled(self) -> None:
        parsed = parse_measurement("12.5%")
        self.assertEqual(parsed.value, 12.5)
        self.assertTrue(parsed.is_percent)

    def test_uncertainty_forms(self) -> None:
        for text in ("0.45 ± 0.02", "0.45 +/- 0.02", "0.45+-0.02"):
            with self.subTest(text=text):
                parsed = parse_measurement(text)
                self.assertAlmostEqual(parsed.value, 0.45)
                self.assertAlmostEqual(parsed.error, 0.02)

    def test_parenthesised_uncertainty(self) -> None:
        parsed = parse_measurement("0.45 (0.02)")
        self.assertAlmostEqual(parsed.value, 0.45)
        self.assertAlmostEqual(parsed.error, 0.02)

    def test_bold_emphasis_is_recorded(self) -> None:
        parsed = parse_measurement("**0.94**")
        self.assertAlmostEqual(parsed.value, 0.94)
        self.assertTrue(parsed.emphasised)

    def test_significance_markers_are_stripped(self) -> None:
        self.assertAlmostEqual(parse_measurement("0.42**").value, 0.42)
        self.assertAlmostEqual(parse_measurement("1.5†").value, 1.5)

    def test_missing_markers_return_none(self) -> None:
        for text in ("", "NA", "n/a", "--", "—", "none", "NaN"):
            with self.subTest(text=text):
                self.assertIsNone(parse_measurement(text))
                self.assertTrue(is_missing(text) or text.lower() == "nan")

    def test_text_is_not_numeric(self) -> None:
        self.assertIsNone(parse_measurement("Traffic"))
        self.assertIsNone(parse_measurement(True))


class BooleanParsingTests(unittest.TestCase):
    def test_affirmative_and_negative_forms(self) -> None:
        for text in ("yes", "TRUE", "✓", "1", "y"):
            self.assertIs(parse_boolean(text), True, text)
        for text in ("no", "false", "×", "0", "n"):
            self.assertIs(parse_boolean(text), False, text)

    def test_unrecognised_text_is_none(self) -> None:
        self.assertIsNone(parse_boolean("maybe"))


class ColumnLetterTests(unittest.TestCase):
    def test_letters_match_spreadsheet_convention(self) -> None:
        self.assertEqual(column_letter(0), "A")
        self.assertEqual(column_letter(25), "Z")
        self.assertEqual(column_letter(26), "AA")
        self.assertEqual(column_letter(27), "AB")

    def test_negative_index_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            column_letter(-1)


class ReaderTests(unittest.TestCase):
    def test_workbook_yields_one_table_per_sheet(self) -> None:
        tables = read_tables(EXAMPLES / "forecasting_mse.xlsx")
        self.assertEqual({table.name for table in tables}, {"MSE", "notes"})

    def test_spreadsheet_references_are_a1_style(self) -> None:
        table = read_table(EXAMPLES / "forecasting_mse.xlsx", name="MSE")
        self.assertEqual(table.cell_reference("iTransformer", 0), "MSE!B2")
        self.assertEqual(table.column("Dataset").labels()[0], "Traffic")

    def test_markdown_table_uses_the_nearest_heading_as_its_name(self) -> None:
        table = read_table(EXAMPLES / "pathology_auroc.md")
        self.assertEqual(table.name, "Pathology foundation model AUROC")
        self.assertIn("EAGLE", table.column_names)

    def test_markdown_bold_winners_are_detected(self) -> None:
        table = read_table(EXAMPLES / "pathology_auroc.md")
        emphasised = sum(
            len(table.column(name).emphasised_rows())
            for name in table.column_names
            if name != "Task"
        )
        self.assertEqual(emphasised, 10)

    def test_csv_delimiter_is_sniffed(self) -> None:
        table = read_table(EXAMPLES / "energy_mix_long.csv")
        self.assertEqual(table.column_names, ("Sector", "Fuel", "Energy_EJ"))

    def test_tsv_is_read_as_tab_separated(self) -> None:
        table = read_table(EXAMPLES / "policy_mix.tsv")
        self.assertIn("Carbon price", table.column_names)

    def test_json_record_array_becomes_a_table(self) -> None:
        table = read_table(EXAMPLES / "variance_contributions.json")
        self.assertEqual(table.column_names, ("factor", "contribution"))
        self.assertTrue(table.cell_reference("contribution", 0).startswith("$.factors"))

    def test_jsonl_is_read_line_by_line(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text(
                "\n".join(
                    json.dumps({"model": name, "score": value})
                    for name, value in (("a", 1.0), ("b", 2.0), ("c", 3.0))
                ),
                encoding="utf-8",
            )
            table = read_table(path)
            self.assertEqual(table.n_rows, 3)
            self.assertEqual(table.column("score").numeric().tolist(), [1.0, 2.0, 3.0])

    def test_boolean_columns_beat_numeric_for_flags(self) -> None:
        table = read_table(EXAMPLES / "variant_concordance.csv")
        self.assertEqual(table.column("Plasma_specific").kind, ColumnKind.BOOLEAN)
        self.assertEqual(table.column("Patients").kind, ColumnKind.NUMERIC)

    def test_dates_are_recognised_as_ordered(self) -> None:
        table = read_table(EXAMPLES / "n2o_flux_timeseries.csv")
        self.assertEqual(table.column("Date").kind, ColumnKind.TEMPORAL)

    def test_headerless_grid_gets_synthetic_column_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bare.csv"
            path.write_text("1,2\n3,4\n5,6\n", encoding="utf-8")
            table = read_table(path)
            self.assertEqual(table.n_rows, 3)
            self.assertTrue(all(name.startswith("column_") for name in table.column_names))

    def test_duplicate_headers_are_disambiguated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dupes.csv"
            path.write_text("group,value,value\na,1,2\nb,3,4\n", encoding="utf-8")
            table = read_table(path)
            self.assertEqual(table.column_names, ("group", "value", "value_1"))

    def test_missing_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            read_tables(EXAMPLES / "does_not_exist.csv")

    def test_unsupported_suffix_lists_what_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.parquet"
            path.write_bytes(b"not really parquet")
            with self.assertRaises(ValueError) as caught:
                read_tables(path)
            self.assertIn(".csv", str(caught.exception))

    def test_read_table_prefers_the_largest_table(self) -> None:
        table = read_table(EXAMPLES / "forecasting_mse.xlsx")
        self.assertEqual(table.name, "MSE")


class TableTests(unittest.TestCase):
    def test_error_columns_are_separated_from_values(self) -> None:
        table = read_table(EXAMPLES / "n2o_flux_timeseries.csv")
        self.assertEqual(table.n_columns, 5)
        self.assertEqual(len(table.numeric_columns()), 4)

    def test_matrix_shape_matches_numeric_columns(self) -> None:
        table = read_table(EXAMPLES / "forecasting_mse.xlsx", name="MSE")
        self.assertEqual(table.matrix().shape, (8, 5))

    def test_unknown_column_lists_the_alternatives(self) -> None:
        table = read_table(EXAMPLES / "otu_overlap.csv")
        with self.assertRaises(KeyError) as caught:
            table.column("nope")
        self.assertIn("Total_OTUs", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
