"""Profiling, recommendation, code generation and verification, end to end."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from rvl.adapters import builder_kwargs
from rvl.codegen import generate
from rvl.contract import DataKind, Feature
from rvl.ingest import read_table
from rvl.profiling import profile_table
from rvl.recommend import features_of, recommend, shape_of
from rvl.registry import load_registry
from rvl.verify import Level, inspect_script, verify

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
REGISTRY = load_registry()


def profile_for(name: str, table: str | None = None):
    return profile_table(read_table(EXAMPLES / name, name=table))


class ProfilingTests(unittest.TestCase):
    def test_wide_benchmark_reads_as_a_matrix(self) -> None:
        profile = profile_for("forecasting_mse.xlsx", "MSE")
        self.assertIn(DataKind.MATRIX, profile.kinds())
        reading = profile.for_kind(DataKind.MATRIX)[0]
        self.assertEqual(reading.n_categories, 8)
        self.assertEqual(reading.n_series, 5)
        self.assertEqual(reading.categories[0], "Traffic")

    def test_error_metric_sets_the_direction(self) -> None:
        self.assertTrue(profile_for("forecasting_mse.xlsx", "MSE").lower_is_better)
        self.assertFalse(profile_for("pathology_auroc.md").lower_is_better)

    def test_percentages_that_sum_to_100_are_detected(self) -> None:
        profile = profile_for("variance_contributions.json")
        self.assertTrue(profile.sums_to_100)
        self.assertIn(DataKind.PARTS_OF_WHOLE, profile.kinds())

    def test_long_form_data_is_pivoted(self) -> None:
        profile = profile_for("energy_mix_long.csv")
        reading = profile.for_kind(DataKind.MATRIX)[0]
        self.assertEqual(reading.n_categories, 4)
        self.assertEqual(reading.n_series, 4)
        self.assertIn(DataKind.NESTED_PARTS, profile.kinds())

    def test_dates_produce_an_ordered_series_reading(self) -> None:
        profile = profile_for("n2o_flux_timeseries.csv")
        self.assertIn(DataKind.SERIES_WITH_TOTALS, profile.kinds())
        reading = profile.for_kind(DataKind.SERIES_WITH_TOTALS)[0]
        self.assertTrue(reading.ordered_categories)
        self.assertIn("errors", reading.extras)

    def test_repeating_group_yields_pairs_and_distributions(self) -> None:
        profile = profile_for("surface_temperature_pairs.csv")
        self.assertIn(DataKind.XY_SAMPLES, profile.kinds())
        self.assertIn(DataKind.DISTRIBUTION_SAMPLES, profile.kinds())

    def test_membership_columns_are_recognised(self) -> None:
        profile = profile_for("variant_concordance.csv")
        self.assertEqual(profile.kinds(), (DataKind.SET_MEMBERSHIP,))
        reading = profile.interpretations[0]
        self.assertEqual(len(reading.extras["memberships"]), 8)
        self.assertEqual(len(reading.extras["sets"]), 3)

    def test_total_and_unique_counts_read_as_overlap(self) -> None:
        profile = profile_for("otu_overlap.csv")
        self.assertIn(DataKind.SET_OVERLAP, profile.kinds())
        reading = profile.for_kind(DataKind.SET_OVERLAP)[0]
        self.assertEqual(reading.extras["core"], 936.0)

    def test_dynamic_range_ignores_data_that_straddles_zero(self) -> None:
        profile = profile_for("residual_strain.csv")
        self.assertFalse(profile.all_non_negative)
        self.assertEqual(profile.dynamic_range, 1.0)

    def test_bold_cells_surface_as_emphasis(self) -> None:
        self.assertEqual(profile_for("pathology_auroc.md").emphasised_cells, 10)

    def test_provenance_survives_into_the_reading(self) -> None:
        reading = profile_for("forecasting_mse.xlsx", "MSE").for_kind(DataKind.MATRIX)[0]
        self.assertEqual(reading.references[0][0], "MSE!B2")
        self.assertEqual(len(reading.all_references()), 40)

    def test_summary_mentions_every_reading(self) -> None:
        summary = profile_for("otu_overlap.csv").summary()
        for kind in profile_for("otu_overlap.csv").kinds():
            self.assertIn(kind.value, summary)


class FeatureTests(unittest.TestCase):
    def test_ordered_and_uncertain_data_reports_both(self) -> None:
        profile = profile_for("n2o_flux_timeseries.csv")
        reading = profile.for_kind(DataKind.SERIES_WITH_TOTALS)[0]
        features = features_of(profile, reading)
        self.assertIn(Feature.ORDERED_CATEGORIES, features)
        self.assertIn(Feature.HAS_UNCERTAINTY, features)
        self.assertIn(Feature.NON_NEGATIVE, features)

    def test_shape_of_counts_pairs_for_xy_readings(self) -> None:
        profile = profile_for("surface_temperature_pairs.csv")
        reading = profile.for_kind(DataKind.XY_SAMPLES)[0]
        categories, series = shape_of(reading)
        self.assertEqual(series, 4)
        self.assertGreaterEqual(categories, 20)


class RecommendationTests(unittest.TestCase):
    def test_every_example_gets_at_least_one_candidate(self) -> None:
        for name, table in (
            ("forecasting_mse.xlsx", "MSE"),
            ("pathology_auroc.md", None),
            ("energy_mix_long.csv", None),
            ("variance_contributions.json", None),
            ("n2o_flux_timeseries.csv", None),
            ("otu_overlap.csv", None),
            ("variant_concordance.csv", None),
            ("surface_temperature_pairs.csv", None),
            ("residual_strain.csv", None),
            ("policy_mix.tsv", None),
        ):
            with self.subTest(data=name):
                result = recommend(profile_for(name, table))
                self.assertTrue(result.candidates, "no template matched")
                self.assertIsNotNone(result.best)

    def test_candidates_are_sorted_by_score(self) -> None:
        result = recommend(profile_for("forecasting_mse.xlsx", "MSE"))
        scores = [candidate.score for candidate in result.candidates]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_every_candidate_explains_itself(self) -> None:
        result = recommend(profile_for("forecasting_mse.xlsx", "MSE"))
        for candidate in result.candidates:
            with self.subTest(template=candidate.template_id):
                self.assertTrue(candidate.reasons)

    def test_rejections_explain_themselves(self) -> None:
        result = recommend(profile_for("variant_concordance.csv"))
        self.assertTrue(result.rejected)
        for template_id, reason in result.rejected:
            with self.subTest(template=template_id):
                self.assertGreater(len(reason), 15)

    def test_composition_data_prefers_a_composition_template(self) -> None:
        result = recommend(profile_for("variance_contributions.json"))
        best = result.best
        self.assertIn(
            best.interpretation.kind,
            {DataKind.PARTS_OF_WHOLE, DataKind.MATRIX},
        )

    def test_ordered_data_is_not_won_by_a_circular_template(self) -> None:
        result = recommend(profile_for("n2o_flux_timeseries.csv"))
        best = result.best
        if best.interpretation.ordered_categories:
            self.assertNotIn(best.spec.geometry.value, {"circular"})

    def test_a_named_template_can_be_looked_up(self) -> None:
        result = recommend(profile_for("forecasting_mse.xlsx", "MSE"))
        found = result.for_template(result.best.template_id)
        self.assertEqual(found, result.best)
        self.assertIsNone(result.for_template("no-such-template"))


class HintPropagationTests(unittest.TestCase):
    """Inferences that would silently misrepresent the data if dropped."""

    def test_error_metric_direction_reaches_the_builder(self) -> None:
        profile = profile_for("forecasting_mse.xlsx", "MSE")
        reading = profile.for_kind(DataKind.MATRIX)[0]
        spec = REGISTRY.get("grouped-ring-bar")
        arguments = builder_kwargs(spec, reading)
        self.assertIs(arguments["lower_is_better"], True)

    def test_score_metric_is_not_marked_lower_is_better(self) -> None:
        profile = profile_for("pathology_auroc.md")
        reading = profile.for_kind(DataKind.MATRIX)[0]
        arguments = builder_kwargs(REGISTRY.get("grouped-ring-bar"), reading)
        self.assertIs(arguments["lower_is_better"], False)

    def test_bold_winner_becomes_the_highlighted_series(self) -> None:
        profile = profile_for("pathology_auroc.md")
        reading = profile.for_kind(DataKind.MATRIX)[0]
        arguments = builder_kwargs(REGISTRY.get("grouped-ring-bar"), reading)
        self.assertIn(arguments["highlight"], reading.series)

    def test_hints_are_skipped_for_builders_that_reject_them(self) -> None:
        profile = profile_for("forecasting_mse.xlsx", "MSE")
        reading = profile.for_kind(DataKind.MATRIX)[0]
        arguments = builder_kwargs(REGISTRY.get("smooth-radar"), reading)
        self.assertNotIn("lower_is_better", arguments)
        self.assertNotIn("highlight", arguments)

    def test_value_label_falls_back_to_a_metric_named_table(self) -> None:
        reading = profile_for("forecasting_mse.xlsx", "MSE").for_kind(DataKind.MATRIX)[0]
        self.assertEqual(reading.value_label, "MSE")

    def test_value_label_uses_the_single_column_name(self) -> None:
        reading = profile_for("energy_mix_long.csv").for_kind(DataKind.MATRIX)[0]
        self.assertEqual(reading.value_label, "Energy_EJ")

    def test_axis_names_reach_an_xy_builder(self) -> None:
        profile = profile_for("surface_temperature_pairs.csv")
        reading = profile.for_kind(DataKind.XY_SAMPLES)[0]
        arguments = builder_kwargs(REGISTRY.get("grouped-lm-marginal"), reading)
        self.assertEqual(arguments["x_label"], "LST_C")
        self.assertEqual(arguments["y_label"], "GST_C")


class CodegenTests(unittest.TestCase):
    def test_generated_script_is_valid_python_and_uses_a_builder(self) -> None:
        result = recommend(profile_for("forecasting_mse.xlsx", "MSE"))
        script = generate(result.best, dpi=40)
        with tempfile.TemporaryDirectory() as directory:
            path = script.write(Path(directory) / "figure.py")
            facts = inspect_script(path)
        self.assertTrue(facts.builder_calls)
        self.assertNotIn("DEFAULT_DATA", facts.imported_names)

    def test_generated_script_inlines_the_source_values(self) -> None:
        result = recommend(profile_for("forecasting_mse.xlsx", "MSE"))
        script = generate(result.best)
        self.assertIn("0.447", script.code)
        self.assertIn("Traffic", script.code)
        self.assertEqual(len(script.plotted_values), 40)

    def test_generated_script_records_its_provenance(self) -> None:
        result = recommend(profile_for("forecasting_mse.xlsx", "MSE"))
        script = generate(result.best)
        self.assertIn("MSE!B2", script.code)
        self.assertIn("forecasting_mse.xlsx", script.code)

    def test_every_applicable_template_can_be_generated(self) -> None:
        """Whatever the recommender offers, the generator must be able to emit."""

        for name, table in (
            ("forecasting_mse.xlsx", "MSE"),
            ("energy_mix_long.csv", None),
            ("variance_contributions.json", None),
            ("n2o_flux_timeseries.csv", None),
            ("otu_overlap.csv", None),
            ("variant_concordance.csv", None),
            ("surface_temperature_pairs.csv", None),
            ("residual_strain.csv", None),
        ):
            result = recommend(profile_for(name, table))
            for candidate in result.candidates:
                with self.subTest(data=name, template=candidate.template_id):
                    script = generate(candidate, dpi=40)
                    compile(script.code, candidate.template_id, "exec")


class VerificationTests(unittest.TestCase):
    def test_a_generated_script_passes_every_check(self) -> None:
        source = EXAMPLES / "forecasting_mse.xlsx"
        result = recommend(profile_for("forecasting_mse.xlsx", "MSE"))
        with tempfile.TemporaryDirectory() as directory:
            script = generate(
                result.best, dpi=40, output_dir=str(Path(directory) / "figures")
            )
            path = script.write(Path(directory) / "figure.py")
            report = verify(path, [source])
        self.assertTrue(report.ok, report.render())
        self.assertTrue(report.rendered)

    def test_an_invented_value_is_caught(self) -> None:
        source = EXAMPLES / "forecasting_mse.xlsx"
        result = recommend(profile_for("forecasting_mse.xlsx", "MSE"))
        script = generate(result.best)
        tampered = script.code.replace("0.447", "0.999123", 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "figure.py"
            path.write_text(tampered, encoding="utf-8")
            report = verify(path, [source], run=False)
        self.assertFalse(report.ok)
        self.assertTrue(
            any(finding.check == "provenance" for finding in report.errors),
            report.render(),
        )

    def test_importing_default_data_is_caught(self) -> None:
        source = EXAMPLES / "forecasting_mse.xlsx"
        spec = REGISTRY.get("grouped-ring-bar")
        code = (
            f"from {spec.module_name} import DEFAULT_DATA, PALETTES, create_figure\n"
            "figure = create_figure(palette=PALETTES[0], data=DEFAULT_DATA)\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "leak.py"
            path.write_text(code, encoding="utf-8")
            report = verify(path, [source], run=False)
        self.assertFalse(report.ok)
        self.assertTrue(any(finding.check == "demo-data" for finding in report.errors))

    def test_a_script_without_a_builder_is_caught(self) -> None:
        source = EXAMPLES / "forecasting_mse.xlsx"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plain.py"
            path.write_text("value = 0.447\n", encoding="utf-8")
            report = verify(path, [source], run=False)
        self.assertFalse(report.ok)
        self.assertTrue(any(finding.check == "structure" for finding in report.errors))

    def test_a_script_that_crashes_is_reported(self) -> None:
        source = EXAMPLES / "forecasting_mse.xlsx"
        result = recommend(profile_for("forecasting_mse.xlsx", "MSE"))
        script = generate(result.best)
        broken = script.code + "\nraise SystemExit(3)\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "boom.py"
            path.write_text(broken, encoding="utf-8")
            report = verify(path, [source])
        self.assertFalse(report.ok)
        self.assertTrue(any(finding.check == "render" for finding in report.errors))

    def test_findings_render_with_a_level_marker(self) -> None:
        source = EXAMPLES / "forecasting_mse.xlsx"
        result = recommend(profile_for("forecasting_mse.xlsx", "MSE"))
        with tempfile.TemporaryDirectory() as directory:
            script = generate(result.best, dpi=40, output_dir=str(Path(directory)))
            path = script.write(Path(directory) / "figure.py")
            report = verify(path, [source], run=False)
        rendered = report.render()
        self.assertIn("RESULT:", rendered)
        for finding in report.findings:
            self.assertIn(finding.check, rendered)
            self.assertIn(finding.level, set(Level))


if __name__ == "__main__":
    unittest.main()
