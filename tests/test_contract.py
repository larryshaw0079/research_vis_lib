"""Contract tests that run against every discovered template.

A new template in ``rvl/templates/`` is covered by this file automatically, which
is why adding one needs no new test module.
"""

from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from rvl.adapters import build_data, builder_kwargs, supported_kinds
from rvl.contract import Extent, Feature, TemplateSpec
from rvl.palettes import Palette
from rvl.registry import load_registry
from rvl.render import save_figure

from .synthetic import probe_shapes, synthetic_reading

REGISTRY = load_registry()


class RegistryTests(unittest.TestCase):
    def test_templates_are_discovered(self) -> None:
        self.assertGreaterEqual(len(REGISTRY), 12)

    def test_ids_are_unique_and_kebab_case(self) -> None:
        self.assertEqual(len(set(REGISTRY.ids)), len(REGISTRY.ids))
        for template_id in REGISTRY.ids:
            self.assertRegex(template_id, r"^[a-z0-9]+(-[a-z0-9]+)*$")

    def test_lookup_is_forgiving_about_separators(self) -> None:
        first = REGISTRY.ids[0]
        self.assertEqual(REGISTRY.get(first.replace("-", "_")).template_id, first)
        self.assertEqual(REGISTRY.get(first.upper()).template_id, first)

    def test_unknown_template_lists_the_alternatives(self) -> None:
        with self.assertRaises(KeyError) as caught:
            REGISTRY.get("no-such-chart")
        self.assertIn(REGISTRY.ids[0], str(caught.exception))

    def test_every_kind_a_template_accepts_has_an_adapter(self) -> None:
        for spec in REGISTRY:
            for kind in spec.kinds:
                with self.subTest(template=spec.template_id, kind=kind):
                    self.assertIn(kind, supported_kinds())


class SpecTests(unittest.TestCase):
    def test_extents_are_self_consistent(self) -> None:
        for spec in REGISTRY:
            with self.subTest(template=spec.template_id):
                for name, extent in (
                    ("categories", spec.categories),
                    ("series", spec.series),
                ):
                    self.assertGreaterEqual(extent.minimum, 1, name)
                    if extent.maximum is not None:
                        self.assertGreaterEqual(extent.maximum, extent.minimum, name)

    def test_guidance_is_written_for_a_human(self) -> None:
        for spec in REGISTRY:
            with self.subTest(template=spec.template_id):
                self.assertTrue(spec.title.strip())
                self.assertTrue(spec.summary.strip())
                self.assertGreater(len(spec.data_contract), 40)
                self.assertTrue(spec.good_for, "good_for guides template choice")
                self.assertTrue(spec.avoid_when, "avoid_when prevents misuse")

    def test_affinities_and_requirements_are_valid(self) -> None:
        for spec in REGISTRY:
            with self.subTest(template=spec.template_id):
                for feature, weight in spec.affinities:
                    self.assertIsInstance(feature, Feature)
                    self.assertNotEqual(weight, 0.0, "a zero weight says nothing")
                for feature in spec.requires:
                    self.assertIsInstance(feature, Feature)

    def test_builder_is_declared_and_callable(self) -> None:
        for spec in REGISTRY:
            with self.subTest(template=spec.template_id):
                module = REGISTRY.module(spec.template_id)
                data_class = getattr(module, spec.data_class)
                self.assertTrue(callable(getattr(data_class, spec.builder_name)))

    def test_extent_rejects_impossible_ranges(self) -> None:
        with self.assertRaises(ValueError):
            Extent(4, 2)
        with self.assertRaises(ValueError):
            Extent(-1)

    def test_spec_rejects_duplicate_affinities(self) -> None:
        with self.assertRaises(ValueError):
            TemplateSpec(
                template_id="broken",
                title="t",
                summary="s",
                kinds=tuple(REGISTRY.specs[0].kinds),
                geometry=REGISTRY.specs[0].geometry,
                categories=Extent(2),
                series=Extent(2),
                builder="X.from_matrix",
                data_contract="a contract long enough to satisfy the checks in place",
                affinities=(
                    (Feature.NON_NEGATIVE, 1.0),
                    (Feature.NON_NEGATIVE, 2.0),
                ),
            )


class PaletteTests(unittest.TestCase):
    def test_palettes_are_well_formed(self) -> None:
        for spec in REGISTRY:
            module = REGISTRY.module(spec.template_id)
            with self.subTest(template=spec.template_id):
                self.assertGreaterEqual(len(module.PALETTES), 1)
                names = [palette.name for palette in module.PALETTES]
                self.assertEqual(len(set(names)), len(names), "palette names collide")
                for palette in module.PALETTES:
                    self.assertIsInstance(palette, Palette)
                    self.assertGreaterEqual(len(palette.colors), 1)

    def test_palette_serves_more_series_than_it_has_colours(self) -> None:
        for spec in REGISTRY:
            module = REGISTRY.module(spec.template_id)
            palette = module.PALETTES[0]
            with self.subTest(template=spec.template_id):
                wanted = len(palette.colors) + 3
                taken = palette.take(wanted)
                self.assertEqual(len(taken), wanted)
                self.assertEqual(len(set(taken)), wanted, "extended colours repeat")


class ReferenceFigureTests(unittest.TestCase):
    def test_default_data_validates_and_renders(self) -> None:
        for spec in REGISTRY:
            module = REGISTRY.module(spec.template_id)
            with self.subTest(template=spec.template_id):
                module.DEFAULT_DATA.validate()
                figure = module.create_figure(palette=module.PALETTES[0])
                try:
                    self.assertGreater(len(figure.axes), 0)
                finally:
                    plt.close(figure)

    def test_reference_figure_writes_every_format(self) -> None:
        spec = REGISTRY.specs[0]
        module = REGISTRY.module(spec.template_id)
        figure = module.create_figure()
        with tempfile.TemporaryDirectory() as directory:
            written = save_figure(
                figure,
                Path(directory),
                "probe",
                formats=("png", "svg", "pdf"),
                dpi=30,
            )
            self.assertEqual([path.suffix for path in written], [".png", ".svg", ".pdf"])
            for path in written:
                self.assertGreater(path.stat().st_size, 0)


class DataDrivenTests(unittest.TestCase):
    """The core property: a template must draw data that is not its reference."""

    def test_templates_render_synthetic_data_across_their_range(self) -> None:
        for spec in REGISTRY:
            module = REGISTRY.module(spec.template_id)
            for kind in spec.kinds:
                for categories, series in probe_shapes(spec):
                    label = f"{spec.template_id}/{kind.value}/{categories}x{series}"
                    with self.subTest(case=label):
                        reading = synthetic_reading(kind, categories, series)
                        data = build_data(spec, reading)
                        data.validate()
                        figure = module.create_figure(
                            palette=module.PALETTES[0], data=data
                        )
                        try:
                            self.assertGreater(len(figure.axes), 0)
                        finally:
                            plt.close(figure)

    def test_builder_arguments_cover_the_declared_roles(self) -> None:
        for spec in REGISTRY:
            for kind in spec.kinds:
                categories, series = probe_shapes(spec)[0]
                with self.subTest(template=spec.template_id, kind=kind):
                    reading = synthetic_reading(kind, categories, series)
                    arguments = builder_kwargs(spec, reading)
                    self.assertTrue(arguments, "builder received no arguments")
                    module = REGISTRY.module(spec.template_id)
                    data_class = getattr(module, spec.data_class)
                    builder = getattr(data_class, spec.builder_name)
                    builder(**arguments)

    def test_small_magnitude_data_is_not_drawn_on_a_reference_axis(self) -> None:
        """A template must rescale for data far smaller than its reference figure.

        Several templates pin the reference figure's axis bounds. Data that fits
        inside those bounds but only fills a sliver of them renders as invisible
        marks, which is why ``resolve_limits`` gives the pin up.
        """

        for spec in REGISTRY:
            module = REGISTRY.module(spec.template_id)
            kind = spec.kinds[0]
            categories, series = probe_shapes(spec)[0]
            with self.subTest(template=spec.template_id):
                reading = synthetic_reading(kind, categories, series)
                tiny = replace(
                    reading,
                    values=tuple(
                        tuple(value * 1e-3 for value in row) for row in reading.values
                    ),
                )
                figure = module.create_figure(data=build_data(spec, tiny))
                plt.close(figure)

    def test_long_labels_do_not_crash_a_template(self) -> None:
        for spec in REGISTRY:
            module = REGISTRY.module(spec.template_id)
            kind = spec.kinds[0]
            categories, series = probe_shapes(spec)[0]
            with self.subTest(template=spec.template_id):
                reading = synthetic_reading(
                    kind, categories, series, long_labels=True
                )
                figure = module.create_figure(data=build_data(spec, reading))
                plt.close(figure)

    def test_validate_rejects_a_ragged_matrix(self) -> None:
        """Builders must catch inconsistent input rather than drawing nonsense."""

        checked = 0
        for spec in REGISTRY:
            module = REGISTRY.module(spec.template_id)
            data_class = getattr(module, spec.data_class)
            builder = getattr(data_class, spec.builder_name)
            kind = spec.kinds[0]
            categories, series = probe_shapes(spec)[0]
            arguments = builder_kwargs(spec, synthetic_reading(kind, categories, series))
            values_key = spec.argument_for("values")
            if values_key not in arguments:
                continue
            rows = arguments[values_key]
            if not isinstance(rows, tuple) or len(rows) < 2:
                continue
            if not isinstance(rows[0], tuple) or len(rows[0]) < 2:
                continue
            broken = dict(arguments)
            broken[values_key] = (rows[0][:-1],) + tuple(rows[1:])
            with self.subTest(template=spec.template_id):
                with self.assertRaises(ValueError):
                    builder(**broken)
            checked += 1
        self.assertGreater(checked, 0, "no template exposed a matrix to check")


class ModuleSurfaceTests(unittest.TestCase):
    def test_modules_expose_the_contract_surface(self) -> None:
        for spec in REGISTRY:
            module = REGISTRY.module(spec.template_id)
            with self.subTest(template=spec.template_id):
                for name in (
                    "SPEC",
                    "PALETTES",
                    "DEFAULT_DATA",
                    "DEFAULT_STYLE",
                    "ChartStyle",
                    "create_figure",
                    "main",
                ):
                    self.assertTrue(hasattr(module, name), f"missing {name}")

    def test_modules_do_not_keep_a_private_cli(self) -> None:
        """CLI plumbing lives in rvl.render; a leftover copy will drift."""

        for spec in REGISTRY:
            module = REGISTRY.module(spec.template_id)
            with self.subTest(template=spec.template_id):
                for name in (
                    "palette_from_selector",
                    "render_palette",
                    "build_argument_parser",
                ):
                    self.assertFalse(
                        hasattr(module, name),
                        f"{name} should come from rvl.render, not the template",
                    )

    def test_list_palettes_runs_for_every_template(self) -> None:
        for spec in REGISTRY:
            module = REGISTRY.module(spec.template_id)
            with self.subTest(template=spec.template_id):
                captured = io.StringIO()
                with contextlib.redirect_stdout(captured):
                    self.assertEqual(module.main(["--list-palettes"]), 0)
                self.assertIn(module.PALETTES[0].name, captured.getvalue())


if __name__ == "__main__":
    unittest.main()
