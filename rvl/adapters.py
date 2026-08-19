"""Turn a reading of the data into keyword arguments for a template's builder.

This is the single place that knows how each :class:`~rvl.contract.DataKind` maps
onto builder parameters. :mod:`rvl.codegen` renders the result as literals and the
test suite calls the builders with it directly, so generated code and tested code
travel the same path.

Templates that name a parameter differently declare it in
``TemplateSpec.argument_names``; nothing here is keyed by template id.
"""

from __future__ import annotations

from typing import Any, Callable, Final, Mapping

import numpy as np

from .contract import DataKind, TemplateSpec
from .profiling import Interpretation


class AdapterError(RuntimeError):
    """The reading does not carry what this data kind needs."""


def _rename(spec: TemplateSpec, arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the template's role-to-parameter overrides."""

    return {spec.argument_for(role): value for role, value in arguments.items()}


def _matrix(spec: TemplateSpec, reading: Interpretation) -> dict[str, Any]:
    values = reading.transposed_values() if spec.transpose_values else reading.values
    arguments: dict[str, Any] = {
        "categories": reading.categories,
        "series": reading.series,
        "values": values,
        "value_label": reading.value_label,
    }
    return _rename(spec, arguments)


def _parts_of_whole(spec: TemplateSpec, reading: Interpretation) -> dict[str, Any]:
    return _rename(
        spec,
        {
            "categories": reading.categories,
            "values": tuple(row[0] for row in reading.values),
            "value_label": reading.value_label,
        },
    )


def _stacked_parts(spec: TemplateSpec, reading: Interpretation) -> dict[str, Any]:
    return _rename(
        spec,
        {
            "categories": reading.categories,
            "series": reading.series,
            "values": reading.values,
            "value_label": reading.value_label,
        },
    )


def _nested_parts(spec: TemplateSpec, reading: Interpretation) -> dict[str, Any]:
    centre = tuple(
        float(value) for value in np.nansum(reading.matrix(), axis=0).tolist()
    )
    return _rename(
        spec,
        {
            "parts": reading.series,
            "center": centre,
            "rings": reading.categories,
            "ring_values": reading.values,
        },
    )


def _series_with_totals(spec: TemplateSpec, reading: Interpretation) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "series": reading.series,
        "points": reading.categories,
        "values": reading.transposed_values(),
    }
    errors = reading.extras.get("errors")
    if errors is not None:
        arguments["errors"] = tuple(
            tuple(errors[category][series] for category in range(reading.n_categories))
            for series in range(reading.n_series)
        )
    arguments["value_label"] = reading.value_label
    return _rename(spec, arguments)


def _xy_samples(spec: TemplateSpec, reading: Interpretation) -> dict[str, Any]:
    extras = reading.extras
    if "x" not in extras or "y" not in extras:
        raise AdapterError("xy reading carries no paired observations")
    return _rename(
        spec,
        {
            "series": reading.series,
            "x": extras["x"],
            "y": extras["y"],
            "x_label": extras.get("x_label", "x"),
            "y_label": extras.get("y_label", "y"),
        },
    )


def _distribution_samples(spec: TemplateSpec, reading: Interpretation) -> dict[str, Any]:
    extras = reading.extras
    if "samples" not in extras:
        raise AdapterError("distribution reading carries no samples")
    return _rename(
        spec,
        {
            "series": reading.series,
            "samples": extras["samples"],
            "value_label": extras.get("value_label", reading.value_label),
        },
    )


def _set_membership(spec: TemplateSpec, reading: Interpretation) -> dict[str, Any]:
    extras = reading.extras
    for key in ("sets", "memberships", "counts"):
        if key not in extras:
            raise AdapterError(f"membership reading carries no {key!r}")
    arguments: dict[str, Any] = {
        "sets": extras["sets"],
        "memberships": extras["memberships"],
        "counts": extras["counts"],
    }
    groups = tuple(extras.get("groups", ()))
    if groups and any(groups):
        arguments["groups"] = groups
    arguments["count_label"] = extras.get("count_label", "Count")
    return _rename(spec, arguments)


def _set_overlap(spec: TemplateSpec, reading: Interpretation) -> dict[str, Any]:
    return _rename(
        spec,
        {
            "groups": reading.categories,
            "totals": tuple(row[0] for row in reading.values),
            "uniques": tuple(row[1] for row in reading.values),
            "core": float(reading.extras.get("core", 0.0)),
        },
    )


_ADAPTERS: Final[
    Mapping[DataKind, Callable[[TemplateSpec, Interpretation], dict[str, Any]]]
] = {
    DataKind.MATRIX: _matrix,
    DataKind.PARTS_OF_WHOLE: _parts_of_whole,
    DataKind.STACKED_PARTS: _stacked_parts,
    DataKind.NESTED_PARTS: _nested_parts,
    DataKind.SERIES_WITH_TOTALS: _series_with_totals,
    DataKind.XY_SAMPLES: _xy_samples,
    DataKind.DISTRIBUTION_SAMPLES: _distribution_samples,
    DataKind.SET_MEMBERSHIP: _set_membership,
    DataKind.SET_OVERLAP: _set_overlap,
}


def supported_kinds() -> frozenset[DataKind]:
    return frozenset(_ADAPTERS)


def _resolve_builder(spec: TemplateSpec) -> Any:
    import importlib

    module = importlib.import_module(spec.module_name)
    data_class = getattr(module, spec.data_class)
    return getattr(data_class, spec.builder_name)


def _accepted_parameters(spec: TemplateSpec) -> frozenset[str]:
    import inspect

    try:
        signature = inspect.signature(_resolve_builder(spec))
    except (TypeError, ValueError):  # pragma: no cover - builders are plain functions
        return frozenset()
    return frozenset(signature.parameters)


def _apply_hints(
    spec: TemplateSpec, reading: Interpretation, arguments: dict[str, Any]
) -> dict[str, Any]:
    """Pass on what the profiler inferred, for builders that can use it.

    Two inferences matter enough to forward automatically. A metric whose name says
    smaller is better must set ``lower_is_better``, or a longer bar will mean a
    worse result. A source that marks winners in bold identifies a series worth
    highlighting. Builders that do not take these arguments simply do not get them.
    """

    accepted = _accepted_parameters(spec)

    lower_is_better = reading.extras.get("lower_is_better")
    if (
        lower_is_better is not None
        and "lower_is_better" in accepted
        and "lower_is_better" not in arguments
    ):
        arguments["lower_is_better"] = bool(lower_is_better)

    emphasised = reading.extras.get("emphasised")
    if emphasised and "highlight" in accepted and "highlight" not in arguments:
        winner = max(emphasised, key=lambda name: len(emphasised[name]))
        if emphasised[winner] and winner in reading.series:
            arguments["highlight"] = winner

    return arguments


def builder_kwargs(spec: TemplateSpec, reading: Interpretation) -> dict[str, Any]:
    """Keyword arguments for ``spec.builder`` that reproduce ``reading``.

    Keys are already renamed to the template's own parameter names, and values are
    plain Python containers so they can be either called directly or rendered as
    source literals.
    """

    adapter = _ADAPTERS.get(reading.kind)
    if adapter is None:
        raise AdapterError(f"no adapter for data kind {reading.kind}")
    if reading.kind not in spec.kinds:
        raise AdapterError(
            f"{spec.template_id} does not accept {reading.kind.value} data"
        )
    return _apply_hints(spec, reading, adapter(spec, reading))


def build_data(spec: TemplateSpec, reading: Interpretation) -> Any:
    """Call the template's builder and return the validated data object."""

    return _resolve_builder(spec)(**builder_kwargs(spec, reading))
