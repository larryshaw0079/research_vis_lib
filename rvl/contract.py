"""Declarative metadata every chart template publishes as ``SPEC``.

The recommender never imports drawing code to decide what fits a dataset; it
reads these specs.  Adding a template therefore means dropping a module into
``rvl/templates`` with a ``SPEC`` — no central list to edit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final, Sequence


class DataKind(StrEnum):
    """The shape of experimental data a template can faithfully draw."""

    MATRIX = "matrix"
    """Numeric value per (category, series) pair. Nothing has to sum to a total."""

    PARTS_OF_WHOLE = "parts_of_whole"
    """One series of non-negative shares over categories, read as a composition."""

    NESTED_PARTS = "nested_parts"
    """A top-level composition plus a per-sector breakdown over the same keys."""

    STACKED_PARTS = "stacked_parts"
    """Per-category segments that stack into a category total."""

    XY_SAMPLES = "xy_samples"
    """Paired (x, y) observations per series, for regression or correlation."""

    DISTRIBUTION_SAMPLES = "distribution_samples"
    """Many raw observations per series, summarised as a distribution."""

    SET_MEMBERSHIP = "set_membership"
    """Boolean membership over set columns plus a count per combination."""

    SET_OVERLAP = "set_overlap"
    """Per-series total and unique counts plus one shared core count."""

    SERIES_WITH_TOTALS = "series_with_totals"
    """An ordered per-series measurement sequence plus a per-series aggregate."""


class Geometry(StrEnum):
    """Coarse visual family, used to diversify ranked recommendations."""

    CARTESIAN = "cartesian"
    POLAR = "polar"
    CIRCULAR = "circular"
    COMPOSITE = "composite"


class Feature(StrEnum):
    """Properties of a dataset that make a template more or less suitable.

    A template declares weights over these in ``TemplateSpec.affinities``, so the
    recommender never needs to know template names. Adding a template with new
    preferences is a matter of declaring weights, not editing the scorer.
    """

    ORDERED_CATEGORIES = "ordered_categories"
    """The category axis is a sequence, such as dates, whose order carries meaning."""

    NON_NEGATIVE = "non_negative"
    """Every value is at or above zero."""

    SUMS_TO_100 = "sums_to_100"
    """Values already form a percentage composition."""

    BOUNDED_SCALE = "bounded_scale"
    """Values sit on a bounded metric scale such as 0-1 or 0-100."""

    WIDE_DYNAMIC_RANGE = "wide_dynamic_range"
    """Largest and smallest magnitudes differ by more than about two decades."""

    HAS_UNCERTAINTY = "has_uncertainty"
    """The source supplies errors, standard deviations or confidence intervals."""

    HAS_EMPHASIS = "has_emphasis"
    """The source marks a winning value, e.g. bold cells in a benchmark table."""

    LONG_LABELS = "long_labels"
    """Category labels are too long to sit in a tight slot."""

    MANY_CATEGORIES = "many_categories"
    """More than about a dozen categories."""

    SINGLE_SERIES = "single_series"
    """Exactly one series, so no colour comparison is needed."""

    MANY_SERIES = "many_series"
    """More than about five series."""


def feature_weights(
    pairs: Sequence[tuple[Feature, float]],
) -> dict[Feature, float]:
    """Collapse declared affinity pairs into a lookup, rejecting duplicates."""

    weights: dict[Feature, float] = {}
    for feature, weight in pairs:
        if feature in weights:
            raise ValueError(f"duplicate affinity for {feature}")
        weights[feature] = float(weight)
    return weights


@dataclass(frozen=True, slots=True)
class Extent:
    """Inclusive count range a template supports. ``None`` means unbounded."""

    minimum: int
    maximum: int | None = None

    def __post_init__(self) -> None:
        if self.minimum < 0:
            raise ValueError("minimum must be non-negative")
        if self.maximum is not None and self.maximum < self.minimum:
            raise ValueError("maximum must not be below minimum")

    def accepts(self, count: int) -> bool:
        if count < self.minimum:
            return False
        return self.maximum is None or count <= self.maximum

    def describe(self) -> str:
        if self.maximum is None:
            return f"{self.minimum}+"
        if self.maximum == self.minimum:
            return str(self.minimum)
        return f"{self.minimum}-{self.maximum}"


@dataclass(frozen=True, slots=True)
class TemplateSpec:
    """What a template draws, what data it needs, and when it is a good fit."""

    template_id: str
    title: str
    summary: str
    kinds: tuple[DataKind, ...]
    geometry: Geometry
    categories: Extent
    series: Extent
    builder: str
    data_contract: str
    good_for: tuple[str, ...] = ()
    avoid_when: tuple[str, ...] = ()
    ordered_categories: bool = False
    """True when the category axis carries meaning in order, e.g. dates."""

    long_category_labels: bool = True
    """False when the layout only has room for short category labels."""

    affinities: tuple[tuple[Feature, float], ...] = ()
    """Signed weights over :class:`Feature`; positive attracts, negative repels."""

    requires: tuple[Feature, ...] = ()
    """Features a dataset must have, else the template is not offered at all."""

    argument_names: tuple[tuple[str, str], ...] = ()
    """Overrides mapping a generic role to this builder's parameter name.

    The code generator addresses builders by role — ``categories``, ``series``,
    ``values`` and so on — so a template that calls its category axis ``spokes``
    declares ``(("categories", "spokes"),)`` and needs no special-casing.
    """

    transpose_values: bool = False
    """True when the builder wants ``values[series][category]`` instead of
    ``values[category][series]``."""

    default_dpi: int = 200
    reference: str = ""
    module: str = field(default="", compare=False)
    palette_count: int = field(default=0, compare=False)

    def __post_init__(self) -> None:
        if not self.template_id or self.template_id != self.template_id.strip():
            raise ValueError("template_id must be a non-empty, trimmed string")
        if self.template_id != self.template_id.lower():
            raise ValueError("template_id must be lowercase")
        if not self.kinds:
            raise ValueError(f"{self.template_id}: kinds must not be empty")
        if len(set(self.kinds)) != len(self.kinds):
            raise ValueError(f"{self.template_id}: kinds must be unique")
        if not self.builder:
            raise ValueError(f"{self.template_id}: builder must name a constructor")
        if not self.data_contract.strip():
            raise ValueError(f"{self.template_id}: data_contract must not be empty")
        feature_weights(self.affinities)
        if len(set(self.requires)) != len(self.requires):
            raise ValueError(f"{self.template_id}: requires must be unique")

    @property
    def affinity(self) -> dict[Feature, float]:
        return feature_weights(self.affinities)

    def argument_for(self, role: str) -> str:
        """This builder's parameter name for a generic role."""

        for declared_role, parameter in self.argument_names:
            if declared_role == role:
                return parameter
        return role

    @property
    def module_name(self) -> str:
        """Import path of the module that owns this template."""

        return self.module or f"rvl.templates.{self.template_id.replace('-', '_')}"

    @property
    def data_class(self) -> str:
        """Name of the data class the builder is attached to."""

        return self.builder.split(".", 1)[0]

    @property
    def builder_name(self) -> str:
        """Name of the constructor used to build data from a real table."""

        return self.builder.split(".", 1)[1] if "." in self.builder else self.builder

    def accepts_shape(self, *, categories: int, series: int) -> bool:
        return self.categories.accepts(categories) and self.series.accepts(series)

    def describe_shape(self) -> str:
        return (
            f"{self.categories.describe()} categories x "
            f"{self.series.describe()} series"
        )


SUPPORTED_FORMATS: Final[tuple[str, ...]] = ("png", "svg", "pdf")


# Builder arguments that carry presentation rather than measurements. Their values
# are chosen by the generator or the author, so the fidelity verifier must not
# expect them to appear in the data file.
_PRESENTATION_SUFFIXES: Final[tuple[str, ...]] = (
    "_label",
    "_format",
    "_color",
    "_colors",
    "_limits",
)

_PRESENTATION_ARGUMENTS: Final[frozenset[str]] = frozenset(
    {
        "bins",
        "confidence_level",
        "highlight",
        "length_scale",
        "lower_is_better",
        "normalize",
        "normalize_center",
        "period_boundaries",
        "scale_max",
        "show_confidence_band",
        "show_normal_fit",
        "show_totals",
        "significance",
        "start_angle",
        "sweep_degrees",
        "title",
    }
)


def is_presentation_argument(name: str) -> bool:
    """True when a builder argument styles the figure instead of supplying data."""

    return name in _PRESENTATION_ARGUMENTS or name.endswith(_PRESENTATION_SUFFIXES)
