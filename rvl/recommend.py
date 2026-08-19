"""Rank chart templates against a profiled dataset.

Scoring is driven entirely by :class:`rvl.contract.TemplateSpec` metadata, so a
new template becomes rankable by declaring its data kinds, shape limits and
:class:`rvl.contract.Feature` affinities. Nothing here knows any template by name.

The output is a ranked list with a written reason per candidate. The agent makes
the final call using ``references/selection-rubric.md``; this module exists so
that call starts from a deterministic, reproducible shortlist.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Iterable, Sequence

import numpy as np

from .contract import DataKind, Feature, TemplateSpec
from .profiling import DataProfile, Interpretation
from .registry import Registry, load_registry

# Baseline weights applied to every template, so a template that declares no
# affinities still ranks sensibly. Spec-declared weights are added on top.
_BASELINE_AFFINITY: Final[dict[Feature, float]] = {
    Feature.WIDE_DYNAMIC_RANGE: -4.0,
    Feature.LONG_LABELS: -1.0,
}

_KIND_BASELINE: Final[dict[DataKind, dict[Feature, float]]] = {
    DataKind.PARTS_OF_WHOLE: {Feature.SUMS_TO_100: 10.0, Feature.SINGLE_SERIES: 6.0},
    DataKind.STACKED_PARTS: {Feature.NON_NEGATIVE: 6.0},
    DataKind.NESTED_PARTS: {Feature.NON_NEGATIVE: 6.0},
    DataKind.SERIES_WITH_TOTALS: {
        Feature.ORDERED_CATEGORIES: 12.0,
        Feature.HAS_UNCERTAINTY: 5.0,
    },
    DataKind.MATRIX: {Feature.HAS_EMPHASIS: 3.0},
}

_MAX_SHAPE_SCORE: Final[float] = 26.0
_KIND_SCORE: Final[float] = 30.0


@dataclass(frozen=True, slots=True)
class Candidate:
    """One template scored against one reading of the data."""

    spec: TemplateSpec
    interpretation: Interpretation
    score: float
    reasons: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def template_id(self) -> str:
        return self.spec.template_id

    def describe(self) -> str:
        categories, series = shape_of(self.interpretation)
        head = (
            f"{self.template_id}  score {self.score:.1f}  "
            f"({categories} categories x {series} series as "
            f"{self.interpretation.kind})"
        )
        lines = [head]
        lines.extend(f"  + {reason}" for reason in self.reasons)
        lines.extend(f"  ! {warning}" for warning in self.warnings)
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class Recommendation:
    """The ranked shortlist for one profiled table."""

    profile: DataProfile
    candidates: tuple[Candidate, ...]
    rejected: tuple[tuple[str, str], ...]

    @property
    def best(self) -> Candidate | None:
        return self.candidates[0] if self.candidates else None

    def top(self, count: int = 3) -> tuple[Candidate, ...]:
        return self.candidates[:count]

    def for_template(self, template_id: str) -> Candidate | None:
        for candidate in self.candidates:
            if candidate.template_id == template_id:
                return candidate
        return None

    def describe(self, limit: int = 5) -> str:
        lines = [self.profile.summary(), "", "ranked templates:"]
        if not self.candidates:
            lines.append("  (none matched)")
        for position, candidate in enumerate(self.candidates[:limit], start=1):
            lines.append(f"{position}. {candidate.describe()}")
        if self.rejected:
            lines.append("")
            lines.append("not applicable:")
            for template_id, reason in self.rejected:
                lines.append(f"  {template_id}: {reason}")
        return "\n".join(lines)


def shape_of(interpretation: Interpretation) -> tuple[int, int]:
    """The ``(categories, series)`` counts a spec's extents are checked against.

    Most kinds use the interpretation's own grid, but sample-based kinds measure
    their category axis differently: an xy reading counts paired observations and
    a distribution reading counts histogram bins.
    """

    extras = interpretation.extras
    if interpretation.kind is DataKind.XY_SAMPLES:
        pairs = extras.get("x", ())
        smallest = min((len(item) for item in pairs), default=0)
        return smallest, len(pairs) or interpretation.n_series
    if interpretation.kind is DataKind.DISTRIBUTION_SAMPLES:
        samples = extras.get("samples", ())
        smallest = min((len(item) for item in samples), default=0)
        # A histogram needs enough observations per bin to be readable.
        bins = max(4, min(20, int(math.sqrt(max(smallest, 1)))))
        return bins, len(samples) or interpretation.n_series
    if interpretation.kind is DataKind.SET_MEMBERSHIP:
        groups = [group for group in extras.get("groups", ()) if group]
        distinct = len(dict.fromkeys(groups)) if groups else 1
        return len(extras.get("memberships", ())), distinct
    if interpretation.kind is DataKind.SET_OVERLAP:
        return interpretation.n_categories, interpretation.n_categories
    return interpretation.n_categories, interpretation.n_series


def features_of(
    profile: DataProfile, interpretation: Interpretation
) -> frozenset[Feature]:
    """Which :class:`Feature` flags this reading of the data exhibits."""

    categories, series = shape_of(interpretation)
    present: set[Feature] = set()

    if interpretation.ordered_categories:
        present.add(Feature.ORDERED_CATEGORIES)
    if profile.all_non_negative:
        present.add(Feature.NON_NEGATIVE)
    if profile.sums_to_100:
        present.add(Feature.SUMS_TO_100)
    if profile.bounded_unit_scale:
        present.add(Feature.BOUNDED_SCALE)
    if profile.dynamic_range > 100.0:
        present.add(Feature.WIDE_DYNAMIC_RANGE)
    if profile.has_uncertainty:
        present.add(Feature.HAS_UNCERTAINTY)
    if profile.emphasised_cells:
        present.add(Feature.HAS_EMPHASIS)
    if interpretation.categories and max(len(name) for name in interpretation.categories) > 12:
        present.add(Feature.LONG_LABELS)
    if categories > 12:
        present.add(Feature.MANY_CATEGORIES)
    if series == 1:
        present.add(Feature.SINGLE_SERIES)
    if series > 5:
        present.add(Feature.MANY_SERIES)
    return frozenset(present)


def _shape_score(spec: TemplateSpec, categories: int, series: int) -> float:
    """How comfortably the data sits inside the template's supported extents."""

    def comfort(count: int, minimum: int, maximum: int | None) -> float:
        if maximum is None:
            # Unbounded: reward clearing the minimum, then plateau.
            return min(1.0, (count - minimum + 1) / max(minimum, 1) / 2.0 + 0.5)
        if maximum == minimum:
            return 1.0
        span = maximum - minimum
        position = (count - minimum) / span
        # Peak in the middle of the supported band, taper toward the edges.
        return max(0.0, 1.0 - abs(position - 0.45) * 1.6)

    category_fit = comfort(categories, spec.categories.minimum, spec.categories.maximum)
    series_fit = comfort(series, spec.series.minimum, spec.series.maximum)
    return _MAX_SHAPE_SCORE * 0.5 * (category_fit + series_fit)


def _affinity_score(
    spec: TemplateSpec, kind: DataKind, present: frozenset[Feature]
) -> tuple[float, list[str]]:
    weights = dict(_BASELINE_AFFINITY)
    weights.update(_KIND_BASELINE.get(kind, {}))
    weights.update(spec.affinity)

    total = 0.0
    notes: list[str] = []
    for feature, weight in weights.items():
        if feature not in present:
            continue
        total += weight
        if weight > 0:
            notes.append(f"suits data that is {feature.value.replace('_', ' ')}")
        elif weight < 0:
            notes.append(f"weakened by data that is {feature.value.replace('_', ' ')}")
    return total, notes


def _order_penalty(spec: TemplateSpec, present: frozenset[Feature]) -> tuple[float, list[str]]:
    if Feature.ORDERED_CATEGORIES not in present:
        return 0.0, []
    if spec.ordered_categories:
        return 6.0, ["keeps the reading order of an ordered category axis"]
    if spec.geometry.value in {"polar", "circular"}:
        return -10.0, [
            "wraps an ordered axis around a circle, which hides where the sequence "
            "starts and ends"
        ]
    return -3.0, ["does not present the category order as a sequence"]


def _label_penalty(
    spec: TemplateSpec, interpretation: Interpretation
) -> tuple[float, list[str]]:
    if spec.long_category_labels or not interpretation.categories:
        return 0.0, []
    longest = max(len(name) for name in interpretation.categories)
    if longest <= 12:
        return 0.0, []
    return -min(8.0, 0.4 * (longest - 12)), [
        f"category labels up to {longest} characters have to be squeezed into a "
        "tight slot"
    ]


def score_candidate(
    spec: TemplateSpec, profile: DataProfile, interpretation: Interpretation
) -> Candidate | None:
    """Score one template against one reading, or ``None`` if it cannot apply."""

    if interpretation.kind not in spec.kinds:
        return None
    categories, series = shape_of(interpretation)
    if not spec.accepts_shape(categories=categories, series=series):
        return None
    present = features_of(profile, interpretation)
    missing = [feature for feature in spec.requires if feature not in present]
    if missing:
        return None

    reasons: list[str] = [
        f"accepts {interpretation.kind.value} data at "
        f"{categories} categories x {series} series "
        f"(supports {spec.describe_shape()})"
    ]
    warnings: list[str] = []

    # A reading that is merely arithmetically legal should not beat one that is
    # actually evidenced, so the kind score is scaled by the reading's confidence.
    score = _KIND_SCORE * interpretation.confidence
    if interpretation.confidence < 0.75:
        warnings.append(
            f"this reading of the data is speculative (confidence "
            f"{interpretation.confidence:.0%}); check the note below it in `profile`"
        )
    shape = _shape_score(spec, categories, series)
    score += shape
    if shape >= 0.8 * _MAX_SHAPE_SCORE:
        reasons.append("the data sits in the middle of this template's comfortable range")
    elif shape <= 0.35 * _MAX_SHAPE_SCORE:
        warnings.append("the data sits near the edge of what this template handles well")

    affinity, affinity_notes = _affinity_score(spec, interpretation.kind, present)
    score += affinity
    for note in affinity_notes:
        (reasons if note.startswith("suits") else warnings).append(note)

    order, order_notes = _order_penalty(spec, present)
    score += order
    for note in order_notes:
        (reasons if order > 0 else warnings).append(note)

    labels, label_notes = _label_penalty(spec, interpretation)
    score += labels
    warnings.extend(label_notes)

    if Feature.WIDE_DYNAMIC_RANGE in present:
        warnings.append(
            f"values span {profile.dynamic_range:.0f}x, so small ones will be hard "
            "to see without a log scale"
        )

    return Candidate(
        spec=spec,
        interpretation=interpretation,
        score=round(score, 2),
        reasons=tuple(dict.fromkeys(reasons)),
        warnings=tuple(dict.fromkeys(warnings)),
    )


def recommend(
    profile: DataProfile,
    *,
    registry: Registry | None = None,
    per_template: int = 1,
) -> Recommendation:
    """Rank every template against every reading of the profiled table.

    Only the best-scoring reading per template is kept by default, so the
    shortlist reads as a list of charts rather than a list of pivots.
    """

    resolved = registry or load_registry()
    best: dict[str, Candidate] = {}
    alternates: dict[str, list[Candidate]] = {}
    rejected: list[tuple[str, str]] = []

    for spec in resolved:
        scored: list[Candidate] = []
        for interpretation in profile.interpretations:
            candidate = score_candidate(spec, profile, interpretation)
            if candidate is not None:
                scored.append(candidate)
        if not scored:
            rejected.append((spec.template_id, _rejection_reason(spec, profile)))
            continue
        scored.sort(key=lambda item: item.score, reverse=True)
        best[spec.template_id] = scored[0]
        alternates[spec.template_id] = scored[1:per_template]

    candidates = list(best.values())
    for extra in alternates.values():
        candidates.extend(extra)
    candidates.sort(key=lambda item: (-item.score, item.template_id))
    return Recommendation(
        profile=profile,
        candidates=tuple(candidates),
        rejected=tuple(sorted(rejected)),
    )


def _rejection_reason(spec: TemplateSpec, profile: DataProfile) -> str:
    kinds = ", ".join(kind.value for kind in spec.kinds)
    available = ", ".join(kind.value for kind in profile.kinds()) or "none"
    if not any(kind in spec.kinds for kind in profile.kinds()):
        return f"needs {kinds} data; this table reads as {available}"

    shapes = [
        shape_of(item) for item in profile.interpretations if item.kind in spec.kinds
    ]
    if shapes and not any(
        spec.accepts_shape(categories=categories, series=series)
        for categories, series in shapes
    ):
        got = "; ".join(f"{categories}x{series}" for categories, series in shapes)
        return f"supports {spec.describe_shape()}, but the data is {got}"
    if spec.requires:
        needed = ", ".join(feature.value for feature in spec.requires)
        return f"requires data that is {needed}"
    return "no reading of this table fits"


def rank_tables(
    profiles: Iterable[DataProfile], *, registry: Registry | None = None
) -> tuple[Recommendation, ...]:
    resolved = registry or load_registry()
    return tuple(recommend(profile, registry=resolved) for profile in profiles)
