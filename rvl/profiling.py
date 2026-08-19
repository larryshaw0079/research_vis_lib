"""Summarise a ``Table`` and enumerate the ways it can be read as chart data.

A table rarely admits only one reading: a label column plus several numeric
columns is a matrix, but if the values are non-negative it is also a stackable
composition, and if the label column is a date it is also an ordered series.
:func:`interpretations` enumerates those readings so :mod:`rvl.recommend` can
score templates against concrete extractions rather than guesses.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from typing import Any, Final, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from .contract import DataKind
from .ingest import Column, ColumnKind, Table, is_missing

_TOTAL_HINTS: Final[tuple[str, ...]] = ("total", "all", "sum", "count", "n_")
_UNIQUE_HINTS: Final[tuple[str, ...]] = ("unique", "specific", "exclusive", "only")
_CORE_HINTS: Final[tuple[str, ...]] = ("core", "shared", "common", "intersection")
_ERROR_HINTS: Final[tuple[str, ...]] = (
    "sd",
    "std",
    "stdev",
    "sem",
    "se",
    "err",
    "error",
    "ci",
    "sigma",
    "±",
)
_LOWER_BETTER_HINTS: Final[tuple[str, ...]] = (
    "mse",
    "mae",
    "rmse",
    "loss",
    "error",
    "err",
    "wer",
    "cer",
    "perplexity",
    "ppl",
    "fid",
    "latency",
    "runtime",
    "cost",
    "rank",
)
_BOUNDED_HINTS: Final[tuple[str, ...]] = (
    "auroc",
    "auc",
    "accuracy",
    "acc",
    "f1",
    "precision",
    "recall",
    "iou",
    "dice",
    "r2",
    "correlation",
    "share",
    "percent",
    "proportion",
    "rate",
)

_PERCENT_TOTAL_TOLERANCE: Final[float] = 1.5


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _mentions(text: str, hints: Sequence[str]) -> bool:
    haystack = _normalise(text)
    tokens = set(haystack.split())
    return any(hint in tokens or hint in haystack for hint in hints)


@dataclass(frozen=True, slots=True)
class Interpretation:
    """One concrete way to read a table as a template's input.

    ``values`` is always indexed ``[category][series]``; ``references`` mirrors it
    with the source cell each number came from. ``extras`` carries kind-specific
    payloads such as parsed uncertainties, bubble sizes, per-series raw samples,
    or set membership flags.
    """

    kind: DataKind
    table: str
    categories: tuple[str, ...]
    series: tuple[str, ...]
    values: tuple[tuple[float, ...], ...]
    references: tuple[tuple[str, ...], ...]
    source: str = ""
    confidence: float = 1.0
    """How well-evidenced this reading is, in ``(0, 1]``.

    Some readings are structurally possible but semantically unlikely. Five model
    columns of MSE are non-negative, so they *can* be stacked, but their sum means
    nothing. Confidence below 1 marks a reading as speculative so the recommender
    prefers a sound interpretation over a merely legal one.
    """

    category_label: str = ""
    value_label: str = "Value"
    category_column: str | None = None
    value_columns: tuple[str, ...] = ()
    ordered_categories: bool = False
    notes: tuple[str, ...] = ()
    extras: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 < self.confidence <= 1.0:
            raise ValueError("confidence must sit in (0, 1]")
        if len(self.values) != len(self.categories):
            raise ValueError(
                f"{self.kind}: {len(self.values)} value rows for "
                f"{len(self.categories)} categories"
            )
        for row in self.values:
            if len(row) != len(self.series):
                raise ValueError(
                    f"{self.kind}: value row of width {len(row)} for "
                    f"{len(self.series)} series"
                )
        if len(self.references) != len(self.values):
            raise ValueError(f"{self.kind}: references do not mirror values")

    @property
    def n_categories(self) -> int:
        return len(self.categories)

    @property
    def n_series(self) -> int:
        return len(self.series)

    def matrix(self) -> NDArray[np.float64]:
        return np.asarray(self.values, dtype=float)

    def finite(self) -> NDArray[np.float64]:
        matrix = self.matrix()
        return matrix[np.isfinite(matrix)]

    def transposed_values(self) -> tuple[tuple[float, ...], ...]:
        """Values re-indexed ``[series][category]`` for templates that want rows."""

        return tuple(
            tuple(self.values[category][series] for category in range(self.n_categories))
            for series in range(self.n_series)
        )

    def all_references(self) -> tuple[str, ...]:
        return tuple(
            reference for row in self.references for reference in row if reference
        )

    def describe(self) -> str:
        confidence = (
            "" if self.confidence >= 1.0 else f"  [confidence {self.confidence:.0%}]"
        )
        return (
            f"{self.kind}: {self.n_categories} categories x {self.n_series} series "
            f"from {self.table}{confidence}"
        )


@dataclass(frozen=True, slots=True)
class DataProfile:
    """What the recommender needs to know about a table."""

    table: str
    source: str
    n_rows: int
    n_columns: int
    label_columns: tuple[str, ...]
    numeric_columns: tuple[str, ...]
    boolean_columns: tuple[str, ...]
    temporal_columns: tuple[str, ...]
    error_columns: tuple[str, ...]
    interpretations: tuple[Interpretation, ...]
    value_min: float
    value_max: float
    all_non_negative: bool
    sums_to_100: bool
    bounded_unit_scale: bool
    dynamic_range: float
    lower_is_better: bool
    has_uncertainty: bool
    emphasised_cells: int
    observations_per_label: float
    notes: tuple[str, ...] = ()

    def kinds(self) -> tuple[DataKind, ...]:
        seen: list[DataKind] = []
        for interpretation in self.interpretations:
            if interpretation.kind not in seen:
                seen.append(interpretation.kind)
        return tuple(seen)

    def for_kind(self, kind: DataKind) -> tuple[Interpretation, ...]:
        return tuple(item for item in self.interpretations if item.kind == kind)

    def summary(self) -> str:
        lines = [
            f"table: {self.table}  ({self.n_rows} rows x {self.n_columns} columns)",
            f"source: {self.source}",
            f"label columns:   {', '.join(self.label_columns) or '(none)'}",
            f"numeric columns: {', '.join(self.numeric_columns) or '(none)'}",
        ]
        if self.boolean_columns:
            lines.append(f"boolean columns: {', '.join(self.boolean_columns)}")
        if self.temporal_columns:
            lines.append(f"ordered columns: {', '.join(self.temporal_columns)}")
        if self.error_columns:
            lines.append(f"error columns:   {', '.join(self.error_columns)}")
        lines.append(
            f"values: [{self.value_min:.4g}, {self.value_max:.4g}]  "
            f"non-negative={self.all_non_negative}  sums-to-100={self.sums_to_100}  "
            f"unit-scale={self.bounded_unit_scale}"
        )
        lines.append(
            f"dynamic range: {self.dynamic_range:.4g}x  "
            f"lower-is-better={self.lower_is_better}  "
            f"uncertainty={self.has_uncertainty}  "
            f"emphasised cells={self.emphasised_cells}"
        )
        lines.append("readings:")
        for interpretation in self.interpretations:
            lines.append(f"  - {interpretation.describe()}")
            for note in interpretation.notes:
                lines.append(f"      {note}")
        for note in self.notes:
            lines.append(f"note: {note}")
        return "\n".join(lines)


def _is_error_column(column: Column) -> bool:
    return _mentions(column.name, _ERROR_HINTS)


def _pair_error_column(
    value_column: Column, candidates: Sequence[Column]
) -> Column | None:
    """Find an error column that clearly belongs to ``value_column``."""

    stem = _normalise(value_column.name)
    best: Column | None = None
    for candidate in candidates:
        name = _normalise(candidate.name)
        if not _is_error_column(candidate):
            continue
        if stem and stem in name:
            return candidate
        if best is None:
            best = candidate
    return best


def _label_values(column: Column) -> tuple[str, ...]:
    return column.labels()


def _wide_interpretations(table: Table) -> list[Interpretation]:
    """Readings where numeric column headers are the series."""

    labels = table.label_columns()
    numerics = [column for column in table.numeric_columns() if not _is_error_column(column)]
    errors = [column for column in table.numeric_columns() if _is_error_column(column)]
    if not labels or not numerics:
        return []

    category_column = labels[0]
    categories = _label_values(category_column)
    if len(set(categories)) != len(categories):
        # Repeated labels mean this is long-form data, handled elsewhere.
        return []

    values = tuple(
        tuple(float(column.numeric()[row]) for column in numerics)
        for row in range(table.n_rows)
    )
    references = tuple(
        tuple(column.references[row] for column in numerics)
        for row in range(table.n_rows)
    )
    series = tuple(column.name for column in numerics)
    matrix = np.asarray(values, dtype=float)
    ordered = category_column.kind == ColumnKind.TEMPORAL

    extras: dict[str, Any] = {}
    paired_errors = [_pair_error_column(column, errors) for column in numerics]
    if any(item is not None for item in paired_errors):
        extras["errors"] = tuple(
            tuple(
                float("nan")
                if paired_errors[index] is None
                else float(paired_errors[index].numeric()[row])
                for index in range(len(numerics))
            )
            for row in range(table.n_rows)
        )
    inline_errors = [column.errors() for column in numerics]
    if any(item is not None for item in inline_errors) and "errors" not in extras:
        extras["errors"] = tuple(
            tuple(
                float("nan")
                if inline_errors[index] is None
                else float(inline_errors[index][row])
                for index in range(len(numerics))
            )
            for row in range(table.n_rows)
        )
    emphasised = {
        numerics[index].name: numerics[index].emphasised_rows()
        for index in range(len(numerics))
    }
    if any(emphasised.values()):
        extras["emphasised"] = emphasised
    extras["lower_is_better"] = _lower_is_better(table, numerics)

    base_notes = (
        f"categories from column {category_column.name!r}; "
        f"series from the headers of {', '.join(series)}",
    )
    results = [
        Interpretation(
            kind=DataKind.MATRIX,
            table=table.name,
            categories=categories,
            series=series,
            values=values,
            references=references,
            category_label=category_column.name,
            value_label=_shared_value_label(series, table.name),
            category_column=category_column.name,
            value_columns=series,
            ordered_categories=ordered,
            notes=base_notes,
            extras=extras,
        )
    ]

    finite = matrix[np.isfinite(matrix)]
    if finite.size and float(finite.min()) >= 0.0 and len(series) >= 2:
        confidence, evidence = _composition_evidence(matrix)
        results.append(
            Interpretation(
                kind=DataKind.STACKED_PARTS,
                table=table.name,
                categories=categories,
                series=series,
                values=values,
                references=references,
                confidence=confidence,
                category_label=category_column.name,
                value_label=_shared_value_label(series, table.name),
                category_column=category_column.name,
                value_columns=series,
                ordered_categories=ordered,
                notes=base_notes + (evidence,),
                extras=extras,
            )
        )

    if ordered and len(series) >= 2:
        results.append(
            Interpretation(
                kind=DataKind.SERIES_WITH_TOTALS,
                table=table.name,
                categories=categories,
                series=series,
                values=values,
                references=references,
                category_label=category_column.name,
                value_label=_shared_value_label(series, table.name),
                category_column=category_column.name,
                value_columns=series,
                ordered_categories=True,
                notes=base_notes
                + (
                    f"column {category_column.name!r} is ordered, so each series is a "
                    "sequence whose aggregate can be shown alongside it",
                ),
                extras=extras,
            )
        )

    if len(series) == 1 and finite.size and float(finite.min()) >= 0.0:
        results.append(
            Interpretation(
                kind=DataKind.PARTS_OF_WHOLE,
                table=table.name,
                categories=categories,
                series=series,
                values=values,
                references=references,
                category_label=category_column.name,
                value_label=series[0],
                category_column=category_column.name,
                value_columns=series,
                notes=(
                    f"single non-negative measure {series[0]!r} over "
                    f"{len(categories)} categories, readable as shares of their total",
                ),
                extras=extras,
            )
        )

    overlap = _overlap_interpretation(table, category_column, numerics)
    if overlap is not None:
        results.append(overlap)

    return results


def _composition_evidence(matrix: NDArray[np.float64]) -> tuple[float, str]:
    """How likely a wide table's columns are parts of a per-row total.

    Competing measurements of the same quantity — five models' MSE — are
    non-negative and therefore stackable in the arithmetic sense, but their sum is
    meaningless. Real compositions leave a trace: per-row totals that are similar
    across rows, or that land on 100.
    """

    totals = np.nansum(matrix, axis=1)
    totals = totals[np.isfinite(totals) & (totals > 0)]
    if totals.size < 2:
        return 0.5, "columns may or may not be parts of a per-category total"
    if np.all(np.abs(totals - 100.0) <= _PERCENT_TOTAL_TOLERANCE):
        return 1.0, "every category sums to 100, so the columns are shares of a whole"
    spread = float(totals.std() / totals.mean())
    if spread <= 0.15:
        return (
            0.9,
            "per-category totals are consistent, which is what a composition looks like",
        )
    return (
        0.5,
        "columns are only assumed to be parts of a total; confirm that summing them "
        "means something before using this reading",
    )


def _shared_value_label(series: Sequence[str], table_name: str = "") -> str:
    """A label for what the numeric columns measure.

    Column headers are often the series names — five model names, say — and carry
    no unit. In that case the metric is usually named by the sheet or table
    instead, so ``table_name`` is used before falling back to a placeholder the
    caller is expected to replace.
    """

    if len(series) == 1:
        return series[0]
    tokens = [set(_normalise(name).split()) for name in series]
    shared = set.intersection(*tokens) if tokens else set()
    if shared:
        return " ".join(sorted(shared)).strip().title()
    candidate = table_name.strip()
    if candidate and len(candidate) <= 40 and _mentions(
        candidate, _LOWER_BETTER_HINTS + _BOUNDED_HINTS
    ):
        return candidate
    return "Value"


def _lower_is_better(table: Table, numerics: Sequence[Column]) -> bool:
    """Whether a smaller number is a better result, from the metric's name."""

    naming = [table.name, *(column.name for column in numerics)]
    labels = table.label_columns()
    if labels:
        naming.append(labels[0].name)
    return any(_mentions(text, _LOWER_BETTER_HINTS) for text in naming)


def _xy_interpretations(table: Table) -> list[Interpretation]:
    """Read two numeric columns as paired observations, optionally grouped.

    This is independent of the wide-table reading: a file with a repeating group
    column is not a matrix but is very often an xy scatter.
    """

    numerics = [column for column in table.numeric_columns() if not _is_error_column(column)]
    labels = table.label_columns()
    if len(numerics) < 2:
        return []

    x_column, y_column = numerics[0], numerics[1]
    group_column = None
    for column in labels:
        distinct = column.distinct()
        if 1 < len(distinct) <= 8 and len(distinct) < table.n_rows:
            group_column = column
            break

    x_values = x_column.numeric()
    y_values = y_column.numeric()
    if group_column is None:
        series = ("all",)
        buckets = {"all": list(range(table.n_rows))}
    else:
        series_labels = group_column.labels()
        buckets = {}
        for row, label in enumerate(series_labels):
            if not label:
                continue
            buckets.setdefault(label, []).append(row)
        series = tuple(buckets)

    usable = {
        label: [row for row in rows if np.isfinite(x_values[row]) and np.isfinite(y_values[row])]
        for label, rows in buckets.items()
    }
    if any(len(rows) < 3 for rows in usable.values()):
        return []

    extras = {
        "x": tuple(tuple(float(x_values[row]) for row in usable[label]) for label in series),
        "y": tuple(tuple(float(y_values[row]) for row in usable[label]) for label in series),
        "x_references": tuple(
            tuple(x_column.references[row] for row in usable[label]) for label in series
        ),
        "y_references": tuple(
            tuple(y_column.references[row] for row in usable[label]) for label in series
        ),
        "x_label": x_column.name,
        "y_label": y_column.name,
    }
    # The category axis of an xy reading is the observation index, so expose the
    # per-series pair count as the "matrix" for shape scoring.
    # A tidy file — a group column repeating over many rows — really is paired
    # observations. Two columns out of a wide benchmark table are two competing
    # methods, and plotting one against the other invents a relationship.
    if group_column is not None and len(group_column.distinct()) < table.n_rows:
        confidence = 1.0
        evidence = ""
    elif len(numerics) == 2:
        confidence = 0.6
        evidence = (
            "the only two numeric columns are treated as an x/y pair; confirm they "
            "are paired observations rather than separate measures"
        )
    else:
        confidence = 0.3
        evidence = (
            f"{len(numerics)} numeric columns are present and the first two were "
            "paired arbitrarily; this is probably not an x/y relationship"
        )

    counts = tuple(len(usable[label]) for label in series)
    max_count = max(counts)
    values = tuple(
        tuple(
            float(extras["y"][index][row]) if row < counts[index] else float("nan")
            for index in range(len(series))
        )
        for row in range(max_count)
    )
    references = tuple(
        tuple(
            extras["y_references"][index][row] if row < counts[index] else ""
            for index in range(len(series))
        )
        for row in range(max_count)
    )
    return [
        Interpretation(
            kind=DataKind.XY_SAMPLES,
            table=table.name,
            categories=tuple(f"obs_{index + 1}" for index in range(max_count)),
            series=series,
            values=values,
            references=references,
            confidence=confidence,
            category_label="observation",
            value_label=y_column.name,
            category_column=group_column.name if group_column else None,
            value_columns=(x_column.name, y_column.name),
            notes=tuple(
                note
                for note in (
                    f"paired observations of {x_column.name!r} against "
                    f"{y_column.name!r}"
                    + (f", grouped by {group_column.name!r}" if group_column else ""),
                    evidence,
                )
                if note
            ),
            extras=extras,
        )
    ]


def _overlap_interpretation(
    table: Table, category_column: Column, numerics: Sequence[Column]
) -> Interpretation | None:
    """Detect per-group total/unique counts plus a shared core."""

    total_column = next(
        (column for column in numerics if _mentions(column.name, _TOTAL_HINTS)), None
    )
    unique_column = next(
        (column for column in numerics if _mentions(column.name, _UNIQUE_HINTS)), None
    )
    if total_column is None or unique_column is None or total_column is unique_column:
        return None

    core_column = next(
        (column for column in numerics if _mentions(column.name, _CORE_HINTS)), None
    )
    totals = total_column.numeric()
    uniques = unique_column.numeric()
    if not np.all(np.isfinite(totals)) or not np.all(np.isfinite(uniques)):
        return None
    if np.any(uniques > totals):
        return None

    core = float(np.nanmin(core_column.numeric())) if core_column is not None else 0.0
    categories = _label_values(category_column)
    values = tuple(
        (float(totals[row]), float(uniques[row])) for row in range(table.n_rows)
    )
    references = tuple(
        (total_column.references[row], unique_column.references[row])
        for row in range(table.n_rows)
    )
    return Interpretation(
        kind=DataKind.SET_OVERLAP,
        table=table.name,
        categories=categories,
        series=(total_column.name, unique_column.name),
        values=values,
        references=references,
        category_label=category_column.name,
        value_label=total_column.name,
        category_column=category_column.name,
        value_columns=(total_column.name, unique_column.name),
        notes=(
            f"{total_column.name!r} read as per-group totals and "
            f"{unique_column.name!r} as group-specific counts"
            + (f"; shared core from {core_column.name!r}" if core_column else ""),
        ),
        extras={"core": core, "core_column": core_column.name if core_column else None},
    )


def _long_interpretations(table: Table) -> list[Interpretation]:
    """Readings where two label columns index a single value column."""

    labels = table.label_columns()
    numerics = [column for column in table.numeric_columns() if not _is_error_column(column)]
    if len(labels) < 2 or not numerics:
        return []

    value_column = numerics[0]
    first, second = labels[0], labels[1]
    row_labels = first.labels()
    column_labels = second.labels()
    categories = _unique_in_order(row_labels)
    series = _unique_in_order(column_labels)
    if len(categories) < 2 or len(series) < 2:
        return []
    if len(categories) * len(series) > 4 * table.n_rows:
        # Too sparse to be a real cross-tabulation.
        return []

    numbers = value_column.numeric()
    grid = np.full((len(categories), len(series)), float("nan"))
    cells: list[list[str]] = [["" for _ in series] for _ in categories]
    for row in range(table.n_rows):
        if not row_labels[row] or not column_labels[row]:
            continue
        i = categories.index(row_labels[row])
        j = series.index(column_labels[row])
        grid[i, j] = numbers[row]
        cells[i][j] = value_column.references[row]

    values = tuple(tuple(float(value) for value in row) for row in grid)
    references = tuple(tuple(row) for row in cells)
    notes = (
        f"pivoted long-form data: rows from {first.name!r}, columns from "
        f"{second.name!r}, values from {value_column.name!r}",
    )
    results = [
        Interpretation(
            kind=DataKind.MATRIX,
            table=table.name,
            categories=categories,
            series=series,
            values=values,
            references=references,
            category_label=first.name,
            value_label=value_column.name,
            category_column=first.name,
            value_columns=(value_column.name,),
            ordered_categories=first.kind == ColumnKind.TEMPORAL,
            notes=notes,
            extras={"long_form": True, "series_column": second.name},
        )
    ]

    finite = grid[np.isfinite(grid)]
    if finite.size and float(finite.min()) >= 0.0:
        results.append(
            Interpretation(
                kind=DataKind.NESTED_PARTS,
                table=table.name,
                categories=categories,
                series=series,
                values=values,
                references=references,
                category_label=first.name,
                value_label=value_column.name,
                category_column=first.name,
                value_columns=(value_column.name,),
                notes=notes
                + (
                    "non-negative values, so each row is a composition over "
                    f"{second.name!r} and the column sums give an overall mix",
                ),
                extras={"long_form": True, "series_column": second.name},
            )
        )
        results.append(
            Interpretation(
                kind=DataKind.STACKED_PARTS,
                table=table.name,
                categories=categories,
                series=series,
                values=values,
                references=references,
                category_label=first.name,
                value_label=value_column.name,
                category_column=first.name,
                value_columns=(value_column.name,),
                notes=notes + ("non-negative values can stack per category",),
                extras={"long_form": True, "series_column": second.name},
            )
        )
    return results


def _distribution_interpretations(table: Table) -> list[Interpretation]:
    """Readings where one label column repeats over many observations."""

    labels = table.label_columns()
    numerics = [column for column in table.numeric_columns() if not _is_error_column(column)]
    if not numerics:
        return []

    value_column = numerics[0]
    numbers = value_column.numeric()

    if not labels:
        finite = [float(value) for value in numbers if np.isfinite(value)]
        if len(finite) < 20:
            return []
        return [
            Interpretation(
                kind=DataKind.DISTRIBUTION_SAMPLES,
                table=table.name,
                categories=tuple(f"bin_{index + 1}" for index in range(1)),
                series=(value_column.name,),
                values=((float("nan"),),),
                references=(("",),),
                value_label=value_column.name,
                value_columns=(value_column.name,),
                notes=(
                    f"{len(finite)} ungrouped observations of {value_column.name!r}",
                ),
                extras={
                    "samples": (tuple(finite),),
                    "sample_references": (
                        tuple(
                            value_column.references[row]
                            for row in range(table.n_rows)
                            if np.isfinite(numbers[row])
                        ),
                    ),
                    "value_label": value_column.name,
                },
            )
        ]

    group_column = labels[0]
    group_labels = group_column.labels()
    buckets: dict[str, list[int]] = {}
    for row, label in enumerate(group_labels):
        if not label or not np.isfinite(numbers[row]):
            continue
        buckets.setdefault(label, []).append(row)
    if not buckets or len(buckets) > 8:
        return []
    smallest = min(len(rows) for rows in buckets.values())
    if smallest < 8:
        return []

    series = tuple(buckets)
    return [
        Interpretation(
            kind=DataKind.DISTRIBUTION_SAMPLES,
            table=table.name,
            categories=tuple(f"bin_{index + 1}" for index in range(1)),
            series=series,
            values=tuple((float("nan"),) * len(series) for _ in range(1)),
            references=tuple(("",) * len(series) for _ in range(1)),
            category_label="bin",
            value_label=value_column.name,
            category_column=group_column.name,
            value_columns=(value_column.name,),
            notes=(
                f"{table.n_rows} observations of {value_column.name!r} grouped by "
                f"{group_column.name!r} into {len(series)} series "
                f"(smallest has {smallest})",
            ),
            extras={
                "samples": tuple(
                    tuple(float(numbers[row]) for row in buckets[label]) for label in series
                ),
                "sample_references": tuple(
                    tuple(value_column.references[row] for row in buckets[label])
                    for label in series
                ),
                "value_label": value_column.name,
            },
        )
    ]


def _membership_interpretations(table: Table) -> list[Interpretation]:
    """Readings where boolean columns index a count column."""

    booleans = table.columns_of(ColumnKind.BOOLEAN)
    numerics = [column for column in table.numeric_columns() if not _is_error_column(column)]
    if len(booleans) < 2 or not numerics:
        return []

    count_column = next(
        (column for column in numerics if _mentions(column.name, _TOTAL_HINTS)),
        numerics[0],
    )
    counts = count_column.numeric()
    memberships: list[tuple[bool, ...]] = []
    kept: list[int] = []
    from .ingest import parse_boolean

    for row in range(table.n_rows):
        flags = tuple(parse_boolean(column.values[row]) for column in booleans)
        if any(flag is None for flag in flags) or not np.isfinite(counts[row]):
            continue
        memberships.append(tuple(bool(flag) for flag in flags))
        kept.append(row)
    if len(memberships) < 2 or len(set(memberships)) != len(memberships):
        return []

    group_column = next(
        (
            column
            for column in table.label_columns()
            if 1 < len(column.distinct()) <= 6
        ),
        None,
    )
    groups = (
        tuple(group_column.labels()[row] for row in kept) if group_column else ()
    )
    return [
        Interpretation(
            kind=DataKind.SET_MEMBERSHIP,
            table=table.name,
            categories=tuple(
                "".join("1" if flag else "0" for flag in pattern) for pattern in memberships
            ),
            series=tuple(column.name for column in booleans),
            values=tuple((float(counts[row]),) * len(booleans) for row in kept),
            references=tuple(
                (count_column.references[row],) * len(booleans) for row in kept
            ),
            category_label="combination",
            value_label=count_column.name,
            category_column=group_column.name if group_column else None,
            value_columns=(count_column.name,),
            notes=(
                f"{len(booleans)} membership columns "
                f"({', '.join(column.name for column in booleans)}) with counts from "
                f"{count_column.name!r}",
            ),
            extras={
                "sets": tuple(column.name for column in booleans),
                "memberships": tuple(memberships),
                "counts": tuple(float(counts[row]) for row in kept),
                "count_references": tuple(count_column.references[row] for row in kept),
                "groups": groups,
                "count_label": count_column.name,
            },
        )
    ]


def _unique_in_order(labels: Sequence[str]) -> tuple[str, ...]:
    seen: list[str] = []
    for label in labels:
        if label and label not in seen:
            seen.append(label)
    return tuple(seen)


def interpretations(table: Table) -> tuple[Interpretation, ...]:
    """Every plausible reading of ``table``, most structural first."""

    collected: list[Interpretation] = []
    collected.extend(_wide_interpretations(table))
    collected.extend(_long_interpretations(table))
    collected.extend(_membership_interpretations(table))
    collected.extend(_xy_interpretations(table))
    collected.extend(_distribution_interpretations(table))

    unique: list[Interpretation] = []
    seen: set[tuple[Any, ...]] = set()
    for item in collected:
        key = (item.kind, item.categories, item.series, item.values)
        if key in seen:
            continue
        seen.add(key)
        unique.append(replace(item, source=str(table.source)))
    return tuple(unique)


def profile_table(table: Table) -> DataProfile:
    """Summarise ``table`` and enumerate how it can be plotted."""

    readings = interpretations(table)
    numerics = [column for column in table.numeric_columns() if not _is_error_column(column)]
    errors = [column for column in table.numeric_columns() if _is_error_column(column)]

    pool = np.concatenate([column.numeric() for column in numerics]) if numerics else np.array([])
    finite = pool[np.isfinite(pool)] if pool.size else pool
    value_min = float(finite.min()) if finite.size else float("nan")
    value_max = float(finite.max()) if finite.size else float("nan")
    non_negative = bool(finite.size and value_min >= 0.0)

    # Dynamic range only means something for strictly positive data. Values that
    # straddle zero have small magnitudes by construction, and reporting a huge
    # ratio for them would wrongly warn against every length-based encoding.
    if non_negative and finite.size:
        positive = finite[finite > 0.0]
        dynamic_range = (
            float(positive.max() / positive.min()) if positive.size else 1.0
        )
    else:
        dynamic_range = 1.0

    sums_to_100 = False
    for reading in readings:
        if reading.kind is not DataKind.PARTS_OF_WHOLE:
            continue
        column_total = float(np.nansum(reading.matrix()))
        if abs(column_total - 100.0) <= _PERCENT_TOTAL_TOLERANCE:
            sums_to_100 = True
    if not sums_to_100 and len(numerics) == 1 and non_negative:
        total = float(np.nansum(numerics[0].numeric()))
        sums_to_100 = abs(total - 100.0) <= _PERCENT_TOTAL_TOLERANCE

    labels = table.label_columns()
    naming = [table.name, *(column.name for column in numerics)]
    if labels:
        naming.append(labels[0].name)

    # Sitting inside [0, 1] is not enough to call something a bounded metric: an
    # MSE column does that too. The name has to say so, in a column header, the
    # sheet name, or the label column's header.
    named_bounded = any(_mentions(text, _BOUNDED_HINTS) for text in naming)
    within_unit = bool(finite.size and value_min >= 0.0 and value_max <= 1.0001)
    within_percent = bool(finite.size and value_min >= 0.0 and value_max <= 100.0001)
    bounded = named_bounded and (within_unit or within_percent)

    # The metric name may live in a column header, the sheet name, or the header
    # of the label column ("Dataset" against a sheet called "MSE").
    lower_is_better = any(_mentions(text, _LOWER_BETTER_HINTS) for text in naming)
    has_uncertainty = bool(errors) or any(
        column.errors() is not None for column in numerics
    )
    emphasised = sum(len(column.emphasised_rows()) for column in numerics)

    observations_per_label = (
        table.n_rows / max(len(labels[0].distinct()), 1) if labels else float(table.n_rows)
    )

    notes: list[str] = []
    if not numerics:
        notes.append("no numeric column found; nothing can be plotted from this table")
    if not readings:
        notes.append(
            "no reading matched; check that the table has a label column and at "
            "least one numeric column"
        )
    if sums_to_100:
        notes.append("values sum to 100, so they are already percentages")

    return DataProfile(
        table=table.name,
        source=str(table.source),
        n_rows=table.n_rows,
        n_columns=table.n_columns,
        label_columns=tuple(column.name for column in labels),
        numeric_columns=tuple(column.name for column in numerics),
        boolean_columns=tuple(column.name for column in table.columns_of(ColumnKind.BOOLEAN)),
        temporal_columns=tuple(
            column.name for column in table.columns_of(ColumnKind.TEMPORAL)
        ),
        error_columns=tuple(column.name for column in errors),
        interpretations=readings,
        value_min=value_min,
        value_max=value_max,
        all_non_negative=non_negative,
        sums_to_100=sums_to_100,
        bounded_unit_scale=bounded,
        dynamic_range=dynamic_range,
        lower_is_better=lower_is_better,
        has_uncertainty=has_uncertainty,
        emphasised_cells=emphasised,
        observations_per_label=float(observations_per_label),
        notes=tuple(notes),
    )
