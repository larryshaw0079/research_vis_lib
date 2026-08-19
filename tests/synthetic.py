"""Build synthetic readings for the contract tests.

The point of these is to prove a template can draw a dataset that is *not* its
reference figure: different labels, different counts, different units. Values are
deterministic so a failure is reproducible.
"""

from __future__ import annotations

import math
from typing import Any

from rvl.contract import DataKind, TemplateSpec
from rvl.profiling import Interpretation


def _value(row: int, column: int) -> float:
    """A smooth, strictly positive, non-degenerate value grid."""

    return round(4.0 + 3.0 * math.sin(0.7 * row + 0.4 * column) + 0.35 * column + 0.2 * row, 4)


def _category_labels(count: int, *, long: bool = False) -> tuple[str, ...]:
    if long:
        return tuple(f"Condition {index + 1} with a long name" for index in range(count))
    return tuple(f"cat{index + 1:02d}" for index in range(count))


def _series_labels(count: int) -> tuple[str, ...]:
    return tuple(f"series{index + 1}" for index in range(count))


def _dates(count: int) -> tuple[str, ...]:
    return tuple(f"2025-{(index % 12) + 1:02d}-0{(index % 9) + 1}" for index in range(count))


def _grid(categories: int, series: int) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(_value(row, column) for column in range(series))
        for row in range(categories)
    )


def _references(categories: int, series: int) -> tuple[tuple[str, ...], ...]:
    return tuple(
        tuple(f"synthetic!{chr(ord('B') + column)}{row + 2}" for column in range(series))
        for row in range(categories)
    )


def _membership_patterns(count: int) -> tuple[tuple[bool, ...], ...]:
    """``count`` distinct membership rows over as few sets as possible."""

    sets = max(2, math.ceil(math.log2(max(count, 2))))
    patterns: list[tuple[bool, ...]] = []
    for index in range(1 << sets):
        if len(patterns) == count:
            break
        patterns.append(
            tuple(bool((index >> position) & 1) for position in range(sets))
        )
    return tuple(patterns)


def synthetic_reading(
    kind: DataKind,
    categories: int,
    series: int,
    *,
    long_labels: bool = False,
    ordered: bool | None = None,
) -> Interpretation:
    """A reading of ``kind`` with the requested shape, filled with fake data."""

    values = _grid(categories, series)
    references = _references(categories, series)
    labels = (
        _dates(categories)
        if (ordered if ordered is not None else kind is DataKind.SERIES_WITH_TOTALS)
        else _category_labels(categories, long=long_labels)
    )
    extras: dict[str, Any] = {}

    if kind is DataKind.XY_SAMPLES:
        extras = {
            "x": tuple(
                tuple(1.0 + 0.5 * index for index in range(categories))
                for _ in range(series)
            ),
            "y": tuple(
                tuple(_value(index, column) for index in range(categories))
                for column in range(series)
            ),
            "x_label": "Predictor (unit)",
            "y_label": "Response (unit)",
        }
    elif kind is DataKind.DISTRIBUTION_SAMPLES:
        per_series = max(40, categories * 8)
        extras = {
            "samples": tuple(
                tuple(
                    round(
                        10.0 * column
                        + 5.0 * math.sin(0.31 * index)
                        + 0.05 * index,
                        4,
                    )
                    for index in range(per_series)
                )
                for column in range(series)
            ),
            "value_label": "Residual (unit)",
        }
    elif kind is DataKind.SET_MEMBERSHIP:
        patterns = _membership_patterns(categories)
        sets = tuple(f"set{index + 1}" for index in range(len(patterns[0])))
        groups = tuple(
            f"group{index % max(series, 1) + 1}" for index in range(len(patterns))
        )
        counts = tuple(float(12 + 7 * index) for index in range(len(patterns)))
        return Interpretation(
            kind=kind,
            table="synthetic",
            categories=tuple(
                "".join("1" if flag else "0" for flag in pattern) for pattern in patterns
            ),
            # The colour dimension of an UpSet reading is its set rows, so the
            # value grid is one column per set, matching rvl.profiling.
            series=sets,
            values=tuple((count,) * len(sets) for count in counts),
            references=tuple(
                (f"synthetic!E{index + 2}",) * len(sets)
                for index in range(len(patterns))
            ),
            source="synthetic",
            value_label="Records",
            extras={
                "sets": sets,
                "memberships": patterns,
                "counts": counts,
                "groups": groups if series > 1 else (),
                "count_label": "Records",
            },
        )
    elif kind is DataKind.SET_OVERLAP:
        totals = tuple(float(900 + 60 * index) for index in range(categories))
        uniques = tuple(float(30 + 5 * index) for index in range(categories))
        values = tuple(zip(totals, uniques, strict=True))
        references = tuple(
            (f"synthetic!B{index + 2}", f"synthetic!C{index + 2}")
            for index in range(categories)
        )
        extras = {"core": 415.0}
        return Interpretation(
            kind=kind,
            table="synthetic",
            categories=labels,
            series=("Total", "Unique"),
            values=values,
            references=references,
            source="synthetic",
            value_label="Feature count",
            extras=extras,
        )
    elif kind is DataKind.SERIES_WITH_TOTALS:
        extras = {
            "errors": tuple(
                tuple(round(0.08 * value, 4) for value in row) for row in values
            )
        }

    return Interpretation(
        kind=kind,
        table="synthetic",
        categories=labels,
        series=_series_labels(series),
        values=values,
        references=references,
        source="synthetic",
        category_label="condition",
        value_label="Measurement (unit)",
        ordered_categories=bool(
            ordered if ordered is not None else kind is DataKind.SERIES_WITH_TOTALS
        ),
        extras=extras,
    )


def probe_shapes(spec: TemplateSpec, *, limit: int = 24) -> tuple[tuple[int, int], ...]:
    """A few shapes inside a template's supported range, small end first."""

    def bounds(minimum: int, maximum: int | None, cap: int) -> tuple[int, int]:
        upper = min(maximum, cap) if maximum is not None else min(minimum + 6, cap)
        return minimum, max(minimum, upper)

    category_low, category_high = bounds(
        spec.categories.minimum, spec.categories.maximum, limit
    )
    series_low, series_high = bounds(spec.series.minimum, spec.series.maximum, 8)

    shapes = {
        (category_low, series_low),
        (category_high, series_high),
        ((category_low + category_high) // 2, (series_low + series_high) // 2),
    }
    return tuple(sorted(shapes))
