# Template contract

Every module in `rvl/templates/` implements this contract. The registry, the
recommender, the code generator and the fidelity verifier all depend on it, so a
template that deviates will be rejected by `tests/test_contract.py`.

## Required module surface

| Name | Type | Notes |
|------|------|-------|
| `SPEC` | `rvl.contract.TemplateSpec` | Declarative metadata. Never imports drawing code. |
| `PALETTES` | `tuple[rvl.palettes.Palette, ...]` | Curated colour cycles, order preserved from the reference carousel. |
| `<Name>Data` | frozen dataclass | Holds the plotted numbers. All shape is derived from field lengths. |
| `DEFAULT_DATA` | `<Name>Data` | The original reference example. Demo data only — never plotted for a user. |
| `ChartStyle` | frozen dataclass | Layout and typography. |
| `DEFAULT_STYLE` | `ChartStyle` | |
| `create_figure(palette, data, style) -> Figure` | function | Keyword-friendly, all three arguments defaulted. |
| `main(argv=None) -> int` | function | Two lines, delegating to `rvl.render.run_cli`. |

## Hard rules

1. **No module-level data constants drive geometry.** Counts come from
   `len(data.<field>)`, never from a module-level `GROUPS`/`MODELS`/`DATASETS`
   tuple. Those tuples may remain only as inputs to `DEFAULT_DATA`, prefixed
   `_REFERENCE_`.
2. **`validate()` checks internal consistency, not identity.** It must reject
   ragged arrays, non-finite values, negative counts and out-of-range shares. It
   must **not** reject a dataset for having different labels or a different
   number of series than the reference figure.
3. **Palettes are indexed positionally.** Use `palette.color(i)` and
   `palette.take(n)`. Never `palette.colors[i]` directly, and never look a colour
   up by label; `Palette.color` extends past the cycle length on its own.
4. **Axis titles, units and value formats are data fields**, not literals inside
   drawing helpers. A user's chart must be able to say "Accuracy (%)" where the
   reference said "Number of patients".
5. **Axis limits auto-fit unless overridden.** `ChartStyle` numeric limit fields
   become `tuple[float, float] | None`; when `None`, `create_figure` derives a
   padded range from the data. Reference fidelity is preserved because
   `DEFAULT_STYLE` may still pin the original values where the layout needs it.
   Always route a pinned limit through `rvl.render.resolve_limits`, which gives
   the pin up in two cases: the data would be clipped, or the data would fill less
   than `min_fill` (35% by default) of the pinned span and so render as a sliver.
   Do not reimplement that check locally; pass a higher `min_fill` if an axis
   needs to be filled more.
6. **Every template exposes exactly one `from_*` builder** named in
   `SPEC.builder`. It is a `@classmethod` on the data class, takes plain Python
   sequences and mappings, and is the only entry point generated code uses.
7. **Style validation takes counts as arguments.** `ChartStyle.validate()` may
   accept `*, categories: int, series: int` when the geometry depends on counts;
   it must not read module-level constants.

## Builder conventions

Builders accept plain containers so generated code stays readable and auditable:

```python
data = GroupedRingBarData.from_matrix(
    categories=("Traffic", "ETTh1"),
    series=("iTransformer", "MoFo"),
    values=((0.445, 0.424), (0.495, 0.447)),   # categories x series
    value_label="MSE",
    lower_is_better=True,
)
```

Rules for every builder:

- `values` is indexed `[category][series]` when it is a matrix.
- Missing entries are `None` or `float("nan")`; the builder must keep them as NaN
  rather than substituting zero, because zero is a real measurement.
- The builder calls `validate()` before returning.
- It raises `ValueError` with the offending label in the message.

## Fixed-shape fields

Some templates carry a categorical annotation the reference figure needs, such as
the three set rows of the UpSet matrix or the arc grouping of the radial line
chart. Model these as optional data fields with a `None` default; when `None`,
`create_figure` skips the annotation instead of failing.

## Style pinning

`DEFAULT_STYLE` reproduces the reference image, so it keeps hard-coded limits and
font sizes. When a user's data has a different shape, generated code passes
`ChartStyle()` field overrides or leaves limits at `None` for auto-fit. Keep the
reference pixel dimensions in the `ChartStyle` docstring.

## Module skeleton

```python
"""One-line summary of the chart.

Provenance of DEFAULT_DATA, including the citation or the note that values were
digitised from a reference image.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Final, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ..contract import DataKind, Extent, Geometry, TemplateSpec
from ..palettes import Palette
from ..render import run_cli

SPEC: Final[TemplateSpec] = TemplateSpec(...)

PALETTES: Final[tuple[Palette, ...]] = (Palette("orange-green", ("#FA882F", "#338227")),)


@dataclass(frozen=True, slots=True)
class ExampleData:
    categories: tuple[str, ...]
    series: tuple[str, ...]
    values: tuple[tuple[float, ...], ...]
    value_label: str = "Value"

    @classmethod
    def from_matrix(cls, *, categories, series, values, value_label="Value"):
        built = cls(
            categories=tuple(categories),
            series=tuple(series),
            values=tuple(tuple(float(v) for v in row) for row in values),
            value_label=value_label,
        )
        built.validate()
        return built

    def validate(self) -> None:
        ...


DEFAULT_DATA: Final[ExampleData] = ExampleData.from_matrix(...)


@dataclass(frozen=True, slots=True)
class ChartStyle:
    """Layout tuned to the 2601 x 2601 reference image."""

    figure_size: tuple[float, float] = (10.4, 10.4)
    value_limits: tuple[float, float] | None = None

    def validate(self, *, categories: int, series: int) -> None:
        ...


DEFAULT_STYLE: Final[ChartStyle] = ChartStyle()


def create_figure(
    palette: Palette = PALETTES[0],
    data: ExampleData = DEFAULT_DATA,
    style: ChartStyle = DEFAULT_STYLE,
) -> Figure:
    data.validate()
    style.validate(categories=len(data.categories), series=len(data.series))
    ...
    return figure


def main(argv: Sequence[str] | None = None) -> int:
    return run_cli(sys.modules[__name__], argv)


if __name__ == "__main__":
    raise SystemExit(main())
```

Removed from every template: the per-module `palette_from_selector`,
`render_palette`, `_selected_palettes`, `build_argument_parser` and the argparse
body. Those now live in `rvl/render.py` and `rvl/palettes.py`.
