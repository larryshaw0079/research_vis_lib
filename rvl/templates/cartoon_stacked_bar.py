"""Cartoon stacked capsule bars: rounded segments stacking into a per-bar total.

Every bar is a stack of capsules separated by white gaps, drawn inside a dashed
sketch frame with a soft drop shadow, a caption under each bar that an optional
note can override, and an optional group heading under each run of bars that
share a group.

``DEFAULT_DATA`` holds the nine-bar policy-mix figure digitised from a
Xiaohongshu carousel that repeats the same layout 16 times, changing only the
colour palette; the source publishes the chart but not the table, so the segment
heights are integers that sum to the labelled totals (48, 54, 60, 64, 78, 86,
90, 114, 130).

Palette convention: the curated colours are listed in stack order and the final
entry is the dash/outline ink rather than a fill.  Segment fills always come from
``palette.take(len(data.segments))``; the dash ink is read separately as the last
palette colour and outlines any segment that contributes nothing to any bar, the
way the reference draws its "Trade-offs Missing" placeholder.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Final, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch
from numpy.typing import NDArray

from ..contract import DataKind, Extent, Feature, Geometry, TemplateSpec
from ..palettes import Palette
from ..render import nice_step, resolve_limits, run_cli

SPEC: Final[TemplateSpec] = TemplateSpec(
    template_id="cartoon-stacked-bar",
    title="Cartoon stacked capsule bars",
    summary=(
        "Rounded capsule bars whose coloured segments stack into a per-bar total, "
        "with a sketch-styled outline, dashed segment separators and per-bar notes."
    ),
    kinds=(DataKind.STACKED_PARTS,),
    geometry=Geometry.CARTESIAN,
    categories=Extent(2, 14),
    series=Extent(2, 10),
    builder="CartoonStackedBarData.from_matrix",
    data_contract=(
        "A non-negative contribution for every (bar, segment) pair; segments stack "
        "into the bar total. Bars may carry an optional group label and note."
    ),
    good_for=(
        "showing how a total decomposes into named contributions",
        "comparing composition across a small number of scenarios",
        "informal or presentation figures where a hand-drawn look is wanted",
    ),
    avoid_when=(
        "segment values can be negative, which cannot stack",
        "readers need to compare a single segment precisely across bars",
        "more than about 14 bars or 10 segments",
    ),
    requires=(Feature.NON_NEGATIVE,),
    argument_names=(("categories", "bars"), ("series", "segments")),
    affinities=(
        (Feature.NON_NEGATIVE, 8.0),
        (Feature.SUMS_TO_100, 4.0),
        (Feature.MANY_CATEGORIES, -7.0),
        (Feature.WIDE_DYNAMIC_RANGE, -9.0),
    ),
    default_dpi=200,
    reference="Digitised from a Xiaohongshu carousel.",
)


# Colours sampled from the 3x3 legend pills in carousel frames 1-16, re-ordered
# so a positional lookup walks up the stack.  Named qualitative sets use
# canonical ColorBrewer / Coolors hex values; the rest are median fills snapped
# to nearby saturated hex codes.  The last entry of each palette is dash ink.
PALETTES: Final[tuple[Palette, ...]] = (
    Palette(
        "sky-cream-mint",
        ("#5CA2FA", "#FDF6C2", "#B8E9C9", "#90C3A5", "#FE9F9B", "#CACAFC", "#ABE5E9", "#3FB882"),
    ),
    Palette(
        "pastel1",
        ("#FBB4AE", "#FED9A6", "#E5D8BD", "#B3CDE3", "#CCEBC5", "#FFFFCC", "#DECBE4", "#F1B6D0"),
    ),
    Palette(
        "mint-peach-pastel",
        ("#B3E3CD", "#E5F4C9", "#F2E2CB", "#EAC9AE", "#CAD5E7", "#FFF2AE", "#F5CAE5", "#E8B48A"),
    ),
    Palette(
        "brbg",
        ("#8C510A", "#C7EAE5", "#35978F", "#BF812D", "#DFC27D", "#80CDC1", "#F6E8C3", "#01665E"),
    ),
    Palette(
        "rdbu",
        ("#B2182B", "#D1E5F0", "#4393C3", "#D6604D", "#F4A582", "#92C5DE", "#FDDBC7", "#2166AC"),
    ),
    Palette(
        "piyg",
        ("#C51B7D", "#A1D99B", "#F1B6DA", "#E9A3C9", "#FDE0EF", "#4D9221", "#E6F5D0", "#4D9221"),
    ),
    Palette(
        "navy-magenta-orange",
        ("#003F5C", "#D45087", "#FF7C43", "#2F4B7C", "#665191", "#F95D6A", "#A05195", "#FFA600"),
    ),
    Palette(
        "navy-cream-coral",
        ("#01429E", "#FFF6C8", "#F4777D", "#4A71B1", "#74A2C6", "#FFBCAF", "#A5D5D9", "#C24566"),
    ),
    Palette(
        "ink-cream-aqua",
        ("#22223A", "#F2E9E4", "#0DA4AF", "#4B4D68", "#9A8C9B", "#B5E1FA", "#CAADA7", "#E6DC9F"),
    ),
    Palette(
        "taupe-olive",
        ("#CB997E", "#A5A58D", "#3F423B", "#DCBEAA", "#FFE7D6", "#6B705C", "#B8B7A3", "#D0AB8F"),
    ),
    Palette(
        "pacific-sunset",
        ("#001219", "#E9D8A6", "#CA6702", "#005F73", "#0A9396", "#EE9B00", "#94D2BD", "#BB3E03"),
    ),
    Palette(
        "navy-sky-amber",
        ("#8ECAE6", "#FB8500", "#023047", "#219EBC", "#3D405B", "#E76F51", "#FFB703", "#8AB5A0"),
    ),
    Palette(
        "candy-pastel",
        ("#CDB4DB", "#A2D2FF", "#FDFFB6", "#E7C2D4", "#FFAFCC", "#FFD7A6", "#BDE0FE", "#B8E8B0"),
    ),
    Palette(
        "lime-teal",
        ("#D9ED92", "#52B69A", "#168AAD", "#B5E48C", "#99D98C", "#34A0A4", "#76C893", "#1A759F"),
    ),
    Palette(
        "mediterranean",
        ("#264653", "#E76F51", "#457B9D", "#2A9D8F", "#E9C46A", "#1D3557", "#F4A261", "#80CBC4"),
    ),
    Palette(
        "olive-rust-rose",
        ("#5F6C37", "#BC6C25", "#E3989B", "#283618", "#FFF6D6", "#B5838E", "#DCA15D", "#E8A598"),
    ),
)


@dataclass(frozen=True, slots=True)
class CartoonStackedBarData:
    """A ``bars x segments`` matrix of non-negative contributions.

    ``values`` is indexed ``[bar][segment]``; segments stack bottom-to-top in the
    order they are declared and sum to the bar total.
    """

    bars: tuple[str, ...]
    segments: tuple[str, ...]
    values: tuple[tuple[float, ...], ...]
    bar_groups: tuple[str | None, ...] | None = None
    notes: tuple[str | None, ...] | None = None
    value_label: str = "Value"
    total_format: str = "{:.0f}"
    show_totals: bool = True
    ghost_bars: tuple[str, ...] | None = None
    """Bars that lost an expected component, marked with a hollow dashed capsule."""

    @classmethod
    def from_matrix(
        cls,
        *,
        bars: Sequence[str],
        segments: Sequence[str],
        values: Sequence[Sequence[float]],
        bar_groups: Sequence[str | None] | None = None,
        notes: Sequence[str | None] | None = None,
        value_label: str = "Value",
        total_format: str = "{:.0f}",
        show_totals: bool = True,
        ghost_bars: Sequence[str] | None = None,
    ) -> "CartoonStackedBarData":
        """Build from a ``[bar][segment]`` matrix of non-negative contributions.

        ``bar_groups`` adds a heading under each run of bars that share a value;
        ``notes`` overrides the caption drawn under a bar, so several bars can
        share a display name while ``bars`` stays a set of unique keys.
        ``ghost_bars`` names bars that lost an expected component, drawn as a
        hollow dashed capsule above the stack.
        """

        built = cls(
            bars=tuple(str(name) for name in bars),
            segments=tuple(str(name) for name in segments),
            values=tuple(
                tuple(float("nan") if value is None else float(value) for value in row)
                for row in values
            ),
            bar_groups=_optional_labels(bar_groups),
            notes=_optional_labels(notes),
            value_label=value_label,
            total_format=total_format,
            show_totals=bool(show_totals),
            ghost_bars=None if ghost_bars is None else tuple(str(name) for name in ghost_bars),
        )
        built.validate()
        return built

    def validate(self) -> None:
        n_bars = len(self.bars)
        n_segments = len(self.segments)
        if n_bars < 1:
            raise ValueError("need at least one bar")
        if n_segments < 1:
            raise ValueError("need at least one segment")
        if len(set(self.bars)) != n_bars:
            raise ValueError("bar labels must be unique")
        if len(set(self.segments)) != n_segments:
            raise ValueError("segment labels must be unique")
        if len(self.values) != n_bars:
            raise ValueError(
                f"values has {len(self.values)} rows but there are {n_bars} bars"
            )
        for name, row in zip(self.bars, self.values, strict=True):
            if len(row) != n_segments:
                raise ValueError(
                    f"values[{name!r}] has {len(row)} entries but there are "
                    f"{n_segments} segments"
                )
            for label, value in zip(self.segments, row, strict=True):
                if not np.isfinite(value):
                    raise ValueError(f"values[{name!r}][{label!r}] must be finite")
                if value < 0.0:
                    raise ValueError(
                        f"values[{name!r}][{label!r}] is negative; stacked segments "
                        "must be non-negative"
                    )
        for field_name, column in (
            ("bar_groups", self.bar_groups),
            ("notes", self.notes),
        ):
            if column is not None and len(column) != n_bars:
                raise ValueError(
                    f"{field_name} has {len(column)} entries but there are "
                    f"{n_bars} bars"
                )
        for name in self.ghost_bars or ():
            if name not in self.bars:
                raise ValueError(f"ghost_bars entry {name!r} is not one of the bars")
        if max(self.totals()) <= 0.0:
            raise ValueError("values must contain at least one positive contribution")
        try:
            self.total_format.format(1.0)
        except (IndexError, KeyError, ValueError) as exc:
            raise ValueError(
                f"total_format {self.total_format!r} is not usable"
            ) from exc

    def matrix(self) -> NDArray[np.float64]:
        """Contributions as a ``(bars, segments)`` array."""

        return np.asarray(self.values, dtype=float)

    def totals(self) -> tuple[float, ...]:
        """Per-bar sum of its segments."""

        return tuple(float(sum(row)) for row in self.values)

    def stack(self, index: int) -> tuple[tuple[int, float], ...]:
        """``(segment index, value)`` pairs a bar actually draws, bottom first."""

        return tuple(
            (position, float(value))
            for position, value in enumerate(self.values[index])
            if value > 0.0
        )

    def unused_segments(self) -> frozenset[int]:
        """Segments that contribute nothing anywhere, drawn as hollow outlines."""

        column_sums = self.matrix().sum(axis=0)
        return frozenset(
            index for index, total in enumerate(column_sums) if total <= 0.0
        )

    def group_labels(self) -> tuple[str | None, ...]:
        """One group heading per bar, all ``None`` when no groups were given."""

        if self.bar_groups is None:
            return (None,) * len(self.bars)
        return self.bar_groups

    def ghost_indices(self) -> frozenset[int]:
        """Bars that should carry a hollow dashed capsule above their stack."""

        if not self.ghost_bars:
            return frozenset()
        return frozenset(self.bars.index(name) for name in self.ghost_bars)

    def captions(self) -> tuple[str, ...]:
        """Text drawn under each bar: its note when given, otherwise its label."""

        if self.notes is None:
            return self.bars
        return tuple(
            bar if note is None else note
            for bar, note in zip(self.bars, self.notes, strict=True)
        )


def _optional_labels(
    labels: Sequence[str | None] | None,
) -> tuple[str | None, ...] | None:
    if labels is None:
        return None
    return tuple(None if label is None else str(label) for label in labels)


_REFERENCE_BARS: Final[tuple[str, ...]] = (
    "Policy A",
    "Policy B",
    "Policy C",
    "Dual Trade-offs",
    "Dual Unrelated",
    "Dual Synergies",
    "Multi Trade-offs",
    "Multi Unrelated",
    "Multi Synergies",
)

# Two bars per group repeat a caption, so the unique keys above carry the note
# that is actually printed under the bar.
_REFERENCE_NOTES: Final[tuple[str | None, ...]] = (
    None,
    None,
    None,
    "Trade-offs",
    "Unrelated",
    "Synergies",
    "Trade-offs",
    "Unrelated",
    "Synergies",
)

_REFERENCE_GROUPS: Final[tuple[str, ...]] = (
    *("One-Policy (A/B/C)",) * 3,
    *("Dual-Policies (A \u222a B)",) * 3,
    *("Multi-Policies (A \u222a B \u222a C)",) * 3,
)

# Exclusive policy bases plus the four Venn overlaps.  "Trade-offs Missing" is
# the reference's placeholder for overlaps a trade-off destroys: it contributes
# to no bar and only shows up in the legend, as a hollow dashed pill.
_REFERENCE_SEGMENTS: Final[tuple[str, ...]] = (
    "Policy A Base",
    "Policy B Base",
    "Policy C Base",
    "Overlap (Cyan)",
    "Overlap (Purple)",
    "Overlap (Yellow)",
    "Overlap (Green)",
    "Trade-offs Missing",
    "Synergy Bonus",
)

# Integer segment heights summing to the labelled totals.
_REFERENCE_VALUES: Final[tuple[tuple[float, ...], ...]] = (
    (24, 0, 0, 8, 8, 0, 8, 0, 0),
    (0, 30, 0, 8, 0, 8, 8, 0, 0),
    (0, 0, 36, 0, 8, 8, 8, 0, 0),
    (20, 26, 0, 0, 6, 6, 6, 0, 0),
    (24, 30, 0, 0, 8, 8, 8, 0, 0),
    (24, 30, 0, 8, 8, 8, 8, 0, 0),
    (24, 30, 36, 0, 0, 0, 0, 0, 0),
    (24, 30, 36, 0, 8, 8, 8, 0, 0),
    (24, 30, 36, 8, 8, 8, 8, 0, 8),
)

DEFAULT_DATA: Final[CartoonStackedBarData] = CartoonStackedBarData.from_matrix(
    bars=_REFERENCE_BARS,
    segments=_REFERENCE_SEGMENTS,
    values=_REFERENCE_VALUES,
    bar_groups=_REFERENCE_GROUPS,
    notes=_REFERENCE_NOTES,
    value_label="Effect Score",
    total_format="{:.0f}",
    ghost_bars=("Multi Trade-offs",),
)


@dataclass(frozen=True, slots=True)
class ChartStyle:
    """Layout tuned to the 2555 x 1440 reference image.

    Every position is a fraction of the plotted window so the layout survives a
    change of units: the defaults are the reference's data-unit values divided by
    its 12.53-wide by 186-tall window.  Bar slots share ``bar_span`` up to
    ``max_slot_pitch``, and the value axis is derived from the tallest stack --
    ``y_limits`` only pins it while the reference data keeps filling it.

    ``title`` and ``title_note`` are the reference's own captions; the data model
    carries no title, so generated code should override them.
    """

    figure_size: tuple[float, float] = (12.774, 7.2)
    axes_bounds: tuple[float, float, float, float] = (0.074, 0.138, 0.912, 0.792)
    x_limits: tuple[float, float] | None = (-0.85, 11.68)
    y_limits: tuple[float, float] | None = (-18.0, 168.0)
    y_tick_step: float | None = 30.0
    y_margin: float = 18.0 / 150.0
    bar_span: tuple[float, float] = (0.85 / 12.53, (0.85 + 10.60) / 12.53)
    max_slot_pitch: float = 2.30 / 12.53
    """Widest a bar slot may grow to; a handful of bars centre instead of sprawling."""

    group_gap: float = 0.70 / 1.15
    bar_width: float = 0.62 / 1.15
    segment_gap: float = 1.25 / 130.0
    total_offset: float = 4.6 / 186.0
    caption_y: float = (18.0 - 7.2) / 186.0
    group_y: float = (18.0 - 14.8) / 186.0
    title: str = "Samples"
    title_note: str = "(Synergy and trade-offs only list the most extreme cases)"
    title_position: tuple[float, float] = ((0.85 - 0.35) / 12.53, (18.0 + 158.5) / 186.0)
    note_x: float = (0.85 + 1.22) / 12.53
    legend_anchor: tuple[float, float] = ((0.85 - 0.38) / 12.53, (18.0 + 149.5) / 186.0)
    """Top-left of the legend box, so extra rows grow down towards the bars."""

    legend_columns: int = 3
    legend_column_pitch: float = 1.72 / 12.53
    legend_pill_size: tuple[float, float] = (0.40 / 12.53, 7.0 / 186.0)
    legend_row_gap: float = 5.35 / 186.0
    legend_pad: tuple[float, float, float, float] = (
        0.16 / 12.53,
        0.03 / 12.53,
        3.8 / 186.0,
        4.5 / 186.0,
    )
    """Legend box padding as (left, right, top, bottom) window fractions."""

    legend_text_gap: float = 0.10 / 12.53
    legend_rounding: float = 0.16 / 12.53
    frame_inset: tuple[float, float, float, float] = (
        0.08 / 12.53,
        0.08 / 12.53,
        4.0 / 186.0,
        11.5 / 186.0,
    )
    """Sketch frame inset as (left, right, top, bottom) window fractions."""

    frame_rounding: float = 0.18 / 12.53
    shadow_colors: tuple[str, ...] = ("#E4E4E4", "#D8D8D8")
    shadow_drop: tuple[float, ...] = (1.05 / 186.0, 0.70 / 186.0)
    shadow_shift: float = 0.028 / 12.53
    title_font_size: float = 16.5
    note_font_size: float = 13.0
    label_font_size: float = 12.5
    tick_font_size: float = 11.5
    legend_font_size: float = 10.8
    total_font_size: float = 13.5
    group_font_size: float = 13.0
    frame_color: str = "#B0B0B0"
    grid_color: str = "#C8C8C8"
    missing_fill: str = "#FFFFFF"
    ghost_height: float = 24.0 / 130.0
    """Height of a ghost capsule, as a fraction of the tallest bar total."""


    def validate(self, *, categories: int, series: int) -> None:
        if min(self.figure_size) <= 0:
            raise ValueError("figure_size must be positive")
        for name, limits in (("x_limits", self.x_limits), ("y_limits", self.y_limits)):
            if limits is not None and limits[1] <= limits[0]:
                raise ValueError(f"{name} must be increasing")
        if not 0.0 < self.bar_width <= 1.0:
            raise ValueError("bar_width must be a fraction of the slot pitch")
        if self.group_gap < 0.0 or self.segment_gap < 0.0:
            raise ValueError("group_gap and segment_gap must be non-negative")
        if not self.bar_span[0] < self.bar_span[1]:
            raise ValueError("bar_span must be increasing")
        if self.legend_columns < 1:
            raise ValueError("legend_columns must be positive")
        if len(self.shadow_colors) != len(self.shadow_drop):
            raise ValueError("shadow_colors and shadow_drop must be the same length")
        if categories < 1:
            raise ValueError("need at least one bar")
        if series < 1:
            raise ValueError("need at least one segment")


DEFAULT_STYLE: Final[ChartStyle] = ChartStyle()


@dataclass(frozen=True, slots=True)
class _Window:
    """The plotted window, turning layout fractions into data coordinates."""

    x_limits: tuple[float, float]
    y_limits: tuple[float, float]

    @property
    def x_span(self) -> float:
        return self.x_limits[1] - self.x_limits[0]

    @property
    def y_span(self) -> float:
        return self.y_limits[1] - self.y_limits[0]

    def x(self, fraction: float) -> float:
        return self.x_limits[0] + fraction * self.x_span

    def y(self, fraction: float) -> float:
        return self.y_limits[0] + fraction * self.y_span

    def dx(self, fraction: float) -> float:
        return fraction * self.x_span

    def dy(self, fraction: float) -> float:
        return fraction * self.y_span


def fitted_limits(
    pinned: tuple[float, float] | None,
    low: float,
    high: float,
    *,
    fill: float = 0.35,
    **padding: Any,
) -> tuple[float, float]:
    """Resolve axis limits, dropping a pin the data would barely fill.

    :func:`rvl.render.resolve_limits` keeps a pinned range whenever the data fits
    inside it, which is what preserves the reference layout.  A dataset in other
    units can fit inside the reference range and still occupy a sliver of it, so
    a pin the data fills less than ``fill`` of is dropped.
    """

    if pinned is not None and (high - low) < fill * (pinned[1] - pinned[0]):
        pinned = None
    return resolve_limits(pinned, low, high, **padding)


def axis_step(pinned: float | None, span: float) -> float:
    """Honour a pinned tick step while it still cuts the axis into a few ticks."""

    if pinned is not None and pinned > 0.0 and 2.0 <= span / pinned <= 12.0:
        return pinned
    return nice_step(span)


def segment_gap(data: CartoonStackedBarData, style: ChartStyle) -> float:
    """White gap between two capsules, in value units."""

    return style.segment_gap * max(data.totals())


def stack_tops(data: CartoonStackedBarData, style: ChartStyle) -> tuple[float, ...]:
    """Top of every drawn stack, gaps between capsules included."""

    gap = segment_gap(data, style)
    return tuple(
        total + gap * max(len(data.stack(index)) - 1, 0)
        for index, total in enumerate(data.totals())
    )


def _legend_grid(data: CartoonStackedBarData, style: ChartStyle) -> tuple[int, int]:
    """Column and row counts of the legend grid."""

    columns = min(style.legend_columns, len(data.segments))
    rows = -(-len(data.segments) // columns)
    return columns, rows


def _legend_box(data: CartoonStackedBarData, style: ChartStyle) -> tuple[float, float]:
    """Legend box size as window fractions."""

    columns, rows = _legend_grid(data, style)
    pill_height = style.legend_pill_size[1]
    left, right, top, bottom = style.legend_pad
    return (
        left + columns * style.legend_column_pitch + right,
        bottom + rows * pill_height + (rows - 1) * style.legend_row_gap + top,
    )


def legend_clearance(data: CartoonStackedBarData, style: ChartStyle) -> float:
    """Fraction of the top gridline the legend's underside sits at."""

    bottom = style.legend_anchor[1] - _legend_box(data, style)[1]
    return max(bottom * (1.0 + 2.0 * style.y_margin) - style.y_margin, 0.05)


def value_window(
    data: CartoonStackedBarData, style: ChartStyle, *, headroom: float = 1.0
) -> tuple[tuple[float, float], float]:
    """Return ``(y limits, tick step)`` for the tallest stack.

    The reference keeps a margin of ``y_margin`` of the top gridline below zero,
    where the captions sit, and the same margin above it for the total labels.
    ``headroom`` scales the value the axis has to cover, which is how a legend
    that cannot stand beside the bars buys itself a band above them.
    """

    top = max(stack_tops(data, style)) * headroom
    step = axis_step(style.y_tick_step, top)
    gridline = np.ceil(top / step) * step
    derived = (-style.y_margin * gridline, (1.0 + style.y_margin) * gridline)
    limits = fitted_limits(
        style.y_limits, *derived, pad=0.0, include_zero=False, snap=False
    )
    return limits, step


def bar_layout(
    data: CartoonStackedBarData, style: ChartStyle, window: _Window
) -> tuple[NDArray[np.float64], float]:
    """Return ``(bar centres, slot pitch)`` for the plotted span.

    A change of group between neighbouring bars widens the gap by ``group_gap``
    of a slot, so the reference's three groups of three keep their spacing while
    any other bar count still fills the frame.  A slot never grows past
    ``max_slot_pitch``: a handful of bars centre in the frame rather than
    stretching into blocks.
    """

    groups = data.group_labels()
    slots = [0.0]
    for previous, current in zip(groups, groups[1:]):
        slots.append(slots[-1] + (1.0 if current == previous else 1.0 + style.group_gap))
    first = window.x(style.bar_span[0])
    last = window.x(style.bar_span[1])
    total = slots[-1]
    if total <= 0.0:
        pitch = min(last - first, window.dx(style.max_slot_pitch))
        return np.array([0.5 * (first + last)]), pitch
    pitch = (last - first) / total
    ceiling = window.dx(style.max_slot_pitch)
    if pitch <= ceiling:
        return first + pitch * np.asarray(slots, dtype=float), pitch
    start = 0.5 * (first + last) - 0.5 * ceiling * total
    return start + ceiling * np.asarray(slots, dtype=float), ceiling


def _mutation_aspect(ax: Axes) -> float:
    bbox = ax.get_window_extent()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    if bbox.height <= 0:
        return 1.0
    return (bbox.width / (xlim[1] - xlim[0])) / (bbox.height / (ylim[1] - ylim[0]))


def _pill(
    ax: Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    facecolor: str,
    edgecolor: str,
    linestyle: str | tuple[int, tuple[int, ...]] = "solid",
    linewidth: float = 0.0,
    zorder: float = 3,
    shadow: bool = False,
    style: ChartStyle,
    window: _Window,
) -> None:
    aspect = _mutation_aspect(ax)
    rounding = min(0.48 * width, 0.48 * height / max(aspect, 1e-6))
    boxstyle = f"round,pad=0,rounding_size={rounding}"
    if shadow:
        for drop, color in zip(style.shadow_drop, style.shadow_colors, strict=True):
            ax.add_patch(
                FancyBboxPatch(
                    (x + window.dx(style.shadow_shift), y - window.dy(drop)),
                    width,
                    height,
                    boxstyle=boxstyle,
                    mutation_aspect=aspect,
                    facecolor=color,
                    edgecolor="none",
                    linewidth=0,
                    zorder=zorder - 0.4,
                )
            )
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle=boxstyle,
            mutation_aspect=aspect,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linestyle=linestyle,
            linewidth=linewidth,
            zorder=zorder,
        )
    )


def _style_axis(
    ax: Axes,
    data: CartoonStackedBarData,
    style: ChartStyle,
    window: _Window,
    step: float,
) -> None:
    ax.set_xlim(*window.x_limits)
    ax.set_ylim(*window.y_limits)
    top = np.floor(window.y_limits[1] / step) * step
    ax.set_yticks(np.arange(0.0, top + 0.5 * step, step))
    ax.set_xticks([])
    ax.tick_params(
        axis="y", length=0, labelsize=style.tick_font_size, colors="#333333", pad=4
    )
    ax.set_ylabel(
        data.value_label,
        fontsize=style.label_font_size,
        fontweight="bold",
        labelpad=10,
        color="#222222",
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.yaxis.grid(
        True,
        linestyle=(0, (3.2, 3.4)),
        color=style.grid_color,
        linewidth=0.95,
        zorder=0,
    )
    ax.set_axisbelow(True)
    ax.set_facecolor("white")
    for label in ax.get_yticklabels():
        label.set_color("#333333")


def _draw_frame(ax: Axes, style: ChartStyle, window: _Window) -> None:
    left, right, top, bottom = style.frame_inset
    ax.add_patch(
        FancyBboxPatch(
            (window.x(left), window.y(bottom)),
            window.dx(1.0 - left - right),
            window.dy(1.0 - top - bottom),
            boxstyle=f"round,pad=0,rounding_size={window.dx(style.frame_rounding)}",
            mutation_aspect=_mutation_aspect(ax),
            facecolor="none",
            edgecolor=style.frame_color,
            linestyle=(0, (4.5, 3.6)),
            linewidth=1.15,
            zorder=0.5,
            clip_on=False,
        )
    )


def _draw_title(ax: Axes, style: ChartStyle, window: _Window) -> None:
    x, y = style.title_position
    ax.text(
        window.x(x),
        window.y(y),
        style.title,
        ha="left",
        va="center",
        fontsize=style.title_font_size,
        fontweight="bold",
        color="#111111",
        zorder=6,
        clip_on=False,
    )
    if not style.title_note:
        return
    ax.text(
        window.x(style.note_x),
        window.y(y),
        style.title_note,
        ha="left",
        va="center",
        fontsize=style.note_font_size,
        fontweight="normal",
        color="#222222",
        zorder=6,
        clip_on=False,
    )


def _legend_origin(
    data: CartoonStackedBarData,
    style: ChartStyle,
    window: _Window,
    centres: NDArray[np.float64],
    pitch: float,
) -> tuple[float, float] | None:
    """Lower-left of the legend box, or ``None`` when no side is clear.

    The reference keeps the box top-left, above its three shortest bars; when a
    dataset puts tall bars there the box mirrors to the other side instead of
    covering them, and when both sides are covered the caller makes room above
    the bars instead.
    """

    anchor_x, anchor_y = style.legend_anchor
    width, height = _legend_box(data, style)
    underside = window.y(anchor_y - height)
    tops = stack_tops(data, style)
    half = 0.5 * style.bar_width * pitch

    for candidate in (anchor_x, 1.0 - anchor_x - width):
        low = window.x(candidate) - half
        high = window.x(candidate + width) + half
        blocked = any(
            top > underside
            for centre, top in zip(centres, tops, strict=True)
            if low <= centre <= high
        )
        if not blocked:
            return candidate, anchor_y
    return None


def _draw_legend(
    ax: Axes,
    data: CartoonStackedBarData,
    palette: Palette,
    style: ChartStyle,
    window: _Window,
    origin: tuple[float, float],
) -> None:
    columns, _ = _legend_grid(data, style)
    width, height = _legend_box(data, style)
    left_pad, _, top_pad, _ = style.legend_pad
    pill_width, pill_height = style.legend_pill_size
    anchor_x, anchor_y = origin

    ax.add_patch(
        FancyBboxPatch(
            (window.x(anchor_x), window.y(anchor_y - height)),
            window.dx(width),
            window.dy(height),
            boxstyle=f"round,pad=0,rounding_size={window.dx(style.legend_rounding)}",
            mutation_aspect=_mutation_aspect(ax),
            facecolor="white",
            edgecolor="#9A9A9A",
            linewidth=1.05,
            zorder=5,
        )
    )

    colors = palette.take(len(data.segments))
    dash_ink = palette.color(len(palette) - 1)
    unused = data.unused_segments()
    for index, name in enumerate(data.segments):
        row, column = divmod(index, columns)
        x = anchor_x + left_pad + column * style.legend_column_pitch
        y = (
            anchor_y
            - top_pad
            - pill_height
            - row * (pill_height + style.legend_row_gap)
        )
        if index in unused:
            _pill(
                ax,
                window.x(x),
                window.y(y),
                window.dx(pill_width),
                window.dy(pill_height),
                facecolor=style.missing_fill,
                edgecolor=dash_ink,
                linestyle=(0, (2.6, 1.8)),
                linewidth=1.55,
                zorder=6,
                style=style,
                window=window,
            )
        else:
            _pill(
                ax,
                window.x(x),
                window.y(y),
                window.dx(pill_width),
                window.dy(pill_height),
                facecolor=colors[index],
                edgecolor="none",
                zorder=6,
                style=style,
                window=window,
            )
        ax.text(
            window.x(x + pill_width + style.legend_text_gap),
            window.y(y + 0.5 * pill_height),
            name,
            ha="left",
            va="center",
            fontsize=style.legend_font_size,
            color="#222222",
            zorder=6,
        )


def _draw_bars(
    ax: Axes,
    data: CartoonStackedBarData,
    palette: Palette,
    style: ChartStyle,
    window: _Window,
    centres: NDArray[np.float64],
    pitch: float,
) -> None:
    colors = palette.take(len(data.segments))
    dash_ink = palette.color(len(palette) - 1)
    width = style.bar_width * pitch
    gap = segment_gap(data, style)
    totals = data.totals()
    ghosts = data.ghost_indices()
    ghost_height = style.ghost_height * max(totals)

    for index, centre in enumerate(centres):
        x = float(centre) - 0.5 * width
        bottom = 0.0
        stack = data.stack(index)
        for position, value in stack:
            _pill(
                ax,
                x,
                bottom,
                width,
                value,
                facecolor=colors[position],
                edgecolor="#FFFFFF",
                linewidth=0.55,
                zorder=3,
                shadow=True,
                style=style,
                window=window,
            )
            bottom += value + gap
        if index in ghosts:
            _pill(
                ax,
                x,
                bottom,
                width,
                ghost_height,
                facecolor=style.missing_fill,
                edgecolor=dash_ink,
                linestyle=(0, (2.6, 1.8)),
                linewidth=1.55,
                zorder=3,
                style=style,
                window=window,
            )
            bottom += ghost_height + gap
        if not data.show_totals:
            continue
        top = bottom - (gap if stack else 0.0)
        ax.text(
            float(centre),
            top + window.dy(style.total_offset),
            data.total_format.format(totals[index]),
            ha="center",
            va="bottom",
            fontsize=style.total_font_size,
            fontweight="bold",
            color="#111111",
            zorder=6,
        )


def _caption_font_size(
    data: CartoonStackedBarData, style: ChartStyle, window: _Window, pitch: float
) -> float:
    """Shrink the captions once the longest one outgrows its bar slot."""

    slot = 72.0 * style.figure_size[0] * style.axes_bounds[2] * pitch / window.x_span
    longest = max(len(caption) for caption in data.captions())
    return min(style.tick_font_size, max(6.0, slot / (0.6 * max(longest, 1))))


def _draw_captions(
    ax: Axes,
    data: CartoonStackedBarData,
    style: ChartStyle,
    window: _Window,
    centres: NDArray[np.float64],
    pitch: float,
) -> None:
    font_size = _caption_font_size(data, style, window, pitch)
    for centre, caption in zip(centres, data.captions(), strict=True):
        ax.text(
            float(centre),
            window.y(style.caption_y),
            caption,
            ha="center",
            va="top",
            fontsize=font_size,
            fontweight="normal",
            color="#222222",
            clip_on=False,
        )

    start = 0
    groups = data.group_labels()
    for index in range(len(groups) + 1):
        if index < len(groups) and groups[index] == groups[start]:
            continue
        label = groups[start]
        if label is not None:
            ax.text(
                0.5 * float(centres[start] + centres[index - 1]),
                window.y(style.group_y),
                label,
                ha="center",
                va="top",
                fontsize=style.group_font_size,
                fontweight="bold",
                color="#222222",
                clip_on=False,
            )
        start = index


def create_figure(
    palette: Palette = PALETTES[0],
    data: CartoonStackedBarData = DEFAULT_DATA,
    style: ChartStyle = DEFAULT_STYLE,
) -> Figure:
    """Create the cartoon stacked-bar figure without writing it to disk."""

    data.validate()
    style.validate(categories=len(data.bars), series=len(data.segments))

    # The category axis is a layout canvas rather than a value axis: bar slots
    # fill whatever window the style pins, so the reference span survives and a
    # cleared pin falls back to about a slot per bar.
    x_limits = resolve_limits(style.x_limits, 0.0, float(len(data.bars)))
    y_limits, step = value_window(data, style)
    window = _Window(x_limits=x_limits, y_limits=y_limits)
    centres, pitch = bar_layout(data, style, window)
    origin = _legend_origin(data, style, window, centres, pitch)
    if origin is None:
        # No side of the chart is clear, so lift the axis top until the legend
        # band clears the tallest stack.
        y_limits, step = value_window(
            data, style, headroom=1.0 / legend_clearance(data, style)
        )
        window = _Window(x_limits=x_limits, y_limits=y_limits)
        centres, pitch = bar_layout(data, style, window)
        origin = style.legend_anchor

    with plt.rc_context(
        {
            "font.family": [
                "Comic Sans MS",
                "Chalkboard SE",
                "Arial Rounded MT Bold",
                "DejaVu Sans",
                "sans-serif",
            ],
            "font.size": style.tick_font_size,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    ):
        figure = plt.figure(figsize=style.figure_size, facecolor="white")
        ax = figure.add_axes(style.axes_bounds)
        _style_axis(ax, data, style, window, step)
        figure.canvas.draw()
        _draw_frame(ax, style, window)
        _draw_bars(ax, data, palette, style, window, centres, pitch)
        _draw_legend(ax, data, palette, style, window, origin)
        _draw_title(ax, style, window)
        _draw_captions(ax, data, style, window, centres, pitch)

    return figure


def main(argv: Sequence[str] | None = None) -> int:
    return run_cli(sys.modules[__name__], argv)


if __name__ == "__main__":
    raise SystemExit(main())
