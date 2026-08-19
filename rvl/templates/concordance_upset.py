"""UpSet-style combination bars drawn above a set-membership matrix.

``DEFAULT_DATA`` reproduces Figure 4b of Zhang et al. (2026), npj Precision
Oncology, CC BY 4.0: patient-level concordance between plasma-specific,
tissue-specific and shared variant calls.  A Xiaohongshu carousel repeats that
figure 18 times changing only the three group colours, which is where
:data:`PALETTES` comes from.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Final, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FixedLocator, MultipleLocator
from numpy.typing import NDArray

from ..contract import DataKind, Extent, Feature, Geometry, TemplateSpec
from ..palettes import Palette, darken_if_pale
from ..render import nice_step, resolve_limits, run_cli

SPEC: Final[TemplateSpec] = TemplateSpec(
    template_id="concordance-upset",
    title="UpSet-style combination bars over a membership matrix",
    summary=(
        "One bar per observed combination of set memberships, above a tick/cross "
        "matrix showing which sets each combination belongs to, with bars coloured "
        "by an optional grouping and group titles above their peaks."
    ),
    kinds=(DataKind.SET_MEMBERSHIP,),
    geometry=Geometry.COMPOSITE,
    categories=Extent(2, 24),
    series=Extent(1, 6),
    builder="ConcordanceUpsetData.from_memberships",
    data_contract=(
        "For each observed combination: a boolean membership flag per set and a "
        "non-negative count. Combinations may carry a group label that drives "
        "colour and an above-bar heading."
    ),
    good_for=(
        "intersection sizes across 2-6 sets, where a Venn diagram stops scaling",
        "counts of records sharing a specific pattern of attributes",
        "showing that some combinations are absent or near-zero",
    ),
    avoid_when=(
        "there are no set memberships, only plain categories",
        "the values are proportions rather than counts",
        "more than about 24 combinations, where the matrix gets unreadable",
    ),
    requires=(Feature.NON_NEGATIVE,),
    affinities=(
        (Feature.NON_NEGATIVE, 6.0),
        (Feature.MANY_CATEGORIES, -4.0),
    ),
    default_dpi=250,
    reference="Zhang et al. (2026), npj Precision Oncology, Fig. 4b, CC BY 4.0.",
)


CHECK_MARK: Final[str] = "✓"
CROSS_MARK: Final[str] = "×"
CROSS_COLOR: Final[str] = "#85A3B8"
CHECK_COLOR: Final[str] = "#111111"

SET_LABEL_COLOR: Final[str] = "#515151"
"""Neutral ink for the matrix row labels when the data names no colours."""


# Named qualitative sets use canonical ColorBrewer / Tableau / Paul Tol hex
# values.  The remaining trios were sampled from bar interiors in frames 1-4.
PALETTES: Final[tuple[Palette, ...]] = (
    Palette("teal-coral-sand", ("#1F7598", "#DE6344", "#FBE2C4")),
    Palette("navy-crimson-ivory", ("#2B4B7C", "#DA534F", "#FFFFFF")),
    Palette("forest-crimson-gold", ("#4A7C59", "#C83E4D", "#F4D35F")),
    Palette("navy-magenta-amber", ("#003F5C", "#BC5090", "#FFA600")),
    Palette("tab10", ("#1F77B4", "#FF7F0E", "#2CA02C")),
    Palette("tab10-earth", ("#8C564B", "#E377C2", "#7F7F7F")),
    Palette("set1", ("#E41A1C", "#377EB8", "#4DAF4A")),
    Palette("set2", ("#66C2A5", "#FC8D62", "#8DA0CB")),
    Palette("dark2", ("#1B9E77", "#D95F02", "#7570B3")),
    Palette("paired", ("#A6CEE3", "#1F78B4", "#B2DF8A")),
    Palette("pastel1", ("#FBB4AE", "#B3CDE3", "#CCEBC5")),
    Palette("pastel2", ("#B3E2CD", "#FDCDAC", "#CBD5E8")),
    Palette("set3", ("#8DD3C7", "#FFFFB3", "#BEBADA")),
    Palette("set3-warm", ("#FB8072", "#80B1D3", "#FDB462")),
    Palette("set1-orchid-amber", ("#984EA3", "#FF7F00", "#FFFF33")),
    Palette("set1-umber", ("#A65628", "#F781BF", "#999999")),
    Palette("tol-muted", ("#332288", "#88CCEE", "#44AA99")),
    Palette("set2-bloom", ("#E78AC3", "#A6D854", "#FFD92F")),
)


@dataclass(frozen=True, slots=True)
class Combination:
    """One UpSet column: which sets it belongs to, how many records, which group."""

    membership: tuple[bool, ...]
    count: float
    group: str | None = None


def _first_appearance(combinations: Sequence[Combination]) -> tuple[str, ...]:
    """Group labels in the order they first appear, ignoring ungrouped columns."""

    order: list[str] = []
    for item in combinations:
        if item.group is not None and item.group not in order:
            order.append(item.group)
    return tuple(order)


@dataclass(frozen=True, slots=True)
class ConcordanceUpsetData:
    """Observed set combinations, their counts and their optional grouping.

    ``sets`` are matrix rows from top to bottom, ``combinations`` are bar columns
    from left to right, and ``combination.membership[i]`` belongs to ``sets[i]``.
    ``groups`` fixes the colour order; while it is empty the order is derived from
    the columns.
    """

    sets: tuple[str, ...]
    combinations: tuple[Combination, ...]
    groups: tuple[str, ...] = ()
    count_label: str = "Count"
    title: str = ""
    set_label_colors: tuple[str, ...] | None = None
    count_format: str = "{:.0f}"
    panel_label: str | None = None

    @classmethod
    def from_memberships(
        cls,
        *,
        sets: Sequence[str],
        memberships: Sequence[Sequence[bool]],
        counts: Sequence[float],
        groups: Sequence[str | None] | None = None,
        count_label: str = "Count",
        title: str = "",
        set_label_colors: Sequence[str] | None = None,
        count_format: str = "{:.0f}",
        panel_label: str | None = None,
    ) -> "ConcordanceUpsetData":
        """Build from one membership row and one count per combination.

        ``memberships[j][i]`` is the membership of combination ``j`` in
        ``sets[i]``. ``groups`` carries one label per combination, or ``None``
        where a column has no grouping; the colour order follows first
        appearance, so passing ``None`` leaves every bar ungrouped.
        """

        rows = tuple(tuple(bool(flag) for flag in row) for row in memberships)
        values = tuple(float(count) for count in counts)
        if len(rows) != len(values):
            raise ValueError(
                f"memberships has {len(rows)} rows but counts has "
                f"{len(values)} entries"
            )
        labels: tuple[str | None, ...] = (
            (None,) * len(values)
            if groups is None
            else tuple(None if name is None else str(name) for name in groups)
        )
        if len(labels) != len(values):
            raise ValueError(
                f"groups has {len(labels)} entries but there are "
                f"{len(values)} combinations"
            )
        combinations = tuple(
            Combination(membership=row, count=count, group=label)
            for row, count, label in zip(rows, values, labels, strict=True)
        )
        built = cls(
            sets=tuple(str(name) for name in sets),
            combinations=combinations,
            groups=_first_appearance(combinations),
            count_label=count_label,
            title=title,
            set_label_colors=(
                None
                if set_label_colors is None
                else tuple(str(color) for color in set_label_colors)
            ),
            count_format=count_format,
            panel_label=panel_label,
        )
        built.validate()
        return built

    def validate(self) -> None:
        if len(self.sets) < 2:
            raise ValueError("need at least two sets")
        if len(set(self.sets)) != len(self.sets):
            raise ValueError("set labels must be unique")
        if len(self.combinations) < 2:
            raise ValueError("need at least two combinations")
        if len(set(self.groups)) != len(self.groups):
            raise ValueError("group labels must be unique")
        if (
            self.set_label_colors is not None
            and len(self.set_label_colors) != len(self.sets)
        ):
            raise ValueError(
                f"set_label_colors has {len(self.set_label_colors)} entries but "
                f"there are {len(self.sets)} sets"
            )
        known = set(self.group_order())
        seen: set[tuple[bool, ...]] = set()
        for index, item in enumerate(self.combinations):
            if len(item.membership) != len(self.sets):
                raise ValueError(
                    f"combination {index} carries {len(item.membership)} membership "
                    f"flags but there are {len(self.sets)} sets"
                )
            if not math.isfinite(item.count) or item.count < 0.0:
                raise ValueError(
                    f"combination {index} has count {item.count!r}; counts must be "
                    "finite and non-negative"
                )
            if item.group is not None and item.group not in known:
                raise ValueError(
                    f"combination {index} names group {item.group!r}, which is not "
                    "one of the declared groups"
                )
            if item.membership in seen:
                raise ValueError(
                    f"combination {index} repeats the membership pattern of an "
                    "earlier column"
                )
            seen.add(item.membership)
        try:
            self.count_format.format(1.0)
        except (IndexError, KeyError, ValueError) as exc:
            raise ValueError(f"count_format {self.count_format!r} is not usable") from exc

    def group_order(self) -> tuple[str, ...]:
        """Colour order for the bars, derived from the columns while unset."""

        return self.groups or _first_appearance(self.combinations)

    def counts(self) -> tuple[float, ...]:
        return tuple(item.count for item in self.combinations)

    def membership_matrix(self) -> NDArray[np.bool_]:
        """Memberships as a ``(sets, combinations)`` boolean array."""

        return np.asarray(
            [item.membership for item in self.combinations],
            dtype=bool,
        ).T

    def count_range(self) -> tuple[float, float]:
        counts = self.counts()
        return min(0.0, min(counts)), max(counts)

    def total_count(self) -> float:
        return float(sum(self.counts()))

    def row_colors(self) -> tuple[str, ...]:
        """Ink for each matrix row label, neutral unless the data names one."""

        if self.set_label_colors is not None:
            return self.set_label_colors
        return tuple(SET_LABEL_COLOR for _ in self.sets)


_REFERENCE_SETS: Final[tuple[str, ...]] = (
    "With plasma-specific variants",
    "With tissue-specific variants",
    "With shared variants",
)

_REFERENCE_SET_LABEL_COLORS: Final[tuple[str, ...]] = (
    "#E17D65",
    "#276E8F",
    "#515151",
)

_REFERENCE_MEMBERSHIPS: Final[tuple[tuple[bool, ...], ...]] = (
    (False, True, False),
    (True, False, False),
    (True, True, False),
    (False, False, False),
    (False, False, True),
    (True, False, True),
    (False, True, True),
    (True, True, True),
)

_REFERENCE_COUNTS: Final[tuple[float, ...]] = (420, 7, 76, 8, 42, 99, 209, 250)

_REFERENCE_COMBINATION_GROUPS: Final[tuple[str, ...]] = (
    "Disconcordant",
    "Disconcordant",
    "Disconcordant",
    "Complete concordant",
    "Complete concordant",
    "Partially concordant",
    "Partially concordant",
    "Partially concordant",
)

DEFAULT_DATA: Final[ConcordanceUpsetData] = ConcordanceUpsetData.from_memberships(
    sets=_REFERENCE_SETS,
    memberships=_REFERENCE_MEMBERSHIPS,
    counts=_REFERENCE_COUNTS,
    groups=_REFERENCE_COMBINATION_GROUPS,
    count_label="Number of patients",
    title="Patient-Level Concordance",
    set_label_colors=_REFERENCE_SET_LABEL_COLORS,
    panel_label="b",
)


@dataclass(frozen=True, slots=True)
class ChartStyle:
    """Layout tuned to the 2431 x 1603 reference image.

    The matrix panel is allotted ``matrix_row_ratio`` of height per set, so the
    reference three sets reproduce the original 3.28:1 panel split while more
    sets grow the matrix instead of squeezing its rows.  Label offsets are
    fractions of the resolved y-range, which keeps them correct for counts of
    any magnitude.
    """

    figure_size: tuple[float, float] = (9.724, 6.412)
    subplot_left: float = 0.358
    subplot_right: float = 0.986
    subplot_top: float = 0.855
    subplot_bottom: float = 0.048
    bar_panel_ratio: float = 3.28
    matrix_row_ratio: float = 1.0 / 3.0
    hspace: float = 0.035
    y_limits: tuple[float, float] | None = (0.0, 478.0)
    y_ticks: tuple[float, ...] | None = (0.0, 150.0, 300.0, 450.0)
    y_minor_tick: float = 50.0
    column_pad: float = 0.65
    bar_width: float = 0.62
    value_pad_fraction: float = 7.0 / 478.0
    group_label_pad_fraction: float = 22.0 / 478.0
    spine_width: float = 1.45
    tick_length: float = 5.0
    bar_edge_width: float = 0.85
    panel_label_position: tuple[float, float] = (0.028, 0.955)
    title_font_size: float = 18.0
    panel_font_size: float = 20.0
    label_font_size: float = 15.0
    tick_font_size: float = 12.0
    value_font_size: float = 12.5
    group_font_size: float = 13.5
    row_font_size: float = 12.5
    mark_font_size: float = 17.0
    luminance_cutoff: float = 0.93

    def validate(self, *, sets: int, combinations: int) -> None:
        if min(self.figure_size) <= 0:
            raise ValueError("figure_size must be positive")
        if sets < 1 or combinations < 1:
            raise ValueError("sets and combinations must be positive")
        if self.y_limits is not None and self.y_limits[1] <= self.y_limits[0]:
            raise ValueError("y_limits must be increasing")
        if not 0 < self.bar_width < 1.0:
            raise ValueError("bar_width must sit in (0, 1)")
        if self.column_pad <= 0:
            raise ValueError("column_pad must be positive")
        if min(self.bar_panel_ratio, self.matrix_row_ratio) <= 0:
            raise ValueError("panel height ratios must be positive")
        if self.subplot_right <= self.subplot_left:
            raise ValueError("subplot x-bounds must be increasing")
        if self.subplot_top <= self.subplot_bottom:
            raise ValueError("subplot y-bounds must be increasing")


DEFAULT_STYLE: Final[ChartStyle] = ChartStyle()


def panel_height_ratios(style: ChartStyle, *, sets: int) -> tuple[float, float]:
    """Height split between the bar panel and the membership matrix."""

    return (style.bar_panel_ratio, style.matrix_row_ratio * sets)


def column_limits(style: ChartStyle, *, combinations: int) -> tuple[float, float]:
    """x-limits shared by the bar panel and the matrix panel."""

    return (-style.column_pad, combinations - 1 + style.column_pad)


def bar_axis_ticks(
    style: ChartStyle, limits: tuple[float, float]
) -> tuple[tuple[float, ...], float]:
    """Major tick values and the minor step for the resolved y-limits.

    The pinned reference ticks are kept while the pinned limits survive; an
    auto-fitted range gets a 1/2/5 tick set instead.
    """

    pinned = style.y_limits
    if style.y_ticks is not None and pinned is not None and np.allclose(pinned, limits):
        return tuple(float(value) for value in style.y_ticks), style.y_minor_tick
    low, high = limits
    step = nice_step(high - low)
    first = float(np.ceil(low / step) * step)
    count = int(np.floor((high - first) / step)) + 1
    return tuple(first + index * step for index in range(count)), 0.5 * step


def label_margin(data: ConcordanceUpsetData, style: ChartStyle) -> float:
    """Left subplot edge, widened when set labels outgrow the reference ones."""

    longest = max(len(name) for name in data.sets)
    # A serif face averages about half its point size per character.
    text_points = 0.5 * style.row_font_size * longest
    gutter_points = 1.75 * style.row_font_size
    needed = (text_points + gutter_points) / (72.0 * style.figure_size[0])
    return max(style.subplot_left, needed)


def bar_colors(data: ConcordanceUpsetData, palette: Palette) -> tuple[str, ...]:
    """One fill colour per combination, indexed by its group's position."""

    order = data.group_order()
    colors = palette.take(max(len(order), 1))
    return tuple(
        colors[0] if item.group is None else colors[order.index(item.group)]
        for item in data.combinations
    )


def _style_bar_axis(
    ax: Axes,
    data: ConcordanceUpsetData,
    style: ChartStyle,
    *,
    limits: tuple[float, float],
    ticks: Sequence[float],
    minor_step: float,
) -> None:
    ax.set_ylim(*limits)
    ax.yaxis.set_major_locator(FixedLocator(list(ticks)))
    ax.yaxis.set_minor_locator(MultipleLocator(minor_step))
    ax.tick_params(
        axis="y",
        which="major",
        direction="out",
        left=True,
        right=False,
        length=style.tick_length,
        width=style.spine_width,
        labelsize=style.tick_font_size,
        pad=5.0,
    )
    ax.tick_params(
        axis="y",
        which="minor",
        direction="out",
        left=True,
        right=False,
        length=style.tick_length * 0.55,
        width=style.spine_width,
    )
    ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    ax.set_ylabel(
        data.count_label,
        fontsize=style.label_font_size,
        labelpad=10.0,
    )
    if data.title:
        ax.set_title(
            data.title,
            fontsize=style.title_font_size,
            pad=10.0,
        )
    for spine_name, spine in ax.spines.items():
        spine.set_visible(spine_name in {"left", "bottom"})
        spine.set_linewidth(style.spine_width)
        spine.set_color("black")
    ax.set_facecolor("white")
    ax.grid(False)


def _draw_bars(
    ax: Axes,
    data: ConcordanceUpsetData,
    palette: Palette,
    style: ChartStyle,
    limits: tuple[float, float],
) -> None:
    centers = np.arange(len(data.combinations), dtype=float)
    value_offset = style.value_pad_fraction * (limits[1] - limits[0])
    ax.bar(
        centers,
        data.counts(),
        width=style.bar_width,
        color=list(bar_colors(data, palette)),
        edgecolor="black",
        linewidth=style.bar_edge_width,
        zorder=2,
    )
    for x, item in zip(centers, data.combinations, strict=True):
        ax.text(
            x,
            item.count + value_offset,
            data.count_format.format(item.count),
            ha="center",
            va="bottom",
            fontsize=style.value_font_size,
            color="black",
            zorder=3,
        )


def _group_label_text(group: str) -> str:
    if " " in group:
        first, rest = group.split(" ", 1)
        return f"{first}\n{rest}"
    return group


def _draw_group_labels(
    ax: Axes,
    data: ConcordanceUpsetData,
    palette: Palette,
    style: ChartStyle,
    limits: tuple[float, float],
) -> None:
    centers = np.arange(len(data.combinations), dtype=float)
    span = limits[1] - limits[0]
    offset = (style.value_pad_fraction + style.group_label_pad_fraction) * span
    for index, group in enumerate(data.group_order()):
        paired = [
            (x, item)
            for x, item in zip(centers, data.combinations, strict=True)
            if item.group == group
        ]
        if not paired:
            continue
        peak = max(item.count for _, item in paired)
        ax.text(
            float(np.mean([x for x, _ in paired])),
            peak + offset,
            _group_label_text(group),
            ha="center",
            va="bottom",
            fontsize=style.group_font_size,
            fontweight="bold",
            color=darken_if_pale(palette.color(index), cutoff=style.luminance_cutoff),
            linespacing=1.05,
            zorder=4,
            clip_on=False,
        )


def _draw_matrix(
    ax: Axes,
    data: ConcordanceUpsetData,
    style: ChartStyle,
) -> None:
    matrix = data.membership_matrix()
    n_rows, n_cols = matrix.shape
    ax.set_ylim(-0.55, n_rows - 0.45)
    ax.set_xlim(*column_limits(style, combinations=n_cols))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("white")
    ax.set_axis_off()

    row_colors = data.row_colors()
    for row, present_row in enumerate(matrix):
        y = n_rows - 1 - row
        for x, present in enumerate(present_row):
            ax.text(
                x,
                y,
                CHECK_MARK if present else CROSS_MARK,
                ha="center",
                va="center",
                fontsize=style.mark_font_size,
                color=CHECK_COLOR if present else CROSS_COLOR,
                fontfamily="DejaVu Sans",
                fontweight="bold",
                zorder=2,
            )
        ax.text(
            -0.08,
            y,
            data.sets[row],
            ha="right",
            va="center",
            fontsize=style.row_font_size,
            fontweight="bold",
            color=row_colors[row],
            transform=ax.get_yaxis_transform(),
            clip_on=False,
            zorder=3,
        )


def create_figure(
    palette: Palette = PALETTES[0],
    data: ConcordanceUpsetData = DEFAULT_DATA,
    style: ChartStyle = DEFAULT_STYLE,
) -> Figure:
    """Create the UpSet-style combination figure without writing it to disk."""

    data.validate()
    n_sets = len(data.sets)
    n_columns = len(data.combinations)
    style.validate(sets=n_sets, combinations=n_columns)

    low, high = data.count_range()
    limits = resolve_limits(style.y_limits, low, high)
    ticks, minor_step = bar_axis_ticks(style, limits)

    with plt.rc_context(
        {
            "font.family": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
            "font.size": style.tick_font_size,
            "axes.linewidth": style.spine_width,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    ):
        figure = plt.figure(figsize=style.figure_size, facecolor="white")
        grid = figure.add_gridspec(
            2,
            1,
            height_ratios=list(panel_height_ratios(style, sets=n_sets)),
            left=label_margin(data, style),
            right=style.subplot_right,
            top=style.subplot_top,
            bottom=style.subplot_bottom,
            hspace=style.hspace,
        )
        ax_bar = figure.add_subplot(grid[0])
        ax_mat = figure.add_subplot(grid[1], sharex=ax_bar)

        _style_bar_axis(
            ax_bar,
            data,
            style,
            limits=limits,
            ticks=ticks,
            minor_step=minor_step,
        )
        _draw_bars(ax_bar, data, palette, style, limits)
        _draw_group_labels(ax_bar, data, palette, style, limits)
        ax_bar.set_xlim(*column_limits(style, combinations=n_columns))
        _draw_matrix(ax_mat, data, style)

        if data.panel_label is not None:
            x, y = style.panel_label_position
            figure.text(
                x,
                y,
                data.panel_label,
                fontsize=style.panel_font_size,
                fontweight="bold",
                ha="left",
                va="top",
            )

    return figure


def main(argv: Sequence[str] | None = None) -> int:
    return run_cli(sys.modules[__name__], argv)


if __name__ == "__main__":
    raise SystemExit(main())
