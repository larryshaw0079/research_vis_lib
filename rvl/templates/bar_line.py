"""Aggregate bars stacked above the ordered measurement series that produced them.

The top panel draws one horizontal bar per series -- the per-series aggregate --
with replicate points, an error bar and an optional significance bracket.  The
bottom panel draws the ordered measurement series the aggregate came from, with
error bars, dashed period guides and shaded period bands.

``DEFAULT_DATA`` holds the LF / HF N2O emission figure digitised from a
Xiaohongshu carousel that repeats the same two panels 18 times, changing only
the colour pair.  The source post publishes the chart but not the table, and
notes that the top-panel bars were a naive sum of the flux points rather than a
true integrated cumulative flux; :func:`naive_cumulative` keeps that shortcut so
the reference bars match the carousel.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Final, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator
from numpy.typing import NDArray

from ..contract import DataKind, Extent, Feature, Geometry, TemplateSpec
from ..palettes import Palette
from ..render import nice_step, resolve_limits, run_cli

SPEC: Final[TemplateSpec] = TemplateSpec(
    template_id="bar-line",
    title="Aggregate bars above an ordered measurement series",
    summary=(
        "A horizontal bar of per-series aggregates with replicate points and a "
        "significance bracket, stacked above the ordered measurement series that "
        "produced them, with dashed period guides and shaded background bands."
    ),
    kinds=(DataKind.SERIES_WITH_TOTALS,),
    geometry=Geometry.COMPOSITE,
    categories=Extent(3, 40),
    series=Extent(2, 6),
    builder="BarLineData.from_series",
    data_contract=(
        "One measurement per (series, ordered timepoint) pair, optional standard "
        "errors, an optional per-series aggregate with replicate observations, and "
        "optional period boundaries drawn as dashed guides."
    ),
    good_for=(
        "repeated measurements over time under a few treatments",
        "an aggregate or cumulative total that must sit beside its own time series",
        "replicate-level scatter and a significance annotation on the aggregates",
    ),
    avoid_when=(
        "the category axis carries no order",
        "more than about 40 timepoints, where markers collide",
        "more than six series, where the aggregate panel runs out of rows",
    ),
    ordered_categories=True,
    affinities=(
        (Feature.HAS_UNCERTAINTY, 7.0),
        (Feature.MANY_SERIES, -12.0),
        (Feature.WIDE_DYNAMIC_RANGE, -5.0),
    ),
    default_dpi=150,
    reference="Digitised from a Xiaohongshu carousel; the source publishes no table.",
)


# Colours sampled from the two bar interiors in carousel frames 2-18.
# Frame 1 is a 3x3 preview sheet; its unused rose-sage pair is palette 18.
PALETTES: Final[tuple[Palette, ...]] = (
    Palette("orange-green", ("#FA882F", "#338227")),
    Palette("coral-steel", ("#DA5D47", "#538CB9")),
    Palette("orange-seafoam", ("#FA882F", "#73B7A0")),
    Palette("rose-teal", ("#CD89A0", "#539780")),
    Palette("salmon-lime", ("#FF9D87", "#93E287")),
    Palette("gold-teal", ("#EDBF4F", "#539780")),
    Palette("peach-seafoam", ("#EDB487", "#73B7A0")),
    Palette("pink-leaf", ("#EDA9C0", "#53A247")),
    Palette("orange-mauve", ("#FA882F", "#AD6980")),
    Palette("burgundy-wine", ("#4D0920", "#9A1240")),
    Palette("navy-amber", ("#336C99", "#FA9300")),
    Palette("gray-silver", ("#808080", "#C0C0C0")),
    Palette("navy-crimson", ("#336C99", "#E72627")),
    Palette("charcoal-coral", ("#404040", "#DA5D47")),
    Palette("navy-sky", ("#336C99", "#93CCF9")),
    Palette("crimson-amber", ("#E72627", "#FA9300")),
    Palette("azure-peach", ("#0678D1", "#EDB487")),
    Palette("rose-sage", ("#E8A0B4", "#A8C86C")),
)

# The reference alternates a square and a circle; longer cycles keep every
# series distinguishable in print.
_MARKER_CYCLE: Final[tuple[str, ...]] = ("s", "o", "^", "D", "v", "P")


def naive_cumulative(flux: Sequence[float]) -> float:
    """Sum measurement points the way the source post built the top bars.

    The reference bars are a plain sum of the plotted flux points rather than a
    time-weighted integral; keeping the shortcut here is what makes
    ``DEFAULT_DATA`` reproduce the carousel.
    """

    values = np.asarray(flux, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("flux must be a non-empty 1-D sequence")
    if not np.all(np.isfinite(values)):
        raise ValueError("flux values must be finite")
    return float(values.sum())


def replicate_sem(values: Sequence[float]) -> float:
    """Standard error of the mean of a replicate sample."""

    sample = np.asarray(values, dtype=float)
    if sample.size < 2:
        raise ValueError("need at least two replicates to compute SEM")
    return float(sample.std(ddof=1) / np.sqrt(sample.size))


@dataclass(frozen=True, slots=True)
class BarLineData:
    """One ordered measurement sequence per series plus its per-series aggregate.

    ``values`` is indexed ``[series][point]`` because each series is its own
    sequence over the shared, ordered ``points`` axis.
    """

    series: tuple[str, ...]
    points: tuple[str, ...]
    values: tuple[tuple[float, ...], ...]
    errors: tuple[tuple[float, ...], ...] | None = None
    totals: tuple[float, ...] | None = None
    total_errors: tuple[float, ...] | None = None
    replicates: tuple[tuple[float, ...], ...] | None = None
    period_boundaries: tuple[float, ...] = ()
    value_label: str = "Value"
    total_label: str = "Total"
    significance: str | None = None
    panel_label: str | None = None

    @classmethod
    def from_series(
        cls,
        *,
        series: Sequence[str],
        points: Sequence[str],
        values: Sequence[Sequence[float]],
        errors: Sequence[Sequence[float]] | None = None,
        totals: Sequence[float] | None = None,
        total_errors: Sequence[float] | None = None,
        replicates: Sequence[Sequence[float]] | None = None,
        period_boundaries: Sequence[float] = (),
        value_label: str = "Value",
        total_label: str = "Total",
        significance: str | None = None,
        panel_label: str | None = None,
    ) -> "BarLineData":
        """Build from one ``[series][point]`` measurement row per series.

        ``totals`` may be omitted, in which case each bar is the sum of its own
        row.  ``replicates`` rows may be ragged: a series with more replicate
        observations than another simply scatters more points.
        """

        built = cls(
            series=tuple(str(name) for name in series),
            points=tuple(str(name) for name in points),
            values=_rows(values),
            errors=None if errors is None else _rows(errors),
            totals=None if totals is None else tuple(float(value) for value in totals),
            total_errors=(
                None
                if total_errors is None
                else tuple(float(value) for value in total_errors)
            ),
            replicates=None if replicates is None else _rows(replicates),
            period_boundaries=tuple(float(edge) for edge in period_boundaries),
            value_label=value_label,
            total_label=total_label,
            significance=significance,
            panel_label=panel_label,
        )
        built.validate()
        return built

    def validate(self) -> None:
        n_series = len(self.series)
        n_points = len(self.points)
        if n_series < 1:
            raise ValueError("need at least one series")
        if n_points < 2:
            raise ValueError("need at least two ordered points")
        if len(set(self.series)) != n_series:
            raise ValueError("series labels must be unique")
        if len(set(self.points)) != n_points:
            raise ValueError("point labels must be unique")

        for field_name, rows in (("values", self.values), ("errors", self.errors)):
            if rows is None:
                continue
            if len(rows) != n_series:
                raise ValueError(
                    f"{field_name} has {len(rows)} rows but there are "
                    f"{n_series} series"
                )
            for name, row in zip(self.series, rows, strict=True):
                if len(row) != n_points:
                    raise ValueError(
                        f"{field_name}[{name!r}] has {len(row)} entries but there "
                        f"are {n_points} points"
                    )
                if not all(np.isfinite(value) for value in row):
                    raise ValueError(f"{field_name}[{name!r}] must be finite")
            if field_name == "errors" and any(
                value < 0.0 for row in rows for value in row
            ):
                raise ValueError("errors must be non-negative")

        for field_name, column in (
            ("totals", self.totals),
            ("total_errors", self.total_errors),
        ):
            if column is None:
                continue
            if len(column) != n_series:
                raise ValueError(
                    f"{field_name} has {len(column)} entries but there are "
                    f"{n_series} series"
                )
            if not all(np.isfinite(value) for value in column):
                raise ValueError(f"{field_name} must be finite")
        if self.total_errors is not None and any(
            value < 0.0 for value in self.total_errors
        ):
            raise ValueError("total_errors must be non-negative")

        if self.replicates is not None:
            if len(self.replicates) != n_series:
                raise ValueError(
                    f"replicates has {len(self.replicates)} rows but there are "
                    f"{n_series} series"
                )
            for name, row in zip(self.series, self.replicates, strict=True):
                if not row:
                    raise ValueError(f"replicates[{name!r}] must not be empty")
                if not all(np.isfinite(value) for value in row):
                    raise ValueError(f"replicates[{name!r}] must be finite")

        edges = self.period_boundaries
        if any(not np.isfinite(edge) for edge in edges):
            raise ValueError("period_boundaries must be finite")
        if any(later <= earlier for earlier, later in zip(edges, edges[1:])):
            raise ValueError("period_boundaries must increase")
        if any(not -0.5 <= edge <= n_points - 0.5 for edge in edges):
            raise ValueError(
                f"period_boundaries must be point-index positions in "
                f"[-0.5, {n_points - 0.5}]"
            )
        if self.significance is not None and not self.significance.strip():
            raise ValueError("significance must be a non-empty marker")

    def matrix(self) -> NDArray[np.float64]:
        """Measurements as a ``(series, points)`` array."""

        return np.asarray(self.values, dtype=float)

    def series_values(self, index: int) -> NDArray[np.float64]:
        return np.asarray(self.values[index], dtype=float)

    def series_errors(self, index: int) -> NDArray[np.float64]:
        if self.errors is None:
            return np.zeros(len(self.points), dtype=float)
        return np.asarray(self.errors[index], dtype=float)

    def replicate_values(self, index: int) -> NDArray[np.float64]:
        if self.replicates is None:
            return np.empty(0, dtype=float)
        return np.asarray(self.replicates[index], dtype=float)

    def effective_totals(self) -> tuple[float, ...]:
        """Declared aggregates, or the per-series sum when none were given."""

        if self.totals is not None:
            return self.totals
        return tuple(float(sum(row)) for row in self.values)

    def total_stats(self, index: int) -> tuple[float, float]:
        """Aggregate and its error bar for one series.

        The error is the declared ``total_errors`` entry, otherwise the SEM of
        that series' replicates, otherwise zero.
        """

        value = self.effective_totals()[index]
        if self.total_errors is not None:
            return value, float(self.total_errors[index])
        sample = self.replicate_values(index)
        if sample.size >= 2:
            return value, replicate_sem(sample)
        return value, 0.0

    def total_extent(self) -> tuple[float, float]:
        """Span the bars, their error bars and their replicate points cover."""

        lows = [0.0]
        highs = [0.0]
        for index in range(len(self.series)):
            value, error = self.total_stats(index)
            lows.append(value - error)
            highs.append(value + error)
            sample = self.replicate_values(index)
            if sample.size:
                lows.append(float(sample.min()))
                highs.append(float(sample.max()))
        return min(lows), max(highs)

    def bar_edge(self) -> float:
        """Right-hand tip of the longest bar, error bar included."""

        return max(
            sum(self.total_stats(index)) for index in range(len(self.series))
        )

    def value_extent(self) -> tuple[float, float]:
        """Span the measurement series and their error bars cover."""

        values = self.matrix()
        errors = np.asarray(
            [self.series_errors(index) for index in range(len(self.series))]
        )
        return (
            min(0.0, float((values - errors).min())),
            max(0.0, float((values + errors).max())),
        )


def _rows(values: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    """Coerce nested sequences to float tuples, keeping missing entries as NaN."""

    return tuple(
        tuple(float("nan") if value is None else float(value) for value in row)
        for row in values
    )


_REFERENCE_SERIES: Final[tuple[str, ...]] = ("LF", "HF")

_REFERENCE_DATES: Final[tuple[str, ...]] = (
    "20 Oct.",
    "26 Oct.",
    "31 Oct.",
    "05 Nov.",
    "13 Nov.",
    "31 Dec.",
    "12 Mar.",
    "20 Mar.",
    "28 Mar.",
    "05 Apr.",
    "15 Apr.",
)

# Digitised from the visible markers in carousel frame 2 (orange / green).
_REFERENCE_FLUX: Final[tuple[tuple[float, ...], ...]] = (
    (16.0, 48.0, 26.0, 9.0, 6.0, 8.0, 10.0, 23.0, 85.0, 58.0, 14.0),
    (26.0, 45.0, 34.0, 25.0, 6.0, 12.0, 34.0, 39.0, 152.0, 79.0, 15.0),
)

_REFERENCE_FLUX_ERR: Final[tuple[tuple[float, ...], ...]] = (
    (3.0, 6.0, 4.0, 2.0, 2.0, 2.0, 3.0, 4.0, 8.0, 6.0, 3.0),
    (4.0, 6.0, 4.0, 4.0, 2.0, 3.0, 4.0, 5.0, 12.0, 8.0, 3.0),
)

# Four visible replicates clustered at the bar tips in the reference frames.
_REFERENCE_REPLICATES: Final[tuple[tuple[float, ...], ...]] = (
    (271.0, 292.0, 311.0, 338.0),
    (428.0, 451.0, 478.0, 511.0),
)

# The first dashed guide sits on 13 Nov., the second on 12 Mar.
_REFERENCE_BOUNDARIES: Final[tuple[float, ...]] = (4.0, 6.0)

DEFAULT_DATA: Final[BarLineData] = BarLineData.from_series(
    series=_REFERENCE_SERIES,
    points=_REFERENCE_DATES,
    values=_REFERENCE_FLUX,
    errors=_REFERENCE_FLUX_ERR,
    totals=tuple(naive_cumulative(row) for row in _REFERENCE_FLUX),
    total_errors=tuple(replicate_sem(row) for row in _REFERENCE_REPLICATES),
    replicates=_REFERENCE_REPLICATES,
    period_boundaries=_REFERENCE_BOUNDARIES,
    value_label=r"$N_2O$ emission flux ($\mu g \cdot m^{-2} \cdot h^{-1}$)",
    total_label=r"$N_2O$ cumulative emission ($kg \cdot N_2O \cdot ha^{-1}$)",
    significance="**",
    panel_label="(a)",
)


@dataclass(frozen=True, slots=True)
class ChartStyle:
    """Layout tuned to the 1260 x 1080 reference image.

    The aggregate panel gives every series one row of unit pitch, of which
    ``bar_height`` is filled, so the reference two rows and the six rows ``SPEC``
    advertises share the same fixed panel.  Both limit tuples stay pinned to the
    reference range and auto-fit whenever data would fall outside them or would
    only fill a sliver of them; the significance and tick offsets are fractions
    of the resolved span so they survive a change of units.
    """

    figure_size: tuple[float, float] = (8.4, 7.2)
    bar_bounds: tuple[float, float, float, float] = (0.168, 0.705, 0.762, 0.198)
    line_bounds: tuple[float, float, float, float] = (0.168, 0.168, 0.762, 0.428)
    bar_x_limits: tuple[float, float] | None = (0.0, 640.0)
    line_y_limits: tuple[float, float] | None = (0.0, 165.0)
    bar_tick_step: float | None = 150.0
    line_tick_step: float | None = None
    row_margin: float = 0.72
    point_margin: float = 0.55
    bar_height: float = 0.46
    replicate_jitter: float = 0.13
    significance_offset: float = 28.0 / 640.0
    significance_label_gap: float = 10.0 / 640.0
    line_width: float = 1.85
    marker_size: float = 7.2
    spine_width: float = 1.45
    tick_length: float = 4.8
    label_font_size: float = 13.0
    tick_font_size: float = 10.5
    panel_font_size: float = 16.0
    significance_font_size: float = 13.0
    panel_color: str = "#1F4E8C"
    panel_label_position: tuple[float, float] = (0.042, 0.955)
    line_facecolor: str = "#FBFCF8"
    guide_color: str = "#8D8D8D"
    x_label_pad: float = 8.0
    y_label_pad: float = 6.0

    def validate(self, *, categories: int, series: int) -> None:
        if min(self.figure_size) <= 0:
            raise ValueError("figure_size must be positive")
        for name, limits in (
            ("bar_x_limits", self.bar_x_limits),
            ("line_y_limits", self.line_y_limits),
        ):
            if limits is not None and limits[1] <= limits[0]:
                raise ValueError(f"{name} must be increasing")
        if not 0.15 < self.bar_height < 0.8:
            raise ValueError("bar_height must sit in (0.15, 0.8)")
        if self.row_margin <= 0 or self.point_margin <= 0:
            raise ValueError("row_margin and point_margin must be positive")
        if categories < 2:
            raise ValueError("the measurement axis needs at least two points")
        if series < 1:
            raise ValueError("need at least one series")


DEFAULT_STYLE: Final[ChartStyle] = ChartStyle()


def row_positions(series: int) -> NDArray[np.float64]:
    """Bar rows from the top down, so the first series sits highest."""

    return np.arange(series, dtype=float)[::-1]


def axis_step(pinned: float | None, span: float) -> float:
    """Honour a pinned tick step while it still cuts the axis into a few ticks."""

    if pinned is not None and pinned > 0.0 and 2.0 <= span / pinned <= 12.0:
        return pinned
    return nice_step(span)


def fitted_limits(
    pinned: tuple[float, float] | None,
    low: float,
    high: float,
    *,
    fill: float = 0.35,
) -> tuple[float, float]:
    """Resolve axis limits, dropping a pin the data would barely fill.

    :func:`rvl.render.resolve_limits` keeps a pinned range whenever the data fits
    inside it, which is what preserves the reference layout.  A dataset in other
    units can fit inside the reference range and still occupy a sliver of it, so
    a pin the data fills less than ``fill`` of is dropped and the range auto-fits.
    """

    if pinned is not None and (high - low) < fill * (pinned[1] - pinned[0]):
        pinned = None
    return resolve_limits(pinned, low, high)


def period_bands(
    boundaries: Sequence[float], low: float, high: float
) -> tuple[tuple[float, float], ...]:
    """Shaded spans for the periods the dashed guides cut the axis into.

    Periods alternate shaded and clear so each guide reads as a separator.  With
    the reference's two boundaries this shades the span before the first guide
    and the span after the last one, leaving the middle period clear.
    """

    if not boundaries:
        return ()
    edges = (low, *boundaries, high)
    return tuple(
        (edges[index], edges[index + 1]) for index in range(0, len(edges) - 1, 2)
    )


def _period_cmap(color: str) -> LinearSegmentedColormap:
    red, green, blue = to_rgb(color)
    return LinearSegmentedColormap.from_list(
        "period",
        (
            (0.0, (red, green, blue, 0.03)),
            (0.45, (red, green, blue, 0.14)),
            (1.0, (red, green, blue, 0.04)),
        ),
        N=256,
    )


def _bracket_x(
    data: BarLineData, style: ChartStyle, limits: tuple[float, float]
) -> float | None:
    """Where the significance bracket stands, or ``None`` when it is not drawn."""

    if data.significance is None:
        return None
    return data.bar_edge() + style.significance_offset * (limits[1] - limits[0])


def _style_bar_axis(
    ax: Axes,
    data: BarLineData,
    style: ChartStyle,
    limits: tuple[float, float],
) -> None:
    n_series = len(data.series)
    ax.set_xlim(*limits)
    ax.set_ylim(-style.row_margin, n_series - 1 + style.row_margin)
    ax.set_yticks(row_positions(n_series), labels=list(data.series))
    ax.xaxis.set_major_locator(
        MultipleLocator(axis_step(style.bar_tick_step, limits[1] - limits[0]))
    )
    ax.tick_params(
        axis="x",
        which="major",
        direction="in",
        top=False,
        bottom=True,
        length=style.tick_length,
        width=style.spine_width,
        labelsize=style.tick_font_size,
        pad=3.5,
    )
    ax.tick_params(
        axis="y",
        which="major",
        direction="in",
        left=True,
        right=False,
        length=style.tick_length,
        width=style.spine_width,
        labelsize=style.tick_font_size,
        pad=4.0,
    )
    ax.set_xlabel(
        data.total_label,
        fontsize=style.label_font_size,
        fontweight="bold",
        labelpad=style.x_label_pad,
    )
    for label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        label.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(style.spine_width)
        spine.set_color("black")
    ax.set_axisbelow(False)
    ax.grid(False)


def _point_font_size(data: BarLineData, style: ChartStyle) -> float:
    """Shrink the point labels once they are closer together than the font."""

    pitch = 72.0 * style.figure_size[0] * style.line_bounds[2] / len(data.points)
    return min(style.tick_font_size, max(5.5, 0.92 * pitch))


def _style_line_axis(
    ax: Axes,
    data: BarLineData,
    style: ChartStyle,
    limits: tuple[float, float],
) -> None:
    n_points = len(data.points)
    ax.set_xlim(-style.point_margin, n_points - 1 + style.point_margin)
    ax.set_ylim(*limits)
    ax.yaxis.set_major_locator(
        MultipleLocator(axis_step(style.line_tick_step, limits[1] - limits[0]))
    )
    ax.set_xticks(np.arange(n_points, dtype=float))
    ax.set_xticklabels(data.points, rotation=90, ha="center", va="top")
    ax.tick_params(
        which="major",
        direction="in",
        top=True,
        right=True,
        length=style.tick_length,
        width=style.spine_width,
        labelsize=style.tick_font_size,
        pad=3.5,
    )
    ax.tick_params(axis="x", labelsize=_point_font_size(data, style))
    ax.set_ylabel(
        data.value_label,
        fontsize=style.label_font_size,
        fontweight="bold",
        labelpad=style.y_label_pad,
    )
    for label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        label.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(style.spine_width)
        spine.set_color("black")
    ax.set_axisbelow(False)
    ax.grid(False)


def _draw_bars(
    ax: Axes,
    data: BarLineData,
    palette: Palette,
    style: ChartStyle,
    limits: tuple[float, float],
    bracket_x: float | None,
) -> None:
    n_series = len(data.series)
    colors = palette.take(n_series)
    rows = row_positions(n_series)
    rng = np.random.default_rng(20260819)

    for index in range(n_series):
        color = colors[index]
        y = float(rows[index])
        value, error = data.total_stats(index)
        ax.barh(
            y,
            value,
            height=style.bar_height,
            color=color,
            edgecolor="black",
            linewidth=1.15,
            zorder=2,
        )
        if error > 0.0:
            ax.errorbar(
                value,
                y,
                xerr=error,
                fmt="none",
                ecolor="black",
                elinewidth=1.25,
                capsize=3.8,
                capthick=1.25,
                zorder=4,
            )
        sample = data.replicate_values(index)
        if not sample.size:
            continue
        jitter = rng.uniform(-style.replicate_jitter, style.replicate_jitter, sample.size)
        ax.scatter(
            sample,
            y + jitter,
            s=36,
            facecolors=color,
            edgecolors="black",
            linewidths=0.85,
            zorder=5,
        )

    if bracket_x is None:
        return
    ax.plot(
        [bracket_x, bracket_x],
        [float(rows[-1]), float(rows[0])],
        color="black",
        linewidth=1.35,
        solid_capstyle="butt",
        zorder=6,
    )
    ax.text(
        bracket_x + style.significance_label_gap * (limits[1] - limits[0]),
        0.5 * float(rows[0] + rows[-1]),
        data.significance,
        ha="left",
        va="center",
        fontsize=style.significance_font_size,
        fontweight="bold",
        zorder=6,
    )


def _draw_periods(
    ax: Axes,
    data: BarLineData,
    palette: Palette,
    style: ChartStyle,
    limits: tuple[float, float],
) -> None:
    if not data.period_boundaries:
        return
    x_low = -style.point_margin
    x_high = len(data.points) - 1 + style.point_margin
    y_low, y_high = limits
    gradient = np.linspace(0.0, 1.0, 256).reshape(1, -1)
    for index, (start, end) in enumerate(
        period_bands(data.period_boundaries, x_low, x_high)
    ):
        image = ax.imshow(
            gradient,
            extent=(start, end, y_low, y_high),
            origin="lower",
            aspect="auto",
            cmap=_period_cmap(palette.color(index)),
            interpolation="bicubic",
            zorder=0,
        )
        image.set_clip_path(ax.patch)
    for edge in data.period_boundaries:
        ax.axvline(
            edge,
            linestyle=(0, (4.2, 3.2)),
            color=style.guide_color,
            linewidth=1.15,
            zorder=1,
        )


def _draw_lines(
    ax: Axes,
    data: BarLineData,
    palette: Palette,
    style: ChartStyle,
) -> None:
    x = np.arange(len(data.points), dtype=float)
    colors = palette.take(len(data.series))
    for index in range(len(data.series)):
        color = colors[index]
        marker = _MARKER_CYCLE[index % len(_MARKER_CYCLE)]
        ax.errorbar(
            x,
            data.series_values(index),
            yerr=None if data.errors is None else data.series_errors(index),
            fmt=f"-{marker}",
            color=color,
            ecolor=color,
            linewidth=style.line_width,
            markersize=style.marker_size,
            markerfacecolor=color,
            markeredgecolor=color,
            markeredgewidth=0.6,
            capsize=3.2,
            capthick=1.15,
            elinewidth=1.15,
            zorder=3,
        )


def create_figure(
    palette: Palette = PALETTES[0],
    data: BarLineData = DEFAULT_DATA,
    style: ChartStyle = DEFAULT_STYLE,
) -> Figure:
    """Create the aggregate-bar-plus-series figure without writing it to disk."""

    data.validate()
    style.validate(categories=len(data.points), series=len(data.series))

    bar_low, bar_high = data.total_extent()
    bar_limits = fitted_limits(style.bar_x_limits, bar_low, bar_high)
    bracket_x = _bracket_x(data, style, bar_limits)
    if bracket_x is not None:
        # The bracket and its marker stand to the right of the longest bar, so
        # the panel has to keep them inside the axes too.
        margin = 2.0 * style.significance_label_gap * (bar_limits[1] - bar_limits[0])
        bar_limits = fitted_limits(
            style.bar_x_limits, bar_low, max(bar_high, bracket_x + margin)
        )
    line_limits = fitted_limits(style.line_y_limits, *data.value_extent())

    with plt.rc_context(
        {
            "font.family": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
            "font.size": style.tick_font_size,
            "axes.linewidth": style.spine_width,
            "mathtext.fontset": "stix",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    ):
        figure = plt.figure(figsize=style.figure_size, facecolor="white")
        ax_bar = figure.add_axes(style.bar_bounds)
        ax_line = figure.add_axes(style.line_bounds)
        ax_bar.set_facecolor("white")
        ax_line.set_facecolor(style.line_facecolor)
        _style_bar_axis(ax_bar, data, style, bar_limits)
        _style_line_axis(ax_line, data, style, line_limits)
        _draw_bars(ax_bar, data, palette, style, bar_limits, bracket_x)
        _draw_periods(ax_line, data, palette, style, line_limits)
        _draw_lines(ax_line, data, palette, style)
        if data.panel_label is not None:
            figure.text(
                *style.panel_label_position,
                data.panel_label,
                ha="left",
                va="top",
                fontsize=style.panel_font_size,
                fontweight="bold",
                color=style.panel_color,
                zorder=7,
            )

    return figure


def main(argv: Sequence[str] | None = None) -> int:
    return run_cli(sys.modules[__name__], argv)


if __name__ == "__main__":
    raise SystemExit(main())
