"""Overlaid per-series histograms drawn as gradient bars with normal fits.

Each series gets grouped frequency bars carrying a vertical colour-to-white
gradient, a dashed normal-density curve scaled to the count axis, and a legend
entry printing its mean and standard deviation.

``DEFAULT_DATA`` comes from a Xiaohongshu carousel that repeats the same
residual-strain histogram 18 times, changing only the three-colour palette.  The
post publishes the chart but not the residual table, so the digitised bin counts
*are* the source: ``DEFAULT_DATA`` is built with ``from_bin_counts`` and the
printed means and standard deviations are passed through as ``stats``.  Nothing
is synthesised — the bars are the digitised counts themselves.
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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MultipleLocator
from numpy.typing import NDArray
from scipy.stats import norm

from ..contract import DataKind, Extent, Feature, Geometry, TemplateSpec
from ..palettes import Palette
from ..render import nice_step, resolve_limits, run_cli

SPEC: Final[TemplateSpec] = TemplateSpec(
    template_id="grouped-gradient-hist",
    title="Grouped histogram with gradient bars and normal fits",
    summary=(
        "Overlaid per-series histograms drawn as vertical colour-to-white gradient "
        "bars, with dashed normal-fit curves and a legend reporting each series' "
        "mean and standard deviation."
    ),
    kinds=(DataKind.DISTRIBUTION_SAMPLES,),
    geometry=Geometry.CARTESIAN,
    categories=Extent(4, 40),
    series=Extent(1, 6),
    builder="GroupedGradientHistData.from_samples",
    data_contract=(
        "Either many raw observations per series, or a shared set of bin edges "
        "with a count per (series, bin). A series needs enough observations for a "
        "distribution to mean anything."
    ),
    good_for=(
        "comparing the spread and centre of a measurement across a few groups",
        "residual or error distributions from competing models",
        "showing that one group is more tightly distributed than another",
    ),
    avoid_when=(
        "there are only a handful of observations per series",
        "the data is already a single summary value per category",
        "more than about six series, where overlaid bars become unreadable",
    ),
    affinities=(
        (Feature.MANY_SERIES, -12.0),
        (Feature.SINGLE_SERIES, 3.0),
    ),
    default_dpi=200,
    reference=(
        "Digitised from a Xiaohongshu carousel; DEFAULT_DATA carries the digitised "
        "bin counts themselves, so no observations are fabricated."
    ),
)


# Colours reconstructed from the legend swatches in carousel images 1-18.
# JPEG compression was snapped to nearby saturated hex values.
PALETTES: Final[tuple[Palette, ...]] = (
    Palette("coral-navy-teal", ("#E74D40", "#3B4A6B", "#1D8C85")),
    Palette("watermelon-azure-lime", ("#FE5A5F", "#1A82C4", "#8CC929")),
    Palette("apricot-plum-sea", ("#FE9351", "#5E4D7C", "#55A788")),
    Palette("crimson-navy-amber", ("#D62828", "#002F49", "#F67F04")),
    Palette("rose-cyan-mint", ("#EE4970", "#108AB3", "#07D6A0")),
    Palette("burgundy-pine-wine", ("#990520", "#114C5C", "#601040")),
    Palette("terracotta-slate-sage", ("#DF7B60", "#3C405D", "#81B39A")),
    Palette("scarlet-navy-steel", ("#E33A47", "#1D3557", "#457B9D")),
    Palette("coral-charcoal-steel", ("#EE6C50", "#293241", "#3D5A80")),
    Palette("tomato-lake-sea", ("#F94245", "#277CA1", "#45AA8C")),
    Palette("brick-cobalt-forest", ("#C23027", "#023F92", "#0E7D59")),
    Palette("peach-teal-mint", ("#E2967B", "#026D77", "#85C4BE")),
    Palette("crimson-ink-slate", ("#D8062A", "#2B2D42", "#8D99AE")),
    Palette("cardinal-navy-sky", ("#BF1420", "#003049", "#679BBB")),
    Palette("red-navy-olive", ("#D5293A", "#202D5A", "#5A8157")),
    Palette("salmon-petrol-teal", ("#DA514D", "#094C63", "#187E89")),
    Palette("coral-pine-sea", ("#FE715C", "#254343", "#45AA8C")),
    Palette("wine-azure-emerald", ("#B43A51", "#2274A4", "#34936E")),
)


@dataclass(frozen=True, slots=True)
class GroupedGradientHistData:
    """One distribution per series, given either as raw draws or as a bin table.

    Two input paths converge on the same drawing code: ``samples`` holds ragged
    rows of observations that are binned on demand, while ``bin_edges`` plus
    ``counts`` describes an already-tabulated histogram.  Supply exactly one.
    """

    series: tuple[str, ...]
    samples: tuple[tuple[float, ...], ...] | None = None
    bin_edges: tuple[float, ...] | None = None
    counts: tuple[tuple[float, ...], ...] | None = None
    bins: int = 12
    value_label: str = "Value"
    count_label: str = "Count"
    show_normal_fit: bool = True
    stats: tuple[tuple[float, float], ...] | None = None
    """Printed ``(mean, sd)`` per series; derived from the data when ``None``."""

    @classmethod
    def from_samples(
        cls,
        *,
        series: Sequence[str],
        samples: Sequence[Sequence[float]],
        bins: int = 12,
        value_label: str = "Value",
        count_label: str = "Count",
        show_normal_fit: bool = True,
    ) -> "GroupedGradientHistData":
        """Build from raw observations, one ragged row per series.

        Every series is binned against one shared set of ``bins`` edges spanning
        all observations, so grouped bars stay comparable across series.
        """

        built = cls(
            series=tuple(str(name) for name in series),
            samples=tuple(tuple(float(value) for value in row) for row in samples),
            bins=int(bins),
            value_label=value_label,
            count_label=count_label,
            show_normal_fit=bool(show_normal_fit),
        )
        built.validate()
        return built

    @classmethod
    def from_bin_counts(
        cls,
        *,
        series: Sequence[str],
        bin_edges: Sequence[float],
        counts: Sequence[Sequence[float]],
        value_label: str = "Value",
        count_label: str = "Count",
        show_normal_fit: bool = True,
        stats: Sequence[Sequence[float]] | None = None,
    ) -> "GroupedGradientHistData":
        """Build from a histogram table: shared edges plus a count per bin.

        ``counts`` is indexed ``[series][bin]`` and ``bin_edges`` has one more
        entry than a counts row.  Pass ``stats`` when the source publishes each
        series' mean and standard deviation; otherwise both are estimated from
        the count-weighted bin midpoints.
        """

        built = cls(
            series=tuple(str(name) for name in series),
            bin_edges=tuple(float(edge) for edge in bin_edges),
            counts=tuple(tuple(float(value) for value in row) for row in counts),
            value_label=value_label,
            count_label=count_label,
            show_normal_fit=bool(show_normal_fit),
            stats=(
                None
                if stats is None
                else tuple((float(pair[0]), float(pair[1])) for pair in stats)
            ),
        )
        built.validate()
        return built

    def validate(self) -> None:
        if not self.series:
            raise ValueError("need at least one series")
        if len(set(self.series)) != len(self.series):
            raise ValueError("series labels must be unique")
        if self.bins < 1:
            raise ValueError("bins must be positive")

        has_samples = self.samples is not None
        has_table = self.bin_edges is not None or self.counts is not None
        if has_samples and has_table:
            raise ValueError(
                "supply raw samples or a (bin_edges, counts) table, not both"
            )
        if not has_samples and not has_table:
            raise ValueError(
                "supply either samples=... or both bin_edges=... and counts=..."
            )

        if self.stats is not None:
            if len(self.stats) != len(self.series):
                raise ValueError(
                    f"stats has {len(self.stats)} entries but there are "
                    f"{len(self.series)} series"
                )
            for label, pair in zip(self.series, self.stats, strict=True):
                mean, standard_deviation = pair
                if not (np.isfinite(mean) and np.isfinite(standard_deviation)):
                    raise ValueError(f"stats[{label!r}] must be finite")
                if standard_deviation <= 0.0:
                    raise ValueError(
                        f"stats[{label!r}] standard deviation must be positive"
                    )

        if has_samples:
            self._validate_samples()
        else:
            self._validate_table()

    def _validate_samples(self) -> None:
        assert self.samples is not None
        if len(self.samples) != len(self.series):
            raise ValueError(
                f"samples has {len(self.samples)} rows but there are "
                f"{len(self.series)} series"
            )
        for label, row in zip(self.series, self.samples, strict=True):
            if len(row) < 2:
                raise ValueError(
                    f"series {label!r} has {len(row)} observation(s); a "
                    "distribution needs at least two"
                )
            if not np.all(np.isfinite(np.asarray(row, dtype=float))):
                raise ValueError(f"series {label!r} contains a non-finite observation")

    def _validate_table(self) -> None:
        if self.bin_edges is None or self.counts is None:
            raise ValueError("bin_edges and counts must be supplied together")
        edges = np.asarray(self.bin_edges, dtype=float)
        if edges.size < 2:
            raise ValueError("bin_edges needs at least two entries")
        if not np.all(np.isfinite(edges)):
            raise ValueError("bin_edges must be finite")
        if not np.all(np.diff(edges) > 0.0):
            raise ValueError("bin_edges must increase strictly")
        n_bins = int(edges.size) - 1
        if len(self.counts) != len(self.series):
            raise ValueError(
                f"counts has {len(self.counts)} rows but there are "
                f"{len(self.series)} series"
            )
        total = 0.0
        for label, row in zip(self.series, self.counts, strict=True):
            if len(row) != n_bins:
                raise ValueError(
                    f"counts[{label!r}] has {len(row)} entries but bin_edges "
                    f"defines {n_bins} bins"
                )
            values = np.asarray(row, dtype=float)
            if not np.all(np.isfinite(values)):
                raise ValueError(f"counts[{label!r}] must be finite")
            if np.any(values < 0.0):
                raise ValueError(f"counts[{label!r}] must not be negative")
            total += float(values.sum())
        if total <= 0.0:
            raise ValueError("counts must contain at least one positive bin")

    def _shared_edges(self) -> NDArray[np.float64]:
        """Bin edges every series is drawn against."""

        if self.bin_edges is not None:
            return np.asarray(self.bin_edges, dtype=float)
        assert self.samples is not None
        flat = np.concatenate(
            [np.asarray(row, dtype=float) for row in self.samples if row]
        )
        low = float(flat.min())
        high = float(flat.max())
        if high <= low:
            # A series of identical readings still deserves a visible bar.
            spread = abs(low) * 0.05 or 0.5
            low, high = low - spread, high + spread
        return np.linspace(low, high, self.bins + 1)

    def histogram(self, index: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Shared bin edges and this series' counts, binning samples on demand."""

        edges = self._shared_edges()
        if self.counts is not None:
            return edges, np.asarray(self.counts[index], dtype=float)
        assert self.samples is not None
        counts, _ = np.histogram(
            np.asarray(self.samples[index], dtype=float), bins=edges
        )
        return edges, counts.astype(float)

    def series_stats(self, index: int) -> tuple[float, float]:
        """``(mean, sd)`` for the fit curve and the legend entry.

        Published statistics win, then raw observations, and finally the
        count-weighted mean and standard deviation of the bin midpoints.
        """

        if self.stats is not None:
            mean, standard_deviation = self.stats[index]
            return float(mean), float(standard_deviation)
        if self.samples is not None:
            sample = np.asarray(self.samples[index], dtype=float)
            return float(sample.mean()), float(sample.std(ddof=0))
        edges, counts = self.histogram(index)
        midpoints = 0.5 * (edges[:-1] + edges[1:])
        total = float(counts.sum())
        if total <= 0.0:
            return float(midpoints.mean()), 0.0
        mean = float(np.dot(counts, midpoints) / total)
        variance = float(np.dot(counts, (midpoints - mean) ** 2) / total)
        return mean, float(np.sqrt(max(variance, 0.0)))


_REFERENCE_SERIES: Final[tuple[str, ...]] = (
    "Zhang et al.",
    "Kioumarsi et al.",
    "Xue et al.",
)

_REFERENCE_BIN_EDGES: Final[tuple[float, ...]] = (
    -106.8,
    -91.6,
    -76.3,
    -61.1,
    -45.8,
    -30.5,
    -15.3,
    0.0,
    15.3,
    30.5,
    45.8,
    61.1,
    76.3,
    91.6,
    106.8,
)

# Digitised from the visible bar heights in the first carousel frame.
_REFERENCE_COUNTS: Final[tuple[tuple[float, ...], ...]] = (
    (0, 0, 0, 0, 3, 15, 34, 21, 6, 1, 0, 0, 0, 0),
    (0, 0, 1, 2, 6, 8, 23, 23, 23, 10, 5, 1, 1, 0),
    (0, 0, 0, 0, 0, 11, 33, 23, 3, 0, 0, 0, 0, 0),
)

# Means and standard deviations as printed in the reference legend.
_REFERENCE_STATS: Final[tuple[tuple[float, float], ...]] = (
    (-4.9, 14.4),
    (7.6, 25.1),
    (-3.3, 10.6),
)

DEFAULT_DATA: Final[GroupedGradientHistData] = GroupedGradientHistData.from_bin_counts(
    series=_REFERENCE_SERIES,
    bin_edges=_REFERENCE_BIN_EDGES,
    counts=_REFERENCE_COUNTS,
    value_label=r"Residual Group ($\mu\epsilon$)",
    count_label="Frequency",
    stats=_REFERENCE_STATS,
)


@dataclass(frozen=True, slots=True)
class ChartStyle:
    """Layout tuned to the 2121 x 1762 reference image.

    Bar width and gap are expressed in bin units and shrink together so that a
    group of any size still fits inside ``group_span`` of one bin; the reference
    three series keep their original 0.26/0.04 spacing.  ``panel_label`` carries
    the reference figure's ``(a)`` marker and should be set to ``None`` for a
    stand-alone chart.
    """

    figure_size: tuple[float, float] = (8.484, 7.048)
    axes_bounds: tuple[float, float, float, float] = (0.118, 0.168, 0.842, 0.768)
    y_limits: tuple[float, float] | None = (0.0, 50.0)
    bar_width_fraction: float = 0.26
    bar_gap_fraction: float = 0.04
    group_span: float = 0.9
    x_margin: float = 0.75
    gradient_steps: int = 256
    fit_grid_points: int = 400
    bin_label_format: str = "[{low}, {high})"
    bin_label_decimals: int = 1
    """Starting precision for bin edges; raised until adjacent labels differ."""

    stat_decimals: int = 1
    """Starting precision for the legend's mean and sd; raised until both show."""

    max_tick_labels: int = 16
    panel_label: str | None = "(a)"
    line_width: float = 2.15
    spine_width: float = 1.55
    tick_length: float = 5.2
    minor_tick_length: float = 2.8
    label_font_size: float = 16.0
    tick_font_size: float = 10.0
    legend_font_size: float = 10.5
    panel_font_size: float = 20.0
    x_label_pad: float = 10.0
    y_label_pad: float = 8.0

    def validate(self, *, categories: int, series: int) -> None:
        if min(self.figure_size) <= 0:
            raise ValueError("figure_size must be positive")
        if self.y_limits is not None and self.y_limits[1] <= self.y_limits[0]:
            raise ValueError("y_limits must be increasing")
        if not 0 < self.bar_width_fraction <= 1.0:
            raise ValueError("bar_width_fraction must sit in (0, 1]")
        if self.bar_gap_fraction < 0:
            raise ValueError("bar_gap_fraction must be non-negative")
        if not 0.1 <= self.group_span <= 1.0:
            raise ValueError("group_span must sit in [0.1, 1.0]")
        if self.gradient_steps < 16:
            raise ValueError("gradient_steps must be at least 16")
        if self.fit_grid_points < 16:
            raise ValueError("fit_grid_points must be at least 16")
        if self.max_tick_labels < 1:
            raise ValueError("max_tick_labels must be positive")
        if self.bin_label_decimals < 0 or self.stat_decimals < 0:
            raise ValueError("decimal counts must be non-negative")
        if categories < 1:
            raise ValueError("need at least one bin")
        width, _ = bar_geometry(self, series=series)
        if width <= 0:
            raise ValueError(f"{series} series do not fit inside one bin")
        try:
            self.bin_label_format.format(low="0.0", high="1.0")
        except (IndexError, KeyError, ValueError) as exc:
            raise ValueError(
                f"bin_label_format {self.bin_label_format!r} is not usable"
            ) from exc


DEFAULT_STYLE: Final[ChartStyle] = ChartStyle()


def bar_geometry(style: ChartStyle, *, series: int) -> tuple[float, float]:
    """Return ``(bar_width, bar_gap)`` in bin units for ``series`` bars."""

    if series < 1:
        raise ValueError("series must be positive")
    width = style.bar_width_fraction
    gap = style.bar_gap_fraction if series > 1 else 0.0
    occupied = series * width + (series - 1) * gap
    if occupied > style.group_span:
        scale = style.group_span / occupied
        width *= scale
        gap *= scale
    return width, gap


_MAX_DECIMALS: Final[int] = 6


def _stat_decimals(values: Sequence[float], minimum: int) -> int:
    """Decimals before the smallest non-zero magnitude stops printing as ``0``.

    A precision tuned to residual microstrain would flatten a column of
    millimetre errors to ``0.0``, so the pinned precision is a floor rather than
    a fixed choice.
    """

    decimals = max(minimum, 0)
    finite = [abs(float(value)) for value in values if np.isfinite(value) and value]
    if not finite:
        return decimals
    smallest = min(finite)
    while decimals < _MAX_DECIMALS and round(smallest, decimals) == 0.0:
        decimals += 1
    return decimals


def _edge_decimals(edges: NDArray[np.float64], minimum: int) -> int:
    """Decimals at which no two bin edges render as the same number."""

    decimals = max(minimum, 0)
    while decimals < _MAX_DECIMALS:
        rendered = {f"{float(edge):.{decimals}f}" for edge in edges}
        if len(rendered) == edges.size:
            break
        decimals += 1
    return decimals


def format_legend_label(
    series: str, mean: float, standard_deviation: float, decimals: int = 1
) -> str:
    return (
        f"{series} ($\\mu = {mean:.{decimals}f}$, "
        f"$\\sigma = {standard_deviation:.{decimals}f}$)"
    )


def _gradient_cmap(color: str) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "bar_gradient",
        [(0.0, "#FFFFFF"), (0.18, "#FFFFFF"), (1.0, color)],
        N=256,
    )


def _draw_gradient_bar(
    ax: Axes,
    x: float,
    height: float,
    width: float,
    color: str,
    style: ChartStyle,
) -> None:
    if height <= 0:
        return
    gradient = np.linspace(0.0, 1.0, style.gradient_steps).reshape(-1, 1)
    image = ax.imshow(
        gradient,
        extent=(x - 0.5 * width, x + 0.5 * width, 0.0, height),
        origin="lower",
        aspect="auto",
        cmap=_gradient_cmap(color),
        interpolation="bicubic",
        zorder=2,
        clip_on=True,
    )
    image.set_clip_path(ax.patch)
    ax.add_patch(
        Rectangle(
            (x - 0.5 * width, 0.0),
            width,
            height,
            fill=False,
            edgecolor=color,
            linewidth=0.35,
            zorder=3,
        )
    )


def peak_height(data: GroupedGradientHistData, style: ChartStyle) -> float:
    """Tallest thing the count axis has to hold: a bar or a fit curve apex."""

    tallest = 0.0
    for index in range(len(data.series)):
        edges, counts = data.histogram(index)
        tallest = max(tallest, float(counts.max(initial=0.0)))
        if not data.show_normal_fit:
            continue
        _, standard_deviation = data.series_stats(index)
        if standard_deviation <= 0.0:
            continue
        bin_width = float(np.mean(np.diff(edges)))
        apex = (
            float(counts.sum())
            * bin_width
            / (standard_deviation * float(np.sqrt(2.0 * np.pi)))
        )
        tallest = max(tallest, apex)
    return tallest


def _style_axis(
    ax: Axes,
    data: GroupedGradientHistData,
    style: ChartStyle,
    y_limits: tuple[float, float],
) -> None:
    ax.set_ylim(*y_limits)
    major = nice_step(y_limits[1] - y_limits[0])
    ax.yaxis.set_major_locator(MultipleLocator(major))
    ax.yaxis.set_minor_locator(MultipleLocator(major / 5.0))
    ax.tick_params(
        which="major",
        direction="in",
        top=True,
        right=True,
        length=style.tick_length,
        width=style.spine_width,
        labelsize=style.tick_font_size,
        pad=4.0,
    )
    ax.tick_params(
        which="minor",
        direction="in",
        top=True,
        right=True,
        left=True,
        length=style.minor_tick_length,
        width=style.spine_width * 0.75,
    )
    ax.set_xlabel(
        data.value_label,
        fontsize=style.label_font_size,
        fontweight="bold",
        labelpad=style.x_label_pad,
    )
    ax.set_ylabel(
        data.count_label,
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
    data: GroupedGradientHistData,
    palette: Palette,
    style: ChartStyle,
) -> None:
    n_series = len(data.series)
    width, gap = bar_geometry(style, series=n_series)
    colors = palette.take(n_series)
    for index in range(n_series):
        _, counts = data.histogram(index)
        offset = (index - 0.5 * (n_series - 1)) * (width + gap)
        centers = np.arange(counts.size, dtype=float) + offset
        for x, height in zip(centers, counts, strict=True):
            _draw_gradient_bar(ax, float(x), float(height), width, colors[index], style)


def _draw_fits(
    ax: Axes,
    data: GroupedGradientHistData,
    palette: Palette,
    style: ChartStyle,
) -> None:
    edges, _ = data.histogram(0)
    bin_width = float(np.mean(np.diff(edges)))
    value_grid = np.linspace(
        float(edges[0]), float(edges[-1]), style.fit_grid_points
    )
    # Bars sit on integer bin indices, so the fit is drawn in edge-index space.
    axis_grid = np.interp(value_grid, edges, np.arange(edges.size, dtype=float))
    colors = palette.take(len(data.series))
    for index in range(len(data.series)):
        _, counts = data.histogram(index)
        mean, standard_deviation = data.series_stats(index)
        if standard_deviation <= 0.0:
            continue
        density = norm.pdf(value_grid, loc=mean, scale=standard_deviation)
        ax.plot(
            axis_grid,
            density * float(counts.sum()) * bin_width,
            linestyle=(0, (4.6, 2.8)),
            color=colors[index],
            linewidth=style.line_width,
            solid_capstyle="round",
            dash_capstyle="round",
            zorder=4,
        )


def _draw_legend(
    ax: Axes,
    data: GroupedGradientHistData,
    palette: Palette,
    style: ChartStyle,
) -> None:
    colors = palette.take(len(data.series))
    summaries = [data.series_stats(index) for index in range(len(data.series))]
    decimals = _stat_decimals(
        [value for pair in summaries for value in pair], style.stat_decimals
    )
    handles: list[Patch | Line2D] = []
    labels: list[str] = []
    for index, name in enumerate(data.series):
        mean, standard_deviation = summaries[index]
        handles.append(Patch(facecolor=colors[index], edgecolor="none"))
        labels.append(format_legend_label(name, mean, standard_deviation, decimals))
    if data.show_normal_fit:
        handles.append(
            Line2D(
                [0],
                [0],
                color="#7A7A7A",
                linestyle=(0, (4.6, 2.8)),
                linewidth=style.line_width,
            )
        )
        labels.append("Normal fits")
    legend = ax.legend(
        handles,
        labels,
        loc="upper left",
        frameon=True,
        fancybox=False,
        edgecolor="black",
        facecolor="white",
        framealpha=1.0,
        borderpad=0.55,
        labelspacing=0.45,
        handlelength=1.35,
        handletextpad=0.55,
        fontsize=style.legend_font_size,
    )
    legend.get_frame().set_linewidth(0.9)
    for text in legend.get_texts():
        text.set_fontweight("bold")


def _draw_bin_ticks(
    ax: Axes,
    edges: NDArray[np.float64],
    style: ChartStyle,
) -> None:
    n_bins = int(edges.size) - 1
    # Thin the labels rather than let 40 rotated intervals collide.
    step = max(1, -(-n_bins // style.max_tick_labels))
    positions = np.arange(0, n_bins, step, dtype=float)
    # Narrow bins need more decimals than the reference before edges separate.
    decimals = _edge_decimals(edges, style.bin_label_decimals)
    labels = [
        style.bin_label_format.format(
            low=f"{float(edges[int(index)]):.{decimals}f}",
            high=f"{float(edges[int(index) + 1]):.{decimals}f}",
        )
        for index in positions
    ]
    ax.set_xlim(-style.x_margin, n_bins - 1 + style.x_margin)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor")
    for label in ax.get_xticklabels():
        label.set_fontweight("bold")
        label.set_fontsize(style.tick_font_size)


def create_figure(
    palette: Palette = PALETTES[0],
    data: GroupedGradientHistData = DEFAULT_DATA,
    style: ChartStyle = DEFAULT_STYLE,
) -> Figure:
    """Create the grouped gradient histogram without writing it to disk."""

    data.validate()
    edges, _ = data.histogram(0)
    style.validate(categories=int(edges.size) - 1, series=len(data.series))
    y_limits = resolve_limits(style.y_limits, 0.0, peak_height(data, style))

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
        ax = figure.add_axes(style.axes_bounds)
        ax.set_facecolor("white")
        _style_axis(ax, data, style, y_limits)
        _draw_bars(ax, data, palette, style)
        if data.show_normal_fit:
            _draw_fits(ax, data, palette, style)
        _draw_legend(ax, data, palette, style)
        _draw_bin_ticks(ax, edges, style)

        if style.panel_label:
            ax.text(
                0.985,
                0.965,
                style.panel_label,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=style.panel_font_size,
                fontweight="bold",
                zorder=6,
            )

    return figure


def main(argv: Sequence[str] | None = None) -> int:
    return run_cli(sys.modules[__name__], argv)


if __name__ == "__main__":
    raise SystemExit(main())
