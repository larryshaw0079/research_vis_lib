"""Grouped annular bar chart: one sector per category, one bar per series.

``DEFAULT_DATA`` holds horizon-720 MSE digitised from a Xiaohongshu carousel that
follows Table 2 of Ma et al., "MoFo: Empowering Long-term Time Series Forecasting
with Periodic Pattern Modeling", NeurIPS 2025
(https://openreview.net/forum?id=sbvLts2HqR), with a few entries that match the
published figure rather than the camera-ready table.

The reference figure encodes bar length as an inverted, within-category min-max
rescaling, so a bar's length is *not* proportional to its value.  That trick is
preserved for the reference data through ``length_scale="within-category"``, but
new datasets default to ``length_scale="absolute"`` where length is proportional
to the value and the radial axis can be read directly.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from typing import Final, Literal, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Arc, FancyBboxPatch, Rectangle, Wedge
from numpy.typing import NDArray

from ..contract import DataKind, Extent, Feature, Geometry, TemplateSpec
from ..palettes import Palette, readable_text_color
from ..render import run_cli

SPEC: Final[TemplateSpec] = TemplateSpec(
    template_id="grouped-ring-bar",
    title="Grouped annular bar chart",
    summary=(
        "One angular sector per category, one coloured bar per series inside it, "
        "with the value printed along each bar and a legend in the middle."
    ),
    kinds=(DataKind.MATRIX,),
    geometry=Geometry.POLAR,
    categories=Extent(2, 16),
    series=Extent(2, 10),
    builder="GroupedRingBarData.from_matrix",
    data_contract=(
        "A numeric value for every (category, series) pair. Values need no common "
        "total. Optionally flag one series to highlight and declare whether lower "
        "values are better."
    ),
    good_for=(
        "benchmark tables comparing a handful of methods across many datasets",
        "compact side-by-side comparison where a wide grouped bar chart would not fit",
        "highlighting one method against baselines",
    ),
    avoid_when=(
        "the category axis is ordered, such as time, since angle hides order",
        "values span orders of magnitude, because radial length compresses them",
        "more than about 16 categories or 10 series, where sectors get too thin",
    ),
    long_category_labels=False,
    affinities=(
        (Feature.HAS_EMPHASIS, 7.0),
        (Feature.MANY_CATEGORIES, -5.0),
        (Feature.LONG_LABELS, -6.0),
        (Feature.WIDE_DYNAMIC_RANGE, -9.0),
    ),
    default_dpi=250,
    reference="Ma et al., MoFo, NeurIPS 2025, Table 2 (L=720 MSE).",
)


LengthScale = Literal["absolute", "global", "within-category"]

_LENGTH_SCALES: Final[frozenset[str]] = frozenset(
    {"absolute", "global", "within-category"}
)


# Colours sampled from the legend swatches in carousel images 1-18.
PALETTES: Final[tuple[Palette, ...]] = (
    Palette("rose-steel-indigo", ("#D9B4A7", "#B0C4DE", "#FDEBD0", "#C8E6C9", "#5E4FA2")),
    Palette("teal-coral", ("#006D77", "#83C5BE", "#FFDDD2", "#E29578", "#FF4D6D")),
    Palette("nighthawk", ("#0B0C10", "#1F2833", "#C5C6C7", "#66FCF1", "#45A29E")),
    Palette("amber-ember", ("#F9A825", "#FBC02D", "#FD831F", "#E65100", "#3E2723")),
    Palette("olive-coral", ("#556B2F", "#8B8589", "#D2691E", "#FF7F50", "#F5F5DC")),
    Palette("neon-cyber", ("#18003C", "#0014FF", "#FE00FE", "#00FF87", "#FF0055")),
    Palette("earthy-cottage", ("#283618", "#606C38", "#FEFAE0", "#DDA15E", "#BC6C25")),
    Palette("peach-blush", ("#FFB5A7", "#FCD5CE", "#F8EDEB", "#F9DCC4", "#FEC89A")),
    Palette("tailwind-blue", ("#0F172A", "#1E293B", "#3B82F6", "#93C5FD", "#F8FAFC")),
    Palette("space-cadet", ("#1D3557", "#457B9D", "#A8DADC", "#F1FAEE", "#E63946")),
    Palette("wine-ember", ("#2C0E37", "#5B1E31", "#8B2635", "#A23B72", "#F78154")),
    Palette("pastel-candy", ("#E8AEB7", "#B8E1DD", "#A997DF", "#F5D6B8", "#ECE4B7")),
    Palette("coffee-terracotta", ("#1A120B", "#3C2A21", "#D5CEA3", "#E5BA73", "#C84B31")),
    Palette("ocean-cyan", ("#071E22", "#1D7874", "#07B1CA", "#3581B8", "#ECE4B7")),
    Palette("dark-cyan-orange", ("#222831", "#393E46", "#00ADB5", "#EEEEEE", "#FF5722")),
    Palette("rose-pink", ("#FF758F", "#FF8FA3", "#FFB3C1", "#FFCCD5", "#C9184A")),
    Palette("northern-ice", ("#3D5A80", "#98C1D9", "#E0FBFC", "#EE6C4D", "#293241")),
    Palette("dusk-sunset", ("#355C7D", "#6C5B7B", "#C06C84", "#F67280", "#F8B195")),
)


@dataclass(frozen=True, slots=True)
class GroupedRingBarData:
    """A ``categories x series`` value matrix plus its presentation metadata."""

    categories: tuple[str, ...]
    series: tuple[str, ...]
    values: tuple[tuple[float, ...], ...]
    value_label: str = "Value"
    value_format: str = "{:.3f}"
    lower_is_better: bool = False
    length_scale: LengthScale = "absolute"
    highlight: str | None = None

    @classmethod
    def from_matrix(
        cls,
        *,
        categories: Sequence[str],
        series: Sequence[str],
        values: Sequence[Sequence[float | None]],
        value_label: str = "Value",
        value_format: str = "{:.3f}",
        lower_is_better: bool = False,
        length_scale: LengthScale = "absolute",
        highlight: str | None = None,
    ) -> "GroupedRingBarData":
        """Build from a ``[category][series]`` matrix of measurements.

        ``None`` entries become NaN and are skipped when drawing, so a partially
        filled benchmark table does not have to be padded with zeros.
        """

        rows = tuple(
            tuple(float("nan") if value is None else float(value) for value in row)
            for row in values
        )
        built = cls(
            categories=tuple(str(name) for name in categories),
            series=tuple(str(name) for name in series),
            values=rows,
            value_label=value_label,
            value_format=value_format,
            lower_is_better=bool(lower_is_better),
            length_scale=length_scale,
            highlight=highlight,
        )
        built.validate()
        return built

    @classmethod
    def from_mapping(
        cls,
        *,
        categories: Sequence[str],
        series: Sequence[str],
        values: Mapping[str, Mapping[str, float | None]],
        **kwargs: object,
    ) -> "GroupedRingBarData":
        """Build from nested ``values[category][series]`` mappings."""

        matrix = [
            [values.get(category, {}).get(name) for name in series]
            for category in categories
        ]
        return cls.from_matrix(
            categories=categories, series=series, values=matrix, **kwargs  # type: ignore[arg-type]
        )

    def validate(self) -> None:
        if len(self.categories) < 2:
            raise ValueError("need at least two categories")
        if len(self.series) < 2:
            raise ValueError("need at least two series")
        if len(set(self.categories)) != len(self.categories):
            raise ValueError("category labels must be unique")
        if len(set(self.series)) != len(self.series):
            raise ValueError("series labels must be unique")
        if len(self.values) != len(self.categories):
            raise ValueError(
                f"values has {len(self.values)} rows but there are "
                f"{len(self.categories)} categories"
            )
        for category, row in zip(self.categories, self.values, strict=True):
            if len(row) != len(self.series):
                raise ValueError(
                    f"values[{category!r}] has {len(row)} entries but there are "
                    f"{len(self.series)} series"
                )
        if self.length_scale not in _LENGTH_SCALES:
            raise ValueError(
                f"length_scale must be one of {sorted(_LENGTH_SCALES)}, "
                f"got {self.length_scale!r}"
            )
        if self.highlight is not None and self.highlight not in self.series:
            raise ValueError(f"highlight {self.highlight!r} is not one of the series")
        finite = self.matrix()[np.isfinite(self.matrix())]
        if finite.size == 0:
            raise ValueError("values must contain at least one finite measurement")
        try:
            self.value_format.format(1.0)
        except (IndexError, KeyError, ValueError) as exc:
            raise ValueError(f"value_format {self.value_format!r} is not usable") from exc

    def matrix(self) -> NDArray[np.float64]:
        """Values as a ``(categories, series)`` array."""

        return np.asarray(self.values, dtype=float)

    def row(self, category: str) -> NDArray[np.float64]:
        return self.matrix()[self.categories.index(category)]

    def value_range(self) -> tuple[float, float]:
        finite = self.matrix()
        finite = finite[np.isfinite(finite)]
        return float(finite.min()), float(finite.max())

    def best_series(self) -> str:
        """Series with the most category-level wins, honouring direction."""

        matrix = self.matrix()
        wins = np.zeros(len(self.series), dtype=int)
        for row in matrix:
            if not np.any(np.isfinite(row)):
                continue
            ordered = np.where(np.isfinite(row), row, np.inf if self.lower_is_better else -np.inf)
            index = int(np.argmin(ordered) if self.lower_is_better else np.argmax(ordered))
            wins[index] += 1
        return self.series[int(np.argmax(wins))]


_REFERENCE_CATEGORIES: Final[tuple[str, ...]] = (
    "Traffic",
    "ETTh1",
    "ETTh2",
    "ETTm1",
    "ETTm2",
    "Weather",
    "Electricity",
    "Solar",
)

_REFERENCE_SERIES: Final[tuple[str, ...]] = (
    "iTransformer",
    "Pathformer",
    "PatchTST",
    "PDF",
    "MoFo (Ours)",
)

# Horizon-720 MSE digitised from the labelled bars in the reference carousel.
_REFERENCE_MSE: Final[tuple[tuple[float, ...], ...]] = (
    (0.445, 0.452, 0.435, 0.438, 0.424),
    (0.495, 0.450, 0.457, 0.456, 0.447),
    (0.424, 0.413, 0.406, 0.398, 0.379),
    (0.429, 0.428, 0.416, 0.408, 0.388),
    (0.375, 0.361, 0.362, 0.349, 0.342),
    (0.320, 0.318, 0.312, 0.323, 0.312),
    (0.214, 0.211, 0.214, 0.199, 0.191),
    (0.223, 0.208, 0.215, 0.212, 0.193),
)

DEFAULT_DATA: Final[GroupedRingBarData] = GroupedRingBarData.from_matrix(
    categories=_REFERENCE_CATEGORIES,
    series=_REFERENCE_SERIES,
    values=_REFERENCE_MSE,
    value_label="MSE",
    value_format="{:.3f}",
    lower_is_better=True,
    length_scale="within-category",
    highlight="MoFo (Ours)",
)


@dataclass(frozen=True, slots=True)
class ChartStyle:
    """Geometry and typography tuned to the 2601 x 2601 reference image.

    Angular geometry is derived from the data: each category owns
    ``360 / n_categories`` degrees, of which ``sector_fill`` is covered by bars.
    With the reference eight categories and five series this reproduces the
    original 45-degree sectors and 6.2-degree bars exactly.
    """

    figure_size: tuple[float, float] = (10.404, 10.404)
    x_limits: tuple[float, float] = (-1300.0, 1300.0)
    y_limits: tuple[float, float] = (-1300.0, 1300.0)
    sector_fill: float = 35.0 / 45.0
    bar_gap_fraction: float = 1.0 / 45.0
    sector_start_fraction: float = 0.5 / 45.0
    inner_radius: float = 447.5
    min_outer_radius: float = 671.2
    max_outer_radius: float = 1231.1
    arc_radius: float = 436.0
    arc_width: float = 3.0
    label_radius: float = 366.0
    value_inset: float = 88.0
    category_font_size: float = 22.0
    value_font_size: float = 17.0
    legend_font_size: float = 15.0
    legend_row_pitch: float = 77.0
    legend_swatch_size: tuple[float, float] = (101.0, 42.0)
    legend_min_width: float = 500.0
    highlight_color: str = "#C00000"
    show_values: bool = True
    value_color: str | None = None
    """Fixed ink for the in-bar value labels; ``None`` picks per-bar contrast."""

    def validate(self, *, categories: int, series: int) -> None:
        if min(self.figure_size) <= 0:
            raise ValueError("figure_size must be positive")
        if not 0.2 <= self.sector_fill <= 1.0:
            raise ValueError("sector_fill must sit in [0.2, 1.0]")
        if not 0 < self.inner_radius < self.min_outer_radius < self.max_outer_radius:
            raise ValueError("radii must satisfy 0 < inner < min_outer < max_outer")
        if self.value_inset <= 0:
            raise ValueError("value_inset must be positive")
        width, gap, _ = bar_geometry(self, categories=categories, series=series)
        if width <= 0:
            raise ValueError(
                f"{series} series do not fit in a {360.0 / categories:.1f}-degree "
                "sector; lower bar_gap_fraction or raise sector_fill"
            )
        if gap < 0:
            raise ValueError("bar gap must be non-negative")


DEFAULT_STYLE: Final[ChartStyle] = ChartStyle()


def bar_geometry(
    style: ChartStyle, *, categories: int, series: int
) -> tuple[float, float, float]:
    """Return ``(bar_width, bar_gap, sector_start_offset)`` in degrees."""

    if categories < 1 or series < 1:
        raise ValueError("categories and series must be positive")
    sector = 360.0 / categories
    occupied = style.sector_fill * sector
    gap = style.bar_gap_fraction * sector
    if series > 1:
        # Keep bars at least four times as wide as the gaps between them.
        max_gap = occupied / (5.0 * (series - 1))
        gap = min(gap, max_gap)
        width = (occupied - (series - 1) * gap) / series
    else:
        gap = 0.0
        width = occupied
    return width, gap, style.sector_start_fraction * sector


def _xy(
    angle_degrees: float | NDArray[np.float64],
    radius: float | NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert top-origin clockwise polar coordinates to Cartesian coordinates."""

    angle = np.radians(angle_degrees)
    radial = np.asarray(radius, dtype=float)
    return radial * np.sin(angle), radial * np.cos(angle)


def _wedge_thetas(start_degrees: float, width_degrees: float) -> tuple[float, float]:
    """Matplotlib wedge angles (CCW from +x) for a clockwise bar."""

    end_degrees = start_degrees + width_degrees
    return 90.0 - end_degrees, 90.0 - start_degrees


def outer_radii(
    row: NDArray[np.float64],
    data: GroupedRingBarData,
    style: ChartStyle,
) -> NDArray[np.float64]:
    """Map one category's values to outer radii under the chosen length scale."""

    values = np.asarray(row, dtype=float)
    midpoint = 0.5 * (style.min_outer_radius + style.max_outer_radius)
    result = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        return result

    if data.length_scale == "absolute":
        low, high = data.value_range()
        baseline = min(0.0, low)
        span = high - baseline
        if np.isclose(span, 0.0):
            result[finite] = midpoint
            return result
        span_radius = style.max_outer_radius - style.inner_radius
        fraction = (values[finite] - baseline) / span
        if data.lower_is_better:
            fraction = 1.0 - fraction
        result[finite] = style.inner_radius + span_radius * fraction
        # Keep every bar visible even when a value sits at the baseline.
        result[finite] = np.maximum(result[finite], style.inner_radius + 0.02 * span_radius)
        return result

    if data.length_scale == "global":
        low, high = data.value_range()
    else:
        low = float(np.nanmin(values))
        high = float(np.nanmax(values))

    if np.isclose(high, low):
        result[finite] = midpoint
        return result
    span_radius = style.max_outer_radius - style.min_outer_radius
    fraction = (values[finite] - low) / (high - low)
    if data.lower_is_better:
        fraction = 1.0 - fraction
    result[finite] = style.min_outer_radius + span_radius * fraction
    return result


def _sector_starts(
    style: ChartStyle, *, categories: int, series: int
) -> NDArray[np.float64]:
    width, gap, offset = bar_geometry(style, categories=categories, series=series)
    sector = 360.0 / categories
    index = np.arange(series, dtype=float)
    return np.stack(
        [
            group * sector + offset + index * (width + gap)
            for group in range(categories)
        ]
    )


def _draw_bars(
    ax: Axes,
    data: GroupedRingBarData,
    palette: Palette,
    style: ChartStyle,
) -> None:
    n_categories = len(data.categories)
    n_series = len(data.series)
    width, _, _ = bar_geometry(style, categories=n_categories, series=n_series)
    starts = _sector_starts(style, categories=n_categories, series=n_series)
    colors = palette.take(n_series)

    for category_index, category in enumerate(data.categories):
        row = data.row(category)
        radii = outer_radii(row, data, style)
        for series_index in range(n_series):
            outer = radii[series_index]
            if not np.isfinite(outer):
                continue
            start = float(starts[category_index, series_index])
            theta1, theta2 = _wedge_thetas(start, width)
            color = colors[series_index]
            ax.add_patch(
                Wedge(
                    (0.0, 0.0),
                    float(outer),
                    theta1,
                    theta2,
                    width=float(outer) - style.inner_radius,
                    facecolor=color,
                    edgecolor=color,
                    linewidth=0.15,
                    zorder=3,
                )
            )
            if not style.show_values:
                continue
            text = data.value_format.format(float(row[series_index]))
            bar_length = float(outer) - style.inner_radius
            # A short bar cannot hold its own label, so park it past the tip.
            needed = 0.62 * style.value_font_size * len(text)
            if bar_length < needed + style.value_inset:
                label_radius = float(outer) + 0.5 * needed + 8.0
                ink = style.value_color or "#111111"
            else:
                label_radius = min(
                    float(outer) - style.value_inset,
                    style.inner_radius + 0.62 * bar_length,
                )
                label_radius = max(label_radius, style.inner_radius + 0.42 * bar_length)
                ink = style.value_color or readable_text_color(color)
            mid_angle = start + 0.5 * width
            x, y = _xy(mid_angle, label_radius)
            ax.text(
                float(x),
                float(y),
                text,
                ha="center",
                va="center",
                rotation=90.0 - mid_angle,
                rotation_mode="anchor",
                fontsize=style.value_font_size,
                color=ink,
                fontweight="bold",
                zorder=6,
            )


def _draw_sector_arcs(
    ax: Axes, data: GroupedRingBarData, style: ChartStyle
) -> None:
    n_categories = len(data.categories)
    n_series = len(data.series)
    width, gap, offset = bar_geometry(style, categories=n_categories, series=n_series)
    occupied = n_series * width + (n_series - 1) * gap
    sector = 360.0 / n_categories
    for group in range(n_categories):
        theta1, theta2 = _wedge_thetas(group * sector + offset, occupied)
        ax.add_patch(
            Arc(
                (0.0, 0.0),
                2.0 * style.arc_radius,
                2.0 * style.arc_radius,
                theta1=theta1,
                theta2=theta2,
                color="black",
                linewidth=style.arc_width,
                capstyle="butt",
                zorder=5,
            )
        )


def _draw_category_labels(
    ax: Axes, data: GroupedRingBarData, style: ChartStyle
) -> None:
    n_categories = len(data.categories)
    n_series = len(data.series)
    width, gap, offset = bar_geometry(style, categories=n_categories, series=n_series)
    occupied = n_series * width + (n_series - 1) * gap
    sector = 360.0 / n_categories

    for index, name in enumerate(data.categories):
        start = index * sector + offset
        end = start + occupied
        pad = 0.07 * (end - start)
        span = (end - pad) - (start + pad)
        # Shrink the font when a long label would otherwise overrun its arc.
        arc_length = np.radians(span) * style.label_radius
        per_character = arc_length / max(len(name), 1)
        font_size = min(style.category_font_size, max(8.0, per_character * 1.35))
        angles = (
            np.linspace(start + pad, end - pad, len(name))
            if len(name) > 1
            else np.array([0.5 * (start + end)])
        )
        for character, angle in zip(name, angles, strict=True):
            x, y = _xy(float(angle), style.label_radius)
            ax.text(
                float(x),
                float(y),
                character,
                ha="center",
                va="baseline",
                rotation=-float(angle),
                rotation_mode="anchor",
                fontsize=font_size,
                fontweight="bold",
                color="black",
                zorder=7,
            )


def _draw_legend(
    ax: Axes,
    data: GroupedRingBarData,
    palette: Palette,
    style: ChartStyle,
) -> None:
    n_series = len(data.series)
    colors = palette.take(n_series)
    swatch_width, swatch_height = style.legend_swatch_size
    pitch = style.legend_row_pitch
    longest = max(len(name) for name in data.series)
    box_width = max(
        style.legend_min_width,
        swatch_width + 60.0 + 0.62 * style.legend_font_size * longest,
    )
    box_height = n_series * pitch + 70.0
    ax.add_patch(
        FancyBboxPatch(
            (-0.50 * box_width, -0.50 * box_height),
            box_width,
            box_height,
            boxstyle="round,pad=0.0,rounding_size=16",
            facecolor="white",
            edgecolor="#C8C8C8",
            linewidth=1.4,
            zorder=20,
        )
    )

    swatch_x = -0.50 * box_width + 30.0
    text_x = swatch_x + swatch_width + 19.0
    first_y = 0.5 * (n_series - 1) * pitch + 6.5
    for index, name in enumerate(data.series):
        y = first_y - index * pitch
        ax.add_patch(
            Rectangle(
                (swatch_x, y - 0.5 * swatch_height),
                swatch_width,
                swatch_height,
                facecolor=colors[index],
                edgecolor="none",
                zorder=21,
            )
        )
        highlighted = name == data.highlight
        ax.text(
            text_x,
            y,
            name,
            ha="left",
            va="center",
            fontsize=style.legend_font_size,
            fontweight="bold" if highlighted else "normal",
            color=style.highlight_color if highlighted else "black",
            zorder=22,
        )


def create_figure(
    palette: Palette = PALETTES[0],
    data: GroupedRingBarData = DEFAULT_DATA,
    style: ChartStyle = DEFAULT_STYLE,
) -> Figure:
    """Create the grouped annular bar figure without writing it to disk."""

    data.validate()
    n_categories = len(data.categories)
    n_series = len(data.series)
    try:
        style.validate(categories=n_categories, series=n_series)
    except ValueError:
        # Auto-relax the angular packing rather than refusing a denser dataset.
        style = replace(style, sector_fill=min(0.94, style.sector_fill * 1.15))
        style.validate(categories=n_categories, series=n_series)

    with plt.rc_context(
        {
            "font.family": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
            "font.size": style.value_font_size,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    ):
        figure = plt.figure(figsize=style.figure_size, facecolor="white")
        ax = figure.add_axes((0.0, 0.0, 1.0, 1.0))
        ax.set_xlim(*style.x_limits)
        ax.set_ylim(*style.y_limits)
        ax.set_aspect("equal", adjustable="box")
        ax.set_axis_off()

        _draw_sector_arcs(ax, data, style)
        _draw_bars(ax, data, palette, style)
        _draw_category_labels(ax, data, style)
        _draw_legend(ax, data, palette, style)

    return figure


def main(argv: Sequence[str] | None = None) -> int:
    return run_cli(sys.modules[__name__], argv)


if __name__ == "__main__":
    raise SystemExit(main())
