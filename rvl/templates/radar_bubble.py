"""Annular radar with a sized bubble per series on its own ring.

Two encodings share one angular axis: a smoothed filled profile per series
carries the primary measurement, and a bubble on a per-series ring carries an
optional second measurement as area.

``DEFAULT_DATA`` is digitised from a Xiaohongshu carousel that publishes the
chart but not its table, so curve radii and bubble sizes were read off the
image.  Every radius, angle and bubble size below is derived from the data
counts, so the same renderer draws any matrix of the declared shape.

The renderer works in Cartesian coordinates rather than on a polar axis so the
annular mask, ticks and the two legends can be positioned precisely.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Final, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle
from numpy.typing import NDArray

from ..contract import DataKind, Extent, Feature, Geometry, TemplateSpec
from ..palettes import Palette
from ..render import nice_step, padded_range, resolve_limits, run_cli

SPEC: Final[TemplateSpec] = TemplateSpec(
    template_id="radar-bubble",
    title="Annular radar with sized bubbles",
    summary=(
        "Smoothed filled radial profiles for a few series across many categories, "
        "with a bubble per series on its own ring whose area encodes a second "
        "measurement."
    ),
    kinds=(DataKind.MATRIX,),
    geometry=Geometry.CIRCULAR,
    categories=Extent(6, 48),
    series=Extent(1, 5),
    builder="RadarBubbleData.from_matrix",
    data_contract=(
        "A primary value for every (category, series) pair, plus an optional "
        "second value per pair drawn as bubble area. Categories are many; series "
        "are few, typically snapshots in time."
    ),
    good_for=(
        "one measurement across many regions, compared over a few time points",
        "pairing a magnitude with a related size measure on the same axis",
        "many categories, where a grouped bar chart would run off the page",
    ),
    avoid_when=(
        "the categories are ordered and that order must be read",
        "there are fewer than six categories, where the ring looks sparse",
        "precise value comparison matters, since radial area is hard to judge",
    ),
    long_category_labels=False,
    affinities=(
        (Feature.MANY_CATEGORIES, 10.0),
        (Feature.MANY_SERIES, -12.0),
        (Feature.LONG_LABELS, -7.0),
        (Feature.WIDE_DYNAMIC_RANGE, -9.0),
    ),
    default_dpi=200,
    reference=(
        "Digitised from a Xiaohongshu carousel; curve radii and bubble sizes "
        "were read off the image."
    ),
)


# Colours sampled from the solid legend swatches in carousel images 1-18.
PALETTES: Final[tuple[Palette, ...]] = (
    Palette("slate-coral-apricot", ("#314666", "#E84B57", "#F2AA73")),
    Palette("apricot-teal-sand", ("#F4AA71", "#3EA497", "#EACB78")),
    Palette("crimson-orange-gold", ("#D73D3E", "#F68A19", "#FBC65D")),
    Palette("violet-magenta-cyan", ("#4B21AB", "#F53890", "#5DCDEF")),
    Palette("orchid-mint-gold", ("#BA2DA6", "#54B296", "#F8CB61")),
    Palette("gold-olive-tangerine", ("#F7CC62", "#9AC47B", "#F37E41")),
    Palette("steel-amber-coral", ("#609999", "#F8A034", "#F75255")),
    Palette("forest-neon-hot-pink", ("#185B36", "#FEFE51", "#FE2B62")),
    Palette("ocean-coral-sage", ("#3B88A8", "#F85255", "#9AC47B")),
    Palette("cyan-cream-salmon", ("#1CB5BD", "#FFDDC2", "#F07E75")),
    Palette("periwinkle-rose-butter", ("#6C6FD2", "#FD5D7A", "#FED576")),
    Palette("midnight-amber-ice", ("#1A1A6C", "#FFA733", "#CFF3F1")),
    Palette("charcoal-bluegray-crimson", ("#3D4053", "#97A1B7", "#EE364D")),
    Palette("crimson-snow-charcoal", ("#DC1C3D", "#F0F2F4", "#3E4054")),
    Palette("hot-pink-gold-purple", ("#FE1964", "#FFC31A", "#4B169F")),
    Palette("sky-pink-peach", ("#7ED9FE", "#FF7DAC", "#FFA17F")),
    Palette("sun-blue-violet", ("#FDC326", "#4C92FE", "#8D49EA")),
    Palette("mint-rose-butter", ("#22D9A7", "#F1597B", "#FED576")),
)


@dataclass(frozen=True, slots=True)
class RadarBubbleData:
    """A ``categories x series`` profile matrix plus optional bubble areas."""

    categories: tuple[str, ...]
    series: tuple[str, ...]
    values: tuple[tuple[float, ...], ...]
    sizes: tuple[tuple[float, ...], ...] | None = None
    value_label: str = "Value"
    size_label: str = ""
    value_format: str = "{:.0f}"

    @classmethod
    def from_matrix(
        cls,
        *,
        categories: Sequence[str],
        series: Sequence[str],
        values: Sequence[Sequence[float | None]],
        sizes: Sequence[Sequence[float | None]] | None = None,
        value_label: str = "Value",
        size_label: str = "",
        value_format: str = "{:.0f}",
    ) -> "RadarBubbleData":
        """Build from ``[category][series]`` matrices of measurements.

        ``sizes`` is optional; without it the bubbles and their legend are
        skipped and only the profiles are drawn.  ``None`` entries in either
        matrix become NaN and are skipped when drawing, so a partly measured
        table does not have to be padded with zeros.
        """

        built = cls(
            categories=tuple(str(name) for name in categories),
            series=tuple(str(name) for name in series),
            values=_as_matrix(values),
            sizes=None if sizes is None else _as_matrix(sizes),
            value_label=value_label,
            size_label=size_label,
            value_format=value_format,
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
        sizes: Mapping[str, Mapping[str, float | None]] | None = None,
        **kwargs: object,
    ) -> "RadarBubbleData":
        """Build from nested ``values[category][series]`` mappings."""

        def matrix(
            source: Mapping[str, Mapping[str, float | None]],
        ) -> list[list[float | None]]:
            return [
                [source.get(category, {}).get(name) for name in series]
                for category in categories
            ]

        return cls.from_matrix(
            categories=categories,
            series=series,
            values=matrix(values),
            sizes=None if sizes is None else matrix(sizes),
            **kwargs,  # type: ignore[arg-type]
        )

    def validate(self) -> None:
        if len(self.categories) < 6:
            raise ValueError("need at least six categories to fill the ring")
        if not self.series:
            raise ValueError("need at least one series")
        if len(set(self.categories)) != len(self.categories):
            raise ValueError("category labels must be unique")
        if len(set(self.series)) != len(self.series):
            raise ValueError("series labels must be unique")
        _check_shape(self.values, self.categories, self.series, "values")
        if np.count_nonzero(np.isfinite(self.matrix())) == 0:
            raise ValueError("values must contain at least one finite measurement")
        if self.sizes is not None:
            _check_shape(self.sizes, self.categories, self.series, "sizes")
            areas = self.size_matrix()
            assert areas is not None
            negative = np.isfinite(areas) & (areas < 0.0)
            if np.any(negative):
                category = self.categories[int(np.argmax(negative.any(axis=1)))]
                raise ValueError(f"sizes[{category!r}] must be non-negative")
        try:
            self.value_format.format(1.0)
        except (IndexError, KeyError, ValueError) as exc:
            raise ValueError(f"value_format {self.value_format!r} is not usable") from exc

    def matrix(self) -> NDArray[np.float64]:
        """Values as a ``(categories, series)`` array."""

        return np.asarray(self.values, dtype=float)

    def size_matrix(self) -> NDArray[np.float64] | None:
        """Bubble measurements as a ``(categories, series)`` array, if any."""

        return None if self.sizes is None else np.asarray(self.sizes, dtype=float)

    def value_range(self) -> tuple[float, float]:
        finite = self.matrix()
        finite = finite[np.isfinite(finite)]
        return float(finite.min()), float(finite.max())

    def size_range(self) -> tuple[float, float]:
        areas = self.size_matrix()
        if areas is None:
            return (0.0, 0.0)
        finite = areas[np.isfinite(areas)]
        if finite.size == 0:
            return (0.0, 0.0)
        return float(finite.min()), float(finite.max())


def _as_matrix(
    rows: Sequence[Sequence[float | None]],
) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(float("nan") if value is None else float(value) for value in row)
        for row in rows
    )


def _check_shape(
    rows: tuple[tuple[float, ...], ...],
    categories: tuple[str, ...],
    series: tuple[str, ...],
    field: str,
) -> None:
    if len(rows) != len(categories):
        raise ValueError(
            f"{field} has {len(rows)} rows but there are {len(categories)} categories"
        )
    for category, row in zip(categories, rows, strict=True):
        if len(row) != len(series):
            raise ValueError(
                f"{field}[{category!r}] has {len(row)} entries but there are "
                f"{len(series)} series"
            )


_REFERENCE_CATEGORIES: Final[tuple[str, ...]] = (
    "NX", "QH", "GS", "SN", "XZ", "YN", "GZ", "SC", "CQ", "HI",
    "GX", "GD", "HN", "HB", "HA", "SD", "JX", "FJ", "AH", "ZJ",
    "JS", "SH", "HL", "JL", "LN", "IM", "SX", "HE", "TJ", "BJ",
    "XJ",
)

_REFERENCE_SERIES: Final[tuple[str, ...]] = ("2020", "2010", "2000")

# Grain yield in 10k tons, read off the curve radii of the reference image.
_REFERENCE_YIELD: Final[tuple[tuple[float, ...], ...]] = (
    (3740, 3150, 4930),
    (3580, 2200, 5030),
    (3460, 2160, 2380),
    (3350, 2060, 2260),
    (3270, 4190, 2120),
    (3150, 4010, 2000),
    (3050, 4010, 1900),
    (2950, 3860, 1770),
    (2830, 3800, 3600),
    (2750, 3660, 3480),
    (2590, 3600, 3400),
    (4960, 3420, 3270),
    (5120, 3360, 3190),
    (4690, 3210, 3070),
    (4630, 3210, 2970),
    (4490, 3050, 2850),
    (4370, 2970, 2750),
    (4270, 4880, 2630),
    (4170, 4610, 2560),
    (4070, 4650, 2380),
    (3960, 4450, 4030),
    (3880, 4390, 4330),
    (2560, 4370, 4450),
    (2400, 2690, 4390),
    (2320, 2750, 4290),
    (2200, 2590, 4190),
    (2100, 2570, 4070),
    (1980, 2440, 3970),
    (1820, 1900, 3860),
    (1940, 1820, 3760),
    (3760, 4210, 5010),
)

# Planting area in Kha, read off the bubble sizes of the reference image.
_REFERENCE_PLANTING_AREA: Final[tuple[tuple[float, ...], ...]] = (
    (3900, 3000, 2900),
    (3500, 2400, 2700),
    (4000, 2500, 2600),
    (4200, 2200, 2300),
    (4100, 3700, 2200),
    (3300, 3400, 2100),
    (3000, 3200, 2000),
    (2900, 3000, 1900),
    (2600, 2900, 2700),
    (3000, 2800, 2600),
    (2800, 2600, 2500),
    (6000, 4400, 3500),
    (5600, 4200, 3300),
    (4600, 3500, 3000),
    (4800, 3700, 3100),
    (5200, 4100, 3400),
    (5400, 3900, 3200),
    (4700, 5200, 2800),
    (5600, 5000, 3000),
    (4500, 5600, 2700),
    (4300, 5300, 3900),
    (4400, 4800, 4200),
    (2700, 4300, 4500),
    (2400, 3100, 4100),
    (2500, 3000, 3800),
    (2300, 2800, 3600),
    (2700, 2700, 3400),
    (2200, 2500, 3300),
    (2100, 2200, 3100),
    (2000, 2100, 3000),
    (4100, 3600, 3700),
)

DEFAULT_DATA: Final[RadarBubbleData] = RadarBubbleData.from_matrix(
    categories=_REFERENCE_CATEGORIES,
    series=_REFERENCE_SERIES,
    values=_REFERENCE_YIELD,
    sizes=_REFERENCE_PLANTING_AREA,
    value_label="Grain Yield\n(10k tons)",
    size_label="Planting Area (Kha)",
    value_format="{:.0f}",
)


@dataclass(frozen=True, slots=True)
class ChartStyle:
    """Geometry and typography tuned to the 1196 x 1080 reference image.

    Angles come from the category count and the bubble rings from the series
    count, so the reference 31 regions on three rings fall out of the data
    rather than being pinned here.  ``value_limits`` and ``size_limits`` keep
    the reference scales; both auto-fit whenever the data would overflow them.
    """

    figure_size: tuple[float, float] = (11.96, 10.8)
    x_limits: tuple[float, float] = (-1.333, 1.620)
    y_limits: tuple[float, float] = (-1.333, 1.333)
    inner_radius: float = 0.30
    outer_radius: float = 1.00
    value_base_radius: float = 0.125
    """Radius of the low end of the value scale, inside the white hub."""

    value_limits: tuple[float, float] | None = (0.0, 7000.0)
    size_limits: tuple[float, float] | None = (0.0, 7000.0)
    min_scale_fill: float = 0.25
    """Share of a pinned range the data must fill before the scale auto-fits."""

    bubble_ring_span: tuple[float, float] = (0.90, 0.50)
    """Radii of the outermost and innermost bubble rings."""

    bubble_area_range: tuple[float, float] = (80.0, 535.0)
    """Marker area in points squared at the two ends of the size scale."""

    bubble_pitch_fraction: float = 0.95
    """Largest bubble diameter as a share of the pitch on the inner ring."""

    fill_alpha: float = 0.46
    curve_width: float = 2.2
    grid_color: str = "#B9B9B9"
    grid_width: float = 1.0
    outer_width: float = 2.4
    label_radius: float = 1.105
    tick_outer_radius: float = 1.052
    samples_per_segment: int = 18
    category_font_size: float = 12.5
    center_font_size: float = 17.0
    axis_font_size: float = 10.5
    legend_font_size: float = 11.0
    legend_title_font_size: float = 11.5
    legend_x: float = 1.21
    legend_bubble_x: float = 1.33
    size_legend_text_x: float = 1.42
    size_legend_title_y: float = 0.46
    size_legend_span: tuple[float, float] = (0.35, 0.135)
    series_legend_title: str = ""
    """Heading above the series swatches; the reference figure says "Year"."""

    series_legend_title_x: float = 1.24
    series_legend_title_y: float = -0.065
    series_legend_text_x: float = 1.40
    series_legend_pitch: float = 0.085
    series_swatch_size: tuple[float, float] = (0.09, 0.036)

    def validate(self, *, categories: int, series: int) -> None:
        if min(self.figure_size) <= 0:
            raise ValueError("figure_size must be positive")
        if categories < 1 or series < 1:
            raise ValueError("categories and series must be positive")
        if not 0.0 < self.inner_radius < self.outer_radius:
            raise ValueError("radii must satisfy 0 < inner_radius < outer_radius")
        if not 0.0 < self.value_base_radius < self.outer_radius:
            raise ValueError("value_base_radius must sit inside the outer radius")
        ring_outer, ring_inner = self.bubble_ring_span
        if not self.inner_radius < ring_inner <= ring_outer < self.outer_radius:
            raise ValueError("bubble rings must sit between the inner and outer radii")
        if min(self.bubble_area_range) <= 0.0:
            raise ValueError("bubble_area_range must be positive")
        if not 0.0 <= self.min_scale_fill < 1.0:
            raise ValueError("min_scale_fill must sit in [0, 1)")
        if self.samples_per_segment < 2:
            raise ValueError("samples_per_segment must be at least two")


DEFAULT_STYLE: Final[ChartStyle] = ChartStyle(series_legend_title="Year")


def _angles(count: int) -> NDArray[np.float64]:
    """Category angles starting at 12 o'clock and running clockwise."""

    return np.pi / 2.0 - np.arange(count, dtype=float) * 2.0 * np.pi / count


def _scale_limits(
    pinned: tuple[float, float] | None,
    low: float,
    high: float,
    *,
    style: ChartStyle,
) -> tuple[float, float]:
    """Reference limits while the data fills them, otherwise a fitted range.

    ``resolve_limits`` alone keeps a pinned range whenever the data fits inside
    it, which would squash a dataset three orders of magnitude smaller than the
    reference into the hub. The extra fill test drops the pin in that case.
    """

    limits = resolve_limits(pinned, low, high)
    span = limits[1] - limits[0]
    if span > 0.0 and (high - low) < style.min_scale_fill * span:
        return padded_range(low, high)
    return limits


def _points_per_unit(style: ChartStyle) -> float:
    """Points spanned by one data unit once the equal aspect is applied."""

    width, height = style.figure_size
    x_span = style.x_limits[1] - style.x_limits[0]
    y_span = style.y_limits[1] - style.y_limits[0]
    return min(72.0 * width / x_span, 72.0 * height / y_span)


def axis_ticks(low: float, high: float) -> NDArray[np.float64]:
    """Round tick values inside ``(low, high]``, excluding the baseline."""

    step = nice_step(high - low)
    first = math.ceil(low / step) * step
    if first <= low + 1e-9 * max(1.0, abs(low)):
        first += step
    count = int(math.floor((high - first) / step + 1e-9)) + 1
    if count < 1:
        return np.array([high], dtype=float)
    return first + step * np.arange(count, dtype=float)


def value_radii(
    values: NDArray[np.float64] | Sequence[float],
    limits: tuple[float, float],
    style: ChartStyle,
) -> NDArray[np.float64]:
    """Map measurements onto the radial band of the ring."""

    low, high = limits
    span = high - low
    fraction = (np.asarray(values, dtype=float) - low) / (span if span else 1.0)
    return style.value_base_radius + fraction * (
        style.outer_radius - style.value_base_radius
    )


def bubble_areas(
    values: NDArray[np.float64] | Sequence[float],
    limits: tuple[float, float],
    style: ChartStyle,
    *,
    categories: int,
) -> NDArray[np.float64]:
    """Map the second measurement onto marker areas in points squared.

    The area range shrinks when many categories crowd the innermost ring, so
    neighbouring bubbles keep their gap however long the category list is.
    """

    low, high = limits
    span = high - low
    fraction = (np.asarray(values, dtype=float) - low) / (span if span else 1.0)
    small, large = style.bubble_area_range
    areas = small + np.clip(fraction, 0.0, 1.0) * (large - small)

    pitch = 2.0 * np.pi * min(style.bubble_ring_span) / categories
    allowed = style.bubble_pitch_fraction * pitch * _points_per_unit(style)
    widest = 2.0 * math.sqrt(large / math.pi)
    scale = min(1.0, allowed / widest) if widest > 0.0 else 1.0
    return areas * scale**2


def bubble_ring_radii(count: int, style: ChartStyle) -> NDArray[np.float64]:
    """One ring radius per series, outermost first."""

    outer, inner = style.bubble_ring_span
    if count == 1:
        return np.array([outer], dtype=float)
    return np.linspace(outer, inner, count)


def _smooth_periodic_radii(
    positions: Sequence[float],
    radii: Sequence[float],
    *,
    categories: int,
    samples_per_segment: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Interpolate cyclic samples with a closed Catmull-Rom spline.

    ``positions`` are the category indices that carry a finite value, so a
    profile with gaps still closes smoothly across the missing spokes.
    """

    nodes = np.asarray(radii, dtype=float)
    index = np.asarray(positions, dtype=float)
    count = len(nodes)
    u = np.linspace(0.0, 1.0, samples_per_segment, endpoint=False)
    pieces: list[NDArray[np.float64]] = []
    parameters: list[NDArray[np.float64]] = []

    for step in range(count):
        p0 = nodes[(step - 1) % count]
        p1 = nodes[step]
        p2 = nodes[(step + 1) % count]
        p3 = nodes[(step + 2) % count]
        pieces.append(
            0.5
            * (
                2.0 * p1
                + (-p0 + p2) * u
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * u**2
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * u**3
            )
        )
        start = index[step]
        end = index[(step + 1) % count] + (categories if step == count - 1 else 0.0)
        parameters.append(start + (end - start) * u)

    smoothed = np.concatenate(pieces)
    parameter = np.concatenate(parameters)
    theta = np.pi / 2.0 - 2.0 * np.pi * parameter / categories
    return (
        np.append(theta, theta[0] - 2.0 * np.pi),
        np.append(smoothed, smoothed[0]),
    )


def _draw_grid(
    ax: Axes,
    data: RadarBubbleData,
    limits: tuple[float, float],
    style: ChartStyle,
) -> None:
    for angle in _angles(len(data.categories)):
        cosine, sine = np.cos(angle), np.sin(angle)
        ax.plot(
            [style.inner_radius * cosine, style.outer_radius * cosine],
            [style.inner_radius * sine, style.outer_radius * sine],
            color=style.grid_color,
            linewidth=style.grid_width,
            zorder=1,
        )

    for radius in value_radii(axis_ticks(*limits), limits, style):
        ax.add_patch(
            Circle(
                (0.0, 0.0),
                float(radius),
                facecolor="none",
                edgecolor=style.grid_color,
                linewidth=style.grid_width,
                zorder=1,
            )
        )


def _profile_curve(
    column: NDArray[np.float64],
    limits: tuple[float, float],
    style: ChartStyle,
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    finite = np.flatnonzero(np.isfinite(column))
    if finite.size < 3:
        return None
    theta, radii = _smooth_periodic_radii(
        finite.astype(float),
        value_radii(column[finite], limits, style),
        categories=len(column),
        samples_per_segment=style.samples_per_segment,
    )
    return radii * np.cos(theta), radii * np.sin(theta)


def _draw_profiles(
    ax: Axes,
    data: RadarBubbleData,
    palette: Palette,
    limits: tuple[float, float],
    style: ChartStyle,
) -> None:
    matrix = data.matrix()
    colors = palette.take(len(data.series))
    # Last series first so the leading series keeps the top of the stack.
    order = list(reversed(range(len(data.series))))
    curves = {
        index: _profile_curve(matrix[:, index], limits, style) for index in order
    }

    for index in order:
        curve = curves[index]
        if curve is None:
            continue
        ax.fill(
            curve[0],
            curve[1],
            facecolor=colors[index],
            edgecolor="none",
            alpha=style.fill_alpha,
            zorder=3,
        )
    for index in order:
        curve = curves[index]
        if curve is None:
            continue
        ax.plot(
            curve[0],
            curve[1],
            color=colors[index],
            linewidth=style.curve_width,
            solid_joinstyle="round",
            solid_capstyle="round",
            zorder=5,
        )


def _draw_bubbles(
    ax: Axes,
    data: RadarBubbleData,
    palette: Palette,
    limits: tuple[float, float],
    style: ChartStyle,
) -> None:
    areas = data.size_matrix()
    if areas is None:
        return
    n_categories = len(data.categories)
    theta = _angles(n_categories)
    colors = palette.take(len(data.series))
    rings = bubble_ring_radii(len(data.series), style)

    for index, radius in enumerate(rings):
        column = areas[:, index]
        finite = np.isfinite(column)
        if not np.any(finite):
            continue
        color = colors[index]
        ax.scatter(
            radius * np.cos(theta[finite]),
            radius * np.sin(theta[finite]),
            s=bubble_areas(column[finite], limits, style, categories=n_categories),
            facecolors=[to_rgba(color, 0.56)],
            edgecolors=[to_rgba(color, 0.92)],
            linewidths=1.0,
            zorder=7,
        )


def _draw_outer_labels(ax: Axes, data: RadarBubbleData, style: ChartStyle) -> None:
    n_categories = len(data.categories)
    pitch = 2.0 * np.pi * style.label_radius / n_categories * _points_per_unit(style)
    font_size = min(style.category_font_size, max(5.0, 0.9 * pitch))

    for label, angle in zip(data.categories, _angles(n_categories), strict=True):
        cosine, sine = np.cos(angle), np.sin(angle)
        ax.plot(
            [style.outer_radius * cosine, style.tick_outer_radius * cosine],
            [style.outer_radius * sine, style.tick_outer_radius * sine],
            color="black",
            linewidth=1.8,
            zorder=15,
        )
        if cosine > 0.18:
            horizontal = "left"
        elif cosine < -0.18:
            horizontal = "right"
        else:
            horizontal = "center"
        if sine > 0.18:
            vertical = "bottom"
        elif sine < -0.18:
            vertical = "top"
        else:
            vertical = "center"
        ax.text(
            style.label_radius * cosine,
            style.label_radius * sine,
            label,
            ha=horizontal,
            va=vertical,
            fontsize=font_size,
            color="black",
            zorder=20,
        )

    ax.add_patch(
        Circle(
            (0.0, 0.0),
            style.outer_radius,
            facecolor="none",
            edgecolor="black",
            linewidth=style.outer_width,
            zorder=14,
        )
    )


def _draw_radial_axis(
    ax: Axes,
    data: RadarBubbleData,
    limits: tuple[float, float],
    style: ChartStyle,
) -> None:
    ax.plot(
        [0.0, 0.0],
        [style.inner_radius, style.tick_outer_radius],
        color="black",
        linewidth=1.8,
        zorder=16,
    )
    ticks = axis_ticks(*limits)
    for tick, radius in zip(ticks, value_radii(ticks, limits, style), strict=True):
        ax.plot(
            [0.0, 0.026],
            [radius, radius],
            color="black",
            linewidth=1.5,
            zorder=17,
        )
        ax.text(
            0.031,
            radius,
            data.value_format.format(float(tick)),
            ha="left",
            va="center",
            fontsize=style.axis_font_size,
            zorder=18,
        )


def _draw_center(ax: Axes, data: RadarBubbleData, style: ChartStyle) -> None:
    ax.add_patch(
        Circle(
            (0.0, 0.0),
            style.inner_radius,
            facecolor="white",
            edgecolor="black",
            linewidth=2.2,
            zorder=24,
        )
    )
    ax.text(
        0.0,
        0.0,
        data.value_label,
        ha="center",
        va="center",
        fontsize=style.center_font_size,
        linespacing=1.05,
        zorder=25,
    )


def _draw_size_legend(
    ax: Axes,
    data: RadarBubbleData,
    limits: tuple[float, float],
    style: ChartStyle,
) -> None:
    ticks = axis_ticks(*limits)
    if len(ticks) > 3:
        ticks = ticks[np.linspace(0, len(ticks) - 1, 3).round().astype(int)]
    top, bottom = style.size_legend_span
    rows = (
        np.array([top]) if len(ticks) == 1 else np.linspace(top, bottom, len(ticks))
    )

    ax.text(
        style.legend_x,
        style.size_legend_title_y,
        data.size_label,
        fontsize=style.legend_title_font_size,
        ha="left",
    )
    ax.scatter(
        np.full_like(rows, style.legend_bubble_x),
        rows,
        s=bubble_areas(
            ticks, limits, style, categories=len(data.categories)
        ),
        facecolors="white",
        edgecolors="black",
        linewidths=1.2,
        zorder=30,
    )
    for tick, y in zip(ticks, rows, strict=True):
        ax.text(
            style.size_legend_text_x,
            y,
            data.value_format.format(float(tick)),
            fontsize=style.legend_font_size,
            ha="left",
            va="center",
        )


def _draw_series_legend(
    ax: Axes,
    data: RadarBubbleData,
    palette: Palette,
    style: ChartStyle,
    *,
    title_y: float,
) -> None:
    colors = palette.take(len(data.series))
    swatch_width, swatch_height = style.series_swatch_size
    if style.series_legend_title:
        ax.text(
            style.series_legend_title_x,
            title_y,
            style.series_legend_title,
            fontsize=style.legend_title_font_size,
            ha="left",
        )
    for index, name in enumerate(data.series):
        y = title_y - 0.08 - index * style.series_legend_pitch
        ax.add_patch(
            Rectangle(
                (style.legend_x, y - 0.5 * swatch_height),
                swatch_width,
                swatch_height,
                facecolor=colors[index],
                edgecolor="none",
                zorder=30,
            )
        )
        ax.text(
            style.series_legend_text_x,
            y,
            name,
            fontsize=style.legend_font_size,
            ha="left",
            va="center",
        )


def create_figure(
    palette: Palette = PALETTES[0],
    data: RadarBubbleData = DEFAULT_DATA,
    style: ChartStyle = DEFAULT_STYLE,
) -> Figure:
    """Create the annular radar-bubble figure without writing it to disk."""

    data.validate()
    style.validate(categories=len(data.categories), series=len(data.series))
    value_limits = _scale_limits(style.value_limits, *data.value_range(), style=style)
    size_limits = (
        None
        if data.sizes is None
        else _scale_limits(style.size_limits, *data.size_range(), style=style)
    )

    with plt.rc_context(
        {
            "font.family": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
            "font.size": style.legend_font_size,
            "axes.linewidth": 1.5,
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

        _draw_grid(ax, data, value_limits, style)
        _draw_profiles(ax, data, palette, value_limits, style)
        if size_limits is not None:
            _draw_bubbles(ax, data, palette, size_limits, style)
        _draw_outer_labels(ax, data, style)
        _draw_radial_axis(ax, data, value_limits, style)
        _draw_center(ax, data, style)
        if size_limits is None:
            # Without bubbles the series legend takes over the top slot.
            _draw_series_legend(
                ax, data, palette, style, title_y=style.size_legend_title_y
            )
        else:
            _draw_size_legend(ax, data, size_limits, style)
            _draw_series_legend(
                ax, data, palette, style, title_y=style.series_legend_title_y
            )

    return figure


def main(argv: Sequence[str] | None = None) -> int:
    return run_cli(sys.modules[__name__], argv)


if __name__ == "__main__":
    raise SystemExit(main())
