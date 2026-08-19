"""Smooth-curve radar over bounded scores, one closed spline per series.

Each series becomes a periodic spline through its value on every spoke, so the
overall shape and the per-spoke weak spots read at once.  Spokes may carry a
group label; the group colours the spoke's text box and appears in a small
legend.

``DEFAULT_DATA`` follows Figure 1c of Neidlinger et al. (2026), Nature
Communications, CC BY 4.0.  The source post publishes no table, so the AUROC
values were digitised from the labelled radii; the published EAGLE spreadsheet
uses a different per-task normalisation and does not match the printed labels.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Final, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch
from numpy.typing import NDArray
from scipy.interpolate import make_interp_spline

from ..contract import DataKind, Extent, Feature, Geometry, TemplateSpec
from ..palettes import Palette
from ..render import nice_step, resolve_limits, run_cli

SPEC: Final[TemplateSpec] = TemplateSpec(
    template_id="smooth-radar",
    title="Smooth-curve radar over bounded scores",
    summary=(
        "One spline-smoothed closed curve per series across labelled spokes, with "
        "spoke labels optionally coloured by the group each spoke belongs to."
    ),
    kinds=(DataKind.MATRIX,),
    geometry=Geometry.POLAR,
    categories=Extent(3, 48),
    series=Extent(1, 8),
    builder="SmoothRadarData.from_matrix",
    data_contract=(
        "A value for every (spoke, series) pair on a shared bounded scale, such as "
        "an AUROC or accuracy. Spokes may carry a group label that colours the "
        "spoke text."
    ),
    good_for=(
        "comparing models across many benchmark tasks on one bounded metric",
        "showing overall shape and per-task weak spots at once",
        "grouping tasks into families via spoke label colour",
    ),
    avoid_when=(
        "values are unbounded or span orders of magnitude",
        "there are fewer than three spokes",
        "the reader must rank series precisely on a single spoke",
    ),
    argument_names=(("categories", "spokes"),),
    affinities=(
        (Feature.BOUNDED_SCALE, 13.0),
        (Feature.MANY_CATEGORIES, 6.0),
        (Feature.WIDE_DYNAMIC_RANGE, -14.0),
        (Feature.LONG_LABELS, -3.0),
    ),
    default_dpi=200,
    reference="Neidlinger et al. (2026), Nature Communications, Fig. 1c, CC BY 4.0.",
)


# Series swatches sampled from carousel legends 1-18.  Named qualitative sets
# use canonical ColorBrewer / Tableau / Paul Tol hex values.
PALETTES: Final[tuple[Palette, ...]] = (
    Palette("crimson-peach-ice", ("#C83E4D", "#F4B38C", "#D0E1F9", "#85C1E9", "#2E86C1")),
    Palette("set1", ("#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00")),
    Palette("dark2", ("#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#66A61E")),
    Palette("set2", ("#FC8D62", "#8DA0CB", "#E78AC3", "#A6D854", "#FFD92F")),
    Palette("tableau10", ("#4E79A7", "#F28E2C", "#E15759", "#76B7B2", "#59A14F")),
    Palette("material-bold", ("#D81B60", "#1E88E5", "#FFC107", "#004D40", "#8E24AA")),
    Palette("neon-rainbow", ("#FF595E", "#FFCA3A", "#8AC926", "#1982C4", "#6A4C93")),
    Palette("sunset-teal", ("#F94144", "#F3722C", "#90BE6D", "#43AA8B", "#577590")),
    Palette("atlantic-sunset", ("#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51")),
    Palette("teal-cream-coral", ("#0081A7", "#00AFB9", "#FDFCDC", "#FED9B7", "#F07167")),
    Palette("tol-muted", ("#332288", "#117733", "#44AA99", "#88CCEE", "#DDCC77")),
    Palette("tol-wine", ("#CC6677", "#AA4499", "#882255", "#332288", "#DDCC77")),
    Palette("venice", ("#E63946", "#F1FAEE", "#A8DADC", "#457B9D", "#1D3557")),
    Palette("space-red", ("#2B2D42", "#8D99AE", "#EDF2F4", "#EF233C", "#D90429")),
    Palette("forest-wine", ("#386641", "#6A994E", "#A7C957", "#F2E8CF", "#BC4749")),
    Palette("spectral", ("#D53E4F", "#FC8D59", "#FEE08B", "#E6F598", "#99D594")),
    Palette("set3", ("#8DD3C7", "#FFFFB3", "#BEBADA", "#FB8072", "#80B1D3")),
    Palette("paired", ("#A6CEE3", "#1F78B4", "#B2DF8A", "#33A02C", "#FB9A99")),
)

# Spoke-label inks, cycled over the groups in first-appearance order.  These are
# the three category colours of the reference carousel, reordered so its first
# group (Biomarkers) keeps its original pink.
_AXIS_GROUP_COLORS: Final[tuple[str, ...]] = ("#C24B7A", "#8B3A2F", "#4B2C7A")

_NEUTRAL_INK: Final[str] = "#555555"
"""Spoke-label ink when the data declares no groups."""

# Small per-series nudges so labels stacked on one spoke stay readable.
_VALUE_LABEL_OFFSETS: Final[tuple[tuple[float, float], ...]] = (
    (0.012, 0.000),
    (-0.010, 0.012),
    (0.014, -0.010),
    (-0.016, -0.008),
    (0.000, 0.014),
)


@dataclass(frozen=True, slots=True)
class SmoothRadarData:
    """A ``spokes x series`` matrix on one bounded scale."""

    spokes: tuple[str, ...]
    series: tuple[str, ...]
    values: tuple[tuple[float, ...], ...]
    spoke_groups: tuple[str | None, ...] | None = None
    value_label: str = "Score"
    value_limits: tuple[float, float] | None = None
    value_format: str = "{:.2f}"

    @classmethod
    def from_matrix(
        cls,
        *,
        spokes: Sequence[str],
        series: Sequence[str],
        values: Sequence[Sequence[float | None]],
        spoke_groups: Sequence[str | None] | None = None,
        value_label: str = "Score",
        value_limits: tuple[float, float] | None = None,
        value_format: str = "{:.2f}",
    ) -> "SmoothRadarData":
        """Build from a ``[spoke][series]`` matrix of bounded scores.

        ``spoke_groups`` is optional; without it every spoke label uses one
        neutral ink and the group legend is skipped.  ``None`` entries in
        ``values`` become NaN and the curve is interpolated across them rather
        than dropping to zero.
        """

        built = cls(
            spokes=tuple(str(name) for name in spokes),
            series=tuple(str(name) for name in series),
            values=tuple(
                tuple(float("nan") if value is None else float(value) for value in row)
                for row in values
            ),
            spoke_groups=(
                None
                if spoke_groups is None
                else tuple(None if group is None else str(group) for group in spoke_groups)
            ),
            value_label=value_label,
            value_limits=None if value_limits is None else (
                float(value_limits[0]),
                float(value_limits[1]),
            ),
            value_format=value_format,
        )
        built.validate()
        return built

    @classmethod
    def from_mapping(
        cls,
        *,
        spokes: Sequence[str],
        series: Sequence[str],
        values: Mapping[str, Mapping[str, float | None]],
        **kwargs: object,
    ) -> "SmoothRadarData":
        """Build from nested ``values[spoke][series]`` mappings."""

        matrix = [
            [values.get(spoke, {}).get(name) for name in series] for spoke in spokes
        ]
        return cls.from_matrix(
            spokes=spokes, series=series, values=matrix, **kwargs  # type: ignore[arg-type]
        )

    def validate(self) -> None:
        if len(self.spokes) < 3:
            raise ValueError("need at least three spokes")
        if not self.series:
            raise ValueError("need at least one series")
        if len(set(self.spokes)) != len(self.spokes):
            raise ValueError("spoke labels must be unique")
        if len(set(self.series)) != len(self.series):
            raise ValueError("series labels must be unique")
        if len(self.values) != len(self.spokes):
            raise ValueError(
                f"values has {len(self.values)} rows but there are "
                f"{len(self.spokes)} spokes"
            )
        for spoke, row in zip(self.spokes, self.values, strict=True):
            if len(row) != len(self.series):
                raise ValueError(
                    f"values[{spoke!r}] has {len(row)} entries but there are "
                    f"{len(self.series)} series"
                )
        if np.count_nonzero(np.isfinite(self.matrix())) == 0:
            raise ValueError("values must contain at least one finite measurement")
        if self.spoke_groups is not None and len(self.spoke_groups) != len(self.spokes):
            raise ValueError(
                f"spoke_groups has {len(self.spoke_groups)} entries but there are "
                f"{len(self.spokes)} spokes"
            )
        if self.value_limits is not None and self.value_limits[0] >= self.value_limits[1]:
            raise ValueError("value_limits must be increasing")
        try:
            self.value_format.format(1.0)
        except (IndexError, KeyError, ValueError) as exc:
            raise ValueError(f"value_format {self.value_format!r} is not usable") from exc

    def matrix(self) -> NDArray[np.float64]:
        """Values as a ``(spokes, series)`` array."""

        return np.asarray(self.values, dtype=float)

    def value_range(self) -> tuple[float, float]:
        finite = self.matrix()
        finite = finite[np.isfinite(finite)]
        return float(finite.min()), float(finite.max())

    def group_order(self) -> tuple[str, ...]:
        """Distinct spoke groups in first-appearance order."""

        if self.spoke_groups is None:
            return ()
        seen: list[str] = []
        for group in self.spoke_groups:
            if group is not None and group not in seen:
                seen.append(group)
        return tuple(seen)

    def group_colors(self) -> Mapping[str, str]:
        """Ink for every group, cycling :data:`_AXIS_GROUP_COLORS`."""

        return {
            group: _AXIS_GROUP_COLORS[index % len(_AXIS_GROUP_COLORS)]
            for index, group in enumerate(self.group_order())
        }


_REFERENCE_SPOKES: Final[tuple[str, ...]] = (
    "CPTAC CRC KRAS",
    "CPTAC CRC PIK3CA",
    "BERN STAD N-STATUS",
    "CPTAC CRC N-STATUS",
    "CPTAC CRC Sidedness",
    "CPTAC LUAD KRAS",
    "IEO BRCA N-STATUS",
    "DACHS CRC KRAS",
    "KIEL STAD M-STATUS",
    "CPTAC NSCLC Subtyping",
    "CPTAC CRC MSI",
    "CPTAC BRCA ESR1",
    "DACHS CRC MSI",
    "KIEL STAD EBV",
    "BERN STAD MSI",
    "KIEL STAD MSI",
    "KIEL STAD LAUREN",
    "DACHS CRC BRAF",
    "CPTAC BRCA PGR",
    "CPTAC CRC BRAF",
    "CPTAC LUAD STK11",
    "BERN STAD LAUREN",
    "CPTAC LUAD TP53",
    "CPTAC LUAD EGFR",
    "DACHS CRC Sidedness",
    "CPTAC BRCA ERBB2",
    "DACHS CRC M-STATUS",
    "DACHS CRC CIMP",
    "CPTAC BRCA PIK3CA",
    "DACHS CRC N-STATUS",
    "KIEL STAD N-STATUS",
)

_REFERENCE_SERIES: Final[tuple[str, ...]] = (
    "EAGLE",
    "CHIEF",
    "GigaPath",
    "CTransPath",
    "Virchow2",
)

_REFERENCE_SPOKE_GROUPS: Final[tuple[str, ...]] = (
    "Biomarkers",
    "Biomarkers",
    "Prognosis",
    "Prognosis",
    "Morphology",
    "Biomarkers",
    "Prognosis",
    "Biomarkers",
    "Prognosis",
    "Morphology",
    "Biomarkers",
    "Biomarkers",
    "Biomarkers",
    "Biomarkers",
    "Biomarkers",
    "Biomarkers",
    "Morphology",
    "Biomarkers",
    "Biomarkers",
    "Biomarkers",
    "Biomarkers",
    "Morphology",
    "Biomarkers",
    "Biomarkers",
    "Morphology",
    "Biomarkers",
    "Prognosis",
    "Biomarkers",
    "Biomarkers",
    "Prognosis",
    "Prognosis",
)

# AUROC per (task, model), digitised from the labelled radii on the carousel.
_REFERENCE_AUROC: Final[tuple[tuple[float, ...], ...]] = (
    (0.81, 0.84, 0.57, 0.59, 0.73),
    (0.83, 0.66, 0.52, 0.53, 0.73),
    (0.67, 0.71, 0.50, 0.54, 0.64),
    (0.77, 0.85, 0.51, 0.42, 0.70),
    (0.72, 0.68, 0.48, 0.48, 0.63),
    (0.73, 0.66, 0.46, 0.44, 0.62),
    (0.77, 0.70, 0.52, 0.38, 0.62),
    (0.89, 0.89, 0.61, 0.65, 0.67),
    (0.67, 0.82, 0.52, 0.49, 0.78),
    (0.76, 0.80, 0.54, 0.69, 0.69),
    (0.86, 0.67, 0.52, 0.50, 0.77),
    (0.85, 0.77, 0.45, 0.47, 0.81),
    (0.87, 0.77, 0.40, 0.40, 0.71),
    (0.67, 0.73, 0.61, 0.60, 0.63),
    (0.91, 0.91, 0.46, 0.54, 0.83),
    (0.88, 0.66, 0.58, 0.58, 0.82),
    (0.91, 0.92, 0.62, 0.60, 0.70),
    (0.76, 0.64, 0.53, 0.52, 0.84),
    (0.83, 0.69, 0.60, 0.57, 0.69),
    (0.80, 0.74, 0.53, 0.68, 0.67),
    (0.81, 0.59, 0.36, 0.35, 0.83),
    (0.74, 0.54, 0.54, 0.56, 0.81),
    (0.73, 0.65, 0.50, 0.40, 0.76),
    (0.64, 0.66, 0.50, 0.48, 0.76),
    (0.76, 0.59, 0.50, 0.49, 0.65),
    (0.74, 0.81, 0.48, 0.51, 0.84),
    (0.84, 0.66, 0.57, 0.60, 0.89),
    (0.82, 0.61, 0.41, 0.40, 0.69),
    (0.84, 0.77, 0.36, 0.35, 0.84),
    (0.69, 0.58, 0.58, 0.60, 0.69),
    (0.67, 0.68, 0.48, 0.57, 0.61),
)

DEFAULT_DATA: Final[SmoothRadarData] = SmoothRadarData.from_matrix(
    spokes=_REFERENCE_SPOKES,
    series=_REFERENCE_SERIES,
    values=_REFERENCE_AUROC,
    spoke_groups=_REFERENCE_SPOKE_GROUPS,
    value_label="AUROC",
    value_limits=(0.0, 1.0),
    value_format="{:.2f}",
)


@dataclass(frozen=True, slots=True)
class ChartStyle:
    """Layout tuned to the 2560 x 2243 reference canvas.

    Spoke angles come from the spoke count and the gridlines from the resolved
    value range, so the reference 31 tasks on five rings fall out of the data.
    The frame is a fixed square around the unit circle with room on the right
    for the two legends.
    """

    figure_size: tuple[float, float] = (10.24, 8.972)
    x_limits: tuple[float, float] = (-1.38, 1.770)
    y_limits: tuple[float, float] = (-1.38, 1.38)
    outer_radius: float = 1.00
    center_radius: float = 0.038
    label_radius: float = 1.20
    value_step: float | None = None
    """Spacing of the ring gridlines; ``None`` derives a 1/2/5 step."""

    samples_per_segment: int = 24
    curve_width: float = 2.55
    grid_color: str = "#D3D3D3"
    outer_color: str = "#777777"
    center_color: str = "#D3D3D3"
    grid_width: float = 0.9
    outer_width: float = 2.15
    value_font_size: float = 7.4
    label_font_size: float = 8.0
    legend_font_size: float = 9.5
    group_font_size: float = 9.0
    show_values: bool = True
    show_value_label: bool = False
    """The reference prints no legend heading; enable it to title the legend."""

    legend_anchor: tuple[float, float] = (0.995, 0.995)
    group_legend_size: tuple[float, float] = (0.30, 0.072)
    group_legend_gap: float = 0.018
    group_legend_margin: tuple[float, float] = (0.04, 0.055)

    def validate(self, *, categories: int, series: int) -> None:
        if min(self.figure_size) <= 0:
            raise ValueError("figure_size must be positive")
        if categories < 1 or series < 1:
            raise ValueError("categories and series must be positive")
        if not 0.0 <= self.center_radius < self.outer_radius:
            raise ValueError("center_radius must sit inside outer_radius")
        if self.label_radius <= self.outer_radius:
            raise ValueError("spoke labels must sit outside the outer ring")
        if self.samples_per_segment < 2:
            raise ValueError("samples_per_segment must be at least two")
        if self.value_step is not None and self.value_step <= 0:
            raise ValueError("value_step must be positive when given")


DEFAULT_STYLE: Final[ChartStyle] = ChartStyle()


def split_spoke_label(spoke: str) -> tuple[str, str]:
    """Split a spoke label into two display lines, breaking before the last word."""

    parts = spoke.split()
    if len(parts) < 2:
        return spoke, ""
    return " ".join(parts[:-1]), parts[-1]


def _angles(count: int) -> NDArray[np.float64]:
    """Spoke angles starting at 12 o'clock and running clockwise."""

    return np.pi / 2.0 - np.arange(count, dtype=float) * 2.0 * np.pi / count


def ring_ticks(limits: tuple[float, float], style: ChartStyle) -> NDArray[np.float64]:
    """Gridline values inside ``(low, high]``, excluding the collapsed centre."""

    low, high = limits
    step = style.value_step or nice_step(high - low)
    if (high - low) / step > 12.0:
        step = nice_step(high - low)
    first = math.floor(low / step + 1e-9) + 1
    last = math.floor(high / step + 1e-9)
    if last < first:
        return np.array([high], dtype=float)
    return step * np.arange(first, last + 1, dtype=float)


def value_radius(
    values: NDArray[np.float64] | Sequence[float] | float,
    limits: tuple[float, float],
    style: ChartStyle,
) -> NDArray[np.float64]:
    """Map bounded scores onto radii between the centre and the outer ring."""

    low, high = limits
    span = high - low
    fraction = (np.asarray(values, dtype=float) - low) / (span if span else 1.0)
    return fraction * style.outer_radius


def smooth_closed_curve(
    positions: Sequence[float],
    radii: Sequence[float],
    *,
    spokes: int,
    samples_per_segment: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Periodic spline through the given spoke radii, closed on itself.

    ``make_interp_spline`` needs more points than its degree, so the degree is
    clamped and a straight-segment fallback covers the few-spoke case.
    """

    nodes = np.asarray(positions, dtype=float)
    values = np.asarray(radii, dtype=float)
    count = len(nodes)
    knots = np.append(nodes, nodes[0] + spokes)
    closed = np.append(values, values[0])
    dense = np.linspace(
        float(knots[0]), float(knots[-1]), count * samples_per_segment + 1
    )
    degree = min(3, count - 1)
    if degree >= 1:
        try:
            spline = make_interp_spline(knots, closed, k=degree, bc_type="periodic")
            smoothed = spline(dense)
        except (ValueError, TypeError):
            smoothed = np.interp(dense, knots, closed)
    else:
        smoothed = np.interp(dense, knots, closed)
    theta = np.pi / 2.0 - 2.0 * np.pi * dense / spokes
    return theta, smoothed


def _darker(color: str, factor: float = 0.62) -> tuple[float, float, float]:
    return tuple(max(0.0, min(1.0, channel * factor)) for channel in to_rgb(color))


def _draw_grid(
    ax: Axes,
    data: SmoothRadarData,
    limits: tuple[float, float],
    ticks: NDArray[np.float64],
    style: ChartStyle,
) -> None:
    for angle in _angles(len(data.spokes)):
        ax.plot(
            [style.center_radius * np.cos(angle), style.outer_radius * np.cos(angle)],
            [style.center_radius * np.sin(angle), style.outer_radius * np.sin(angle)],
            color=style.grid_color,
            linewidth=style.grid_width,
            zorder=1,
        )
    for radius in value_radius(ticks, limits, style):
        outermost = bool(np.isclose(radius, style.outer_radius))
        ax.add_patch(
            Circle(
                (0.0, 0.0),
                float(radius),
                facecolor="none",
                edgecolor=style.outer_color if outermost else style.grid_color,
                linewidth=style.outer_width if outermost else style.grid_width,
                zorder=2,
            )
        )
    ax.add_patch(
        Circle(
            (0.0, 0.0),
            style.center_radius,
            facecolor=style.center_color,
            edgecolor=style.center_color,
            linewidth=0.0,
            zorder=3,
        )
    )


def _draw_curves(
    ax: Axes,
    data: SmoothRadarData,
    palette: Palette,
    limits: tuple[float, float],
    style: ChartStyle,
) -> None:
    matrix = data.matrix()
    colors = palette.take(len(data.series))
    n_spokes = len(data.spokes)
    for index in range(len(data.series)):
        column = matrix[:, index]
        finite = np.flatnonzero(np.isfinite(column))
        if finite.size < 2:
            continue
        theta, radii = smooth_closed_curve(
            finite.astype(float),
            value_radius(column[finite], limits, style),
            spokes=n_spokes,
            samples_per_segment=style.samples_per_segment,
        )
        ax.plot(
            radii * np.cos(theta),
            radii * np.sin(theta),
            color=colors[index],
            linewidth=style.curve_width,
            solid_joinstyle="round",
            solid_capstyle="round",
            zorder=5,
        )


def _draw_values(
    ax: Axes,
    data: SmoothRadarData,
    palette: Palette,
    limits: tuple[float, float],
    style: ChartStyle,
) -> None:
    matrix = data.matrix()
    theta = _angles(len(data.spokes))
    colors = palette.take(len(data.series))
    halo = [patheffects.withStroke(linewidth=3.2, foreground="white")]
    for index in range(len(data.series)):
        color = colors[index]
        dx, dy = _VALUE_LABEL_OFFSETS[index % len(_VALUE_LABEL_OFFSETS)]
        radii = value_radius(matrix[:, index], limits, style)
        for angle, radius, value in zip(theta, radii, matrix[:, index], strict=True):
            if not np.isfinite(value):
                continue
            ax.text(
                radius * np.cos(angle) + dx,
                radius * np.sin(angle) + dy,
                data.value_format.format(float(value)),
                ha="center",
                va="center",
                fontsize=style.value_font_size,
                color=color,
                fontweight="bold",
                zorder=8,
                path_effects=halo,
            )


def _draw_spoke_labels(ax: Axes, data: SmoothRadarData, style: ChartStyle) -> None:
    inks = data.group_colors()
    groups = data.spoke_groups or (None,) * len(data.spokes)
    for spoke, group, angle in zip(
        data.spokes, groups, _angles(len(data.spokes)), strict=True
    ):
        color = _NEUTRAL_INK if group is None else inks[group]
        first, second = split_spoke_label(spoke)
        ax.text(
            style.label_radius * np.cos(angle),
            style.label_radius * np.sin(angle),
            first if not second else f"{first}\n{second}",
            ha="center",
            va="center",
            fontsize=style.label_font_size,
            color=color,
            fontweight="bold",
            linespacing=1.15,
            zorder=10,
            bbox={
                "boxstyle": "round,pad=0.22,rounding_size=0.45",
                "facecolor": "white",
                "edgecolor": color,
                "linewidth": 1.7,
            },
        )


def _draw_series_legend(
    ax: Axes, data: SmoothRadarData, palette: Palette, style: ChartStyle
) -> None:
    colors = palette.take(len(data.series))
    handles = [
        Line2D([0], [0], color=color, linewidth=4.2, solid_capstyle="butt", label=name)
        for color, name in zip(colors, data.series, strict=True)
    ]
    legend = ax.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=style.legend_anchor,
        title=data.value_label if style.show_value_label else None,
        frameon=True,
        fancybox=False,
        edgecolor="#C8C8C8",
        facecolor="white",
        fontsize=style.legend_font_size,
        handlelength=1.55,
        handletextpad=0.55,
        borderpad=0.55,
        labelspacing=0.42,
    )
    legend.set_zorder(20)
    for text in legend.get_texts():
        text.set_fontweight("normal")


def _draw_group_legend(ax: Axes, data: SmoothRadarData, style: ChartStyle) -> None:
    groups = data.group_order()
    if not groups:
        return
    inks = data.group_colors()
    width, height = style.group_legend_size
    right_margin, bottom_margin = style.group_legend_margin
    x = style.x_limits[1] - width - right_margin
    y0 = style.y_limits[0] + bottom_margin
    # Stacked upward, so the first group ends up on top.
    for index, group in enumerate(reversed(groups)):
        color = inks[group]
        y = y0 + index * (height + style.group_legend_gap)
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.006,rounding_size=0.028",
                facecolor=color,
                edgecolor=_darker(color, 0.55),
                linewidth=1.6,
                mutation_aspect=0.85,
                zorder=18,
                clip_on=False,
            )
        )
        ax.text(
            x + 0.5 * width,
            y + 0.5 * height,
            group,
            ha="center",
            va="center",
            fontsize=style.group_font_size,
            color="white",
            fontweight="bold",
            zorder=19,
            clip_on=False,
        )


def create_figure(
    palette: Palette = PALETTES[0],
    data: SmoothRadarData = DEFAULT_DATA,
    style: ChartStyle = DEFAULT_STYLE,
) -> Figure:
    """Create the smooth-curve radar chart without writing it to disk."""

    data.validate()
    style.validate(categories=len(data.spokes), series=len(data.series))
    limits = resolve_limits(data.value_limits, *data.value_range())
    ticks = ring_ticks(limits, style)

    with plt.rc_context(
        {
            "font.family": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
            "font.size": style.label_font_size,
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
        ax.set_facecolor("white")

        _draw_grid(ax, data, limits, ticks, style)
        _draw_curves(ax, data, palette, limits, style)
        if style.show_values:
            _draw_values(ax, data, palette, limits, style)
        _draw_spoke_labels(ax, data, style)
        _draw_series_legend(ax, data, palette, style)
        _draw_group_legend(ax, data, style)

    return figure


def main(argv: Sequence[str] | None = None) -> int:
    return run_cli(sys.modules[__name__], argv)


if __name__ == "__main__":
    raise SystemExit(main())
