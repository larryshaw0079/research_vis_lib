"""Centre pie with concentric stacked rings, one track per sector.

``DEFAULT_DATA`` holds the 2030 Beijing-Tianjin-Hebei terminal energy figure of
a Xiaohongshu carousel that repeats it 18 times, changing only the seven-colour
palette.  Ring values were digitised from the visible arc lengths on a 0-8 EJ
scale (270 degrees clockwise from 12 o'clock); the centre mix follows the
printed 52 / 26 / 13 percent wedges, with the four remaining fuels sharing the
leftover 9 percent.  The source post does not publish the underlying table;
replace ``DEFAULT_DATA`` when plotting another dataset.

One ``parts`` tuple is the colour key for the centre pie, the ring stacks and
the legend alike, and every ring is measured against one radial scale so the
tracks stay comparable.  Ring segments stack in ``parts`` order from the zero
ray; the centre pie instead runs largest share first, so the composition reads
big to small the way the reference figure does.
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
from matplotlib.patches import Arc, Rectangle, Wedge

from ..contract import DataKind, Extent, Feature, Geometry, TemplateSpec
from ..palettes import Palette, readable_text_color
from ..render import nice_step, resolve_limits, run_cli

SPEC: Final[TemplateSpec] = TemplateSpec(
    template_id="pie-ring",
    title="Centre pie with concentric stacked rings",
    summary=(
        "A composition pie in the middle, surrounded by one annular track per "
        "sector, each stacked by the same set of parts on a shared radial scale."
    ),
    kinds=(DataKind.NESTED_PARTS,),
    geometry=Geometry.CIRCULAR,
    categories=Extent(2, 10),
    series=Extent(2, 12),
    builder="PieRingData.from_nested",
    data_contract=(
        "A shared set of parts, an overall composition over those parts for the "
        "centre pie, and a value per (ring, part) pair for the surrounding tracks. "
        "All rings share one radial scale so their lengths are comparable."
    ),
    good_for=(
        "a total composition plus how each sector contributes to it",
        "energy, budget or emission mixes broken down by sector",
        "keeping one colour key across a summary and its breakdown",
    ),
    avoid_when=(
        "the rings share no common parts, so one colour key cannot serve both",
        "values can be negative and cannot stack",
        "more than about 10 rings or 12 parts",
    ),
    requires=(Feature.NON_NEGATIVE,),
    affinities=(
        (Feature.NON_NEGATIVE, 8.0),
        (Feature.SUMS_TO_100, 4.0),
        (Feature.MANY_CATEGORIES, -7.0),
        (Feature.WIDE_DYNAMIC_RANGE, -7.0),
    ),
    default_dpi=200,
    reference="Digitised from a Xiaohongshu carousel; arc lengths were read off the image.",
)


# Colours sampled from the top-left legend swatches in carousel frames 1-18.
PALETTES: Final[tuple[Palette, ...]] = (
    Palette(
        "amber-coral-azure",
        ("#FFD060", "#B3B3B3", "#FA685B", "#76B7B2", "#B2DF8A", "#85D4E3", "#4B92F0"),
    ),
    Palette(
        "terracotta-teal",
        ("#F4A261", "#E9C46A", "#E76F51", "#2A9D8F", "#8AB17D", "#A8DADC", "#264653"),
    ),
    Palette(
        "citrus-mint-navy",
        ("#FF9F1C", "#CBEEF3", "#FF1654", "#2EC4B6", "#A1E8AF", "#70C1B3", "#011627"),
    ),
    Palette(
        "olive-wheat",
        ("#DDA15E", "#FEFAE0", "#BC6C25", "#606C38", "#A3B18A", "#CCD5AE", "#283618"),
    ),
    Palette(
        "sunset-meadow",
        ("#F9C74F", "#F9844A", "#F94144", "#43AA8B", "#90BE6D", "#4D908E", "#577590"),
    ),
    Palette(
        "coral-teal",
        ("#E29578", "#FFDDD2", "#E26D5C", "#83C5BE", "#006D77", "#EDF6F9", "#005F73"),
    ),
    Palette(
        "gold-navy",
        ("#FFB703", "#FB8500", "#9D0208", "#023047", "#219EBC", "#8ECAE6", "#023E8A"),
    ),
    Palette(
        "lilac-rose",
        ("#EFC3E6", "#F0E6EF", "#9C89B8", "#B8BEDD", "#F0A6CA", "#E8D1C5", "#5C4D7D"),
    ),
    Palette(
        "amber-crimson-ink",
        ("#FCA311", "#E5E5E5", "#D90429", "#14213D", "#4A4E69", "#9A8C98", "#22223B"),
    ),
    Palette(
        "saffron-teal-ink",
        ("#EE9B00", "#E9D8A6", "#AE2012", "#0A9396", "#94D2BD", "#E09F3E", "#001219"),
    ),
    Palette(
        "mustard-navy-tomato",
        ("#F4D35E", "#FAF0CA", "#EE964B", "#0D3B66", "#8D99AE", "#F95738", "#2B2D42"),
    ),
    Palette(
        "sand-sky-navy",
        ("#D4A373", "#FAEDCD", "#E63946", "#A8DADC", "#457B9D", "#F1FAEE", "#1D3557"),
    ),
    Palette(
        "lemon-mint-ocean",
        ("#FFD166", "#FFF3B0", "#EF476F", "#06D6A0", "#118AB2", "#73D2DE", "#073B4C"),
    ),
    Palette(
        "apricot-olive-navy",
        ("#F9A03F", "#F7D08A", "#C33C54", "#8A9B68", "#A6B07E", "#E2C2C6", "#254E70"),
    ),
    Palette(
        "ochre-teal-ocean",
        ("#E9C46A", "#F4A261", "#E76F51", "#264653", "#2A9D8F", "#8AB17D", "#1E6091"),
    ),
    Palette(
        "gold-sage-lake",
        ("#F3CA40", "#F2A541", "#F08A4B", "#577590", "#43AA8B", "#90BE6D", "#277DA1"),
    ),
    Palette(
        "ice-terracotta-navy",
        ("#D6E2E9", "#F4F1DE", "#E07A5F", "#3D5A80", "#81B29A", "#F2CC8F", "#293241"),
    ),
    Palette(
        "honey-blush-slate",
        ("#F6BD60", "#F5CAC3", "#F28482", "#84A59D", "#A1C181", "#F7EDE2", "#355070"),
    ),
)


@dataclass(frozen=True, slots=True)
class PieRingData:
    """A composition over ``parts`` plus one stacked ring per sector."""

    parts: tuple[str, ...]
    center: tuple[float, ...]
    rings: tuple[str, ...]
    ring_values: tuple[tuple[float, ...], ...]
    scale_max: float | None = None
    unit_label: str = ""
    center_format: str = "{:.0f}%"
    normalize_center: bool = True
    sweep_degrees: float = 270.0
    title: str = ""

    @classmethod
    def from_nested(
        cls,
        *,
        parts: Sequence[str],
        center: Sequence[float | None],
        rings: Sequence[str],
        ring_values: Sequence[Sequence[float | None]],
        scale_max: float | None = None,
        unit_label: str = "",
        center_format: str = "{:.0f}%",
        normalize_center: bool = True,
        sweep_degrees: float = 270.0,
        title: str = "",
    ) -> "PieRingData":
        """Build from a ``[ring][part]`` matrix plus the overall composition.

        ``rings`` runs innermost track first and ``ring_values`` follows that
        order, each row holding one value per entry of ``parts``.  A part that
        does not appear in a ring contributes ``0.0``; ``None`` becomes NaN and
        is rejected, because a stack cannot silently invent a missing segment.
        """

        built = cls(
            parts=tuple(str(name) for name in parts),
            center=tuple(
                float("nan") if value is None else float(value) for value in center
            ),
            rings=tuple(str(name) for name in rings),
            ring_values=tuple(
                tuple(float("nan") if value is None else float(value) for value in row)
                for row in ring_values
            ),
            scale_max=None if scale_max is None else float(scale_max),
            unit_label=unit_label,
            center_format=center_format,
            normalize_center=bool(normalize_center),
            sweep_degrees=float(sweep_degrees),
            title=title,
        )
        built.validate()
        return built

    def validate(self) -> None:
        if len(self.parts) < 2:
            raise ValueError("need at least two parts")
        if len(self.rings) < 2:
            raise ValueError("need at least two rings")
        if len(set(self.parts)) != len(self.parts):
            raise ValueError("part labels must be unique")
        if len(set(self.rings)) != len(self.rings):
            raise ValueError("ring labels must be unique")
        if len(self.center) != len(self.parts):
            raise ValueError(
                f"center has {len(self.center)} entries but there are "
                f"{len(self.parts)} parts"
            )
        for part, value in zip(self.parts, self.center, strict=True):
            _check_value(value, f"center[{part!r}]")
        if len(self.ring_values) != len(self.rings):
            raise ValueError(
                f"ring_values has {len(self.ring_values)} rows but there are "
                f"{len(self.rings)} rings"
            )
        for ring, row in zip(self.rings, self.ring_values, strict=True):
            if len(row) != len(self.parts):
                raise ValueError(
                    f"ring_values[{ring!r}] has {len(row)} entries but there are "
                    f"{len(self.parts)} parts"
                )
            for part, value in zip(self.parts, row, strict=True):
                _check_value(value, f"ring_values[{ring!r}][{part!r}]")
        if not 0.0 < self.sweep_degrees <= 360.0:
            raise ValueError("sweep_degrees must sit in (0, 360]")
        if self.scale_max is not None:
            largest = max(self.ring_totals())
            if self.scale_max <= 0.0:
                raise ValueError("scale_max must be positive")
            if self.scale_max < largest:
                raise ValueError(
                    f"scale_max {self.scale_max:g} is below the largest ring total "
                    f"{largest:g}, so that ring would overflow its track"
                )
        try:
            self.center_format.format(1.0)
        except (IndexError, KeyError, ValueError) as exc:
            raise ValueError(
                f"center_format {self.center_format!r} is not usable"
            ) from exc
        self.center_shares()

    def ring_totals(self) -> tuple[float, ...]:
        """Stacked length of every ring, in ``rings`` order."""

        return tuple(math.fsum(row) for row in self.ring_values)

    def effective_scale_max(self) -> float:
        """The shared radial scale: pinned when given, rounded up otherwise."""

        if self.scale_max is not None:
            return float(self.scale_max)
        largest = max(self.ring_totals())
        if largest <= 0.0:
            return 1.0
        step = nice_step(largest)
        return float(math.ceil(largest / step - 1e-9) * step)

    def center_total(self) -> float:
        return math.fsum(self.center)

    def center_shares(self) -> tuple[float, ...]:
        """Centre-pie labels in ``parts`` order; percentages when normalising."""

        total = self.center_total()
        if total <= 0.0:
            raise ValueError("center must sum to a positive total")
        if self.normalize_center:
            return tuple(100.0 * value / total for value in self.center)
        return tuple(float(value) for value in self.center)


def _check_value(value: float, where: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{where} must be a finite number")
    if value < 0.0:
        raise ValueError(f"{where} is negative; parts of a total cannot be negative")


_REFERENCE_PARTS: Final[tuple[str, ...]] = (
    "Oil",
    "Gas",
    "Coal",
    "Biomass",
    "Wind",
    "Solar",
    "Electricity",
)

# Innermost track to outermost track.
_REFERENCE_RINGS: Final[tuple[str, ...]] = (
    "Non-road",
    "Residential fuel",
    "Buildings",
    "Road",
    "Industrial",
    "Electricity",
)

# Percent of terminal energy per fuel, from the labelled centre wedges.
_REFERENCE_CENTER: Final[tuple[float, ...]] = (26.0, 13.0, 52.0, 2.0, 3.0, 2.0, 2.0)

# EJ per (ring, part), digitised from carousel frame 1 on a 0.1 EJ grid.
_REFERENCE_RING_VALUES: Final[tuple[tuple[float, ...], ...]] = (
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1),
    (0.2, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0),
    (1.1, 0.8, 0.4, 0.0, 0.0, 0.0, 0.4),
    (2.8, 0.2, 0.0, 0.0, 0.0, 0.0, 1.2),
    (0.0, 1.0, 4.1, 0.0, 0.0, 0.0, 0.9),
    (0.0, 0.5, 4.0, 0.3, 0.8, 0.5, 1.9),
)

DEFAULT_DATA: Final[PieRingData] = PieRingData.from_nested(
    parts=_REFERENCE_PARTS,
    center=_REFERENCE_CENTER,
    rings=_REFERENCE_RINGS,
    ring_values=_REFERENCE_RING_VALUES,
    scale_max=8.0,
    unit_label="Energy: EJ",
    center_format="{:.0f}%",
    normalize_center=True,
    sweep_degrees=270.0,
    title="2030-BTH",
)


@dataclass(frozen=True, slots=True)
class ChartStyle:
    """Layout tuned to the 2370 x 2490 reference canvas.

    Ring tracks always span ``first_ring_inner`` to ``last_ring_outer``, so
    their width and spacing follow the ring count: the reference six rings come
    out 72 units wide with 24-unit gaps, and a shorter dataset gets thicker
    tracks rather than a half-empty annulus.
    """

    figure_size: tuple[float, float] = (11.85, 12.45)
    x_limits: tuple[float, float] | None = (-1185.0, 1185.0)
    y_limits: tuple[float, float] | None = (-1187.0, 1303.0)
    pie_radius: float = 331.0
    first_ring_inner: float = 475.0
    last_ring_outer: float = 1027.0
    ring_gap_fraction: float = 1.0 / 3.0
    scale_radius: float = 1052.0
    pie_gap_degrees: float = 0.0
    separator_width: float = 2.2
    track_color: str = "#F0F0F0"
    axis_color: str = "#A0A0A0"
    scale_color: str = "#A0A0A0"
    edge_color: str = "#FFFFFF"
    title_y: float = 1191.0
    title_font_size: float = 34.0
    label_font_size: float = 20.0
    pie_font_size: float = 17.0
    legend_font_size: float = 18.5
    tick_font_size: float = 18.0
    legend_x: float = -996.0
    legend_top: float = 967.0
    legend_step: float = 78.0
    legend_swatch: tuple[float, float] = (64.0, 55.0)
    legend_text_gap: float = 40.0
    ring_label_x: float = -48.0
    unit_label_radius: float = 718.0
    min_pie_label_percent: float = 10.0
    pie_label_color: str | None = "black"
    """Preferred ink for the centre-pie labels; ``None`` always picks contrast."""

    def validate(self, *, categories: int, series: int) -> None:
        """``categories`` counts the rings and ``series`` the parts."""

        if min(self.figure_size) <= 0:
            raise ValueError("figure_size must be positive")
        if categories < 2 or series < 2:
            raise ValueError("need at least two rings and two parts")
        if self.pie_radius <= 0:
            raise ValueError("pie_radius must be positive")
        if not self.pie_radius < self.first_ring_inner < self.last_ring_outer:
            raise ValueError(
                "radii must satisfy 0 < pie_radius < first_ring_inner < last_ring_outer"
            )
        if self.ring_gap_fraction < 0:
            raise ValueError("ring_gap_fraction must be non-negative")
        if self.legend_step <= 0:
            raise ValueError("legend_step must be positive")
        width, _ = ring_geometry(self, rings=categories)
        if width <= 0:
            raise ValueError(f"{categories} rings do not fit in the ring band")


DEFAULT_STYLE: Final[ChartStyle] = ChartStyle()


def ring_geometry(style: ChartStyle, *, rings: int) -> tuple[float, float]:
    """Return ``(ring_width, ring_gap)`` for ``rings`` concentric tracks."""

    if rings < 1:
        raise ValueError("rings must be positive")
    band = style.last_ring_outer - style.first_ring_inner
    width = band / (rings + (rings - 1) * style.ring_gap_fraction)
    return width, width * style.ring_gap_fraction


def ring_bounds(style: ChartStyle, *, index: int, rings: int) -> tuple[float, float]:
    """Inner and outer radius of one track, counted from the innermost."""

    width, gap = ring_geometry(style, rings=rings)
    inner = style.first_ring_inner + index * (width + gap)
    return inner, inner + width


def _value_to_theta(value: float, *, scale_max: float, sweep: float) -> float:
    """Matplotlib wedge angle (CCW from +x) for a clockwise value on the scale."""

    return 90.0 - value / scale_max * sweep


def _wedge_thetas(
    start: float, end: float, *, scale_max: float, sweep: float
) -> tuple[float, float]:
    """Return ``(theta1, theta2)`` for the clockwise interval ``[start, end]``."""

    return (
        _value_to_theta(end, scale_max=scale_max, sweep=sweep),
        _value_to_theta(start, scale_max=scale_max, sweep=sweep),
    )


def _radial_line(
    ax: Axes,
    inner: float,
    outer: float,
    theta_deg: float,
    *,
    color: str,
    linewidth: float,
    zorder: float,
) -> None:
    theta = np.deg2rad(theta_deg)
    cosine, sine = np.cos(theta), np.sin(theta)
    ax.plot(
        [inner * cosine, outer * cosine],
        [inner * sine, outer * sine],
        color=color,
        linewidth=linewidth,
        solid_capstyle="butt",
        zorder=zorder,
    )


def _draw_axes(ax: Axes, data: PieRingData, style: ChartStyle) -> None:
    """Draw the rays that bound the sweep, at zero and at the scale maximum."""

    outer = style.scale_radius + 18.0
    inner = style.pie_radius + 3.0
    for theta in (90.0, 90.0 - data.sweep_degrees):
        _radial_line(
            ax,
            inner,
            outer,
            theta,
            color=style.axis_color,
            linewidth=1.2,
            zorder=4,
        )


def _tick_values(scale_max: float) -> tuple[float, ...]:
    """Round tick positions from zero up to and including ``scale_max``."""

    step = nice_step(scale_max)
    count = int(math.floor(scale_max / step + 1e-9))
    ticks = [index * step for index in range(count + 1)]
    if ticks[-1] < scale_max - 1e-9:
        ticks.append(scale_max)
    return tuple(ticks)


def _tick_decimals(values: Sequence[float]) -> int:
    """Fewest decimals that print every tick exactly."""

    for decimals in range(4):
        if all(abs(round(value, decimals) - value) < 1e-9 for value in values):
            return decimals
    return 3


def _draw_scale(ax: Axes, data: PieRingData, style: ChartStyle) -> None:
    radius = style.scale_radius
    scale_max = data.effective_scale_max()
    sweep = data.sweep_degrees
    ax.add_patch(
        Arc(
            (0.0, 0.0),
            2.0 * radius,
            2.0 * radius,
            theta1=_value_to_theta(scale_max, scale_max=scale_max, sweep=sweep),
            theta2=_value_to_theta(0.0, scale_max=scale_max, sweep=sweep),
            color=style.scale_color,
            linewidth=1.35,
            zorder=5,
        )
    )
    ticks = _tick_values(scale_max)
    decimals = _tick_decimals(ticks)
    for value in ticks:
        theta = np.deg2rad(_value_to_theta(value, scale_max=scale_max, sweep=sweep))
        inner = radius - 6.0
        outer = radius + 16.0
        ax.plot(
            [inner * np.cos(theta), outer * np.cos(theta)],
            [inner * np.sin(theta), outer * np.sin(theta)],
            color=style.scale_color,
            linewidth=1.2,
            solid_capstyle="butt",
            zorder=6,
        )
        label_r = radius + 68.0
        x = label_r * np.cos(theta)
        y = label_r * np.sin(theta)
        if value == ticks[0]:
            ha, va = "center", "bottom"
            y += 4.0
        elif value == ticks[-1]:
            ha, va = "right", "center"
        else:
            ha, va = "center", "center"
        ax.text(
            x,
            y,
            f"{value:.{decimals}f}",
            ha=ha,
            va=va,
            fontsize=style.tick_font_size,
            fontweight="bold",
            color="black",
            zorder=8,
        )

    if not data.unit_label:
        return
    end = np.deg2rad(90.0 - sweep)
    ax.text(
        style.unit_label_radius * np.cos(end),
        style.unit_label_radius * np.sin(end),
        data.unit_label,
        ha="center",
        va="bottom",
        fontsize=style.tick_font_size + 1.5,
        fontweight="bold",
        color="black",
        zorder=8,
    )


def _draw_rings(
    ax: Axes,
    data: PieRingData,
    palette: Palette,
    style: ChartStyle,
) -> None:
    n_rings = len(data.rings)
    colors = palette.take(len(data.parts))
    scale_max = data.effective_scale_max()
    sweep = data.sweep_degrees
    track_thetas = _wedge_thetas(0.0, scale_max, scale_max=scale_max, sweep=sweep)

    for index, (ring, row) in enumerate(zip(data.rings, data.ring_values, strict=True)):
        inner, outer = ring_bounds(style, index=index, rings=n_rings)
        width = outer - inner
        ax.add_patch(
            Wedge(
                (0.0, 0.0),
                outer,
                track_thetas[0],
                track_thetas[1],
                width=width,
                facecolor=style.track_color,
                edgecolor="none",
                zorder=2,
            )
        )
        cursor = 0.0
        joints: list[float] = []
        for part_index, value in enumerate(row):
            if value <= 0.0:
                continue
            start, end = cursor, cursor + value
            theta1, theta2 = _wedge_thetas(
                start, end, scale_max=scale_max, sweep=sweep
            )
            ax.add_patch(
                Wedge(
                    (0.0, 0.0),
                    outer,
                    theta1,
                    theta2,
                    width=width,
                    facecolor=colors[part_index],
                    edgecolor="none",
                    zorder=3,
                )
            )
            if cursor > 0.0:
                joints.append(cursor)
            cursor = end
        for joint in joints:
            _radial_line(
                ax,
                inner,
                outer,
                _value_to_theta(joint, scale_max=scale_max, sweep=sweep),
                color=style.edge_color,
                linewidth=style.separator_width,
                zorder=4,
            )
        ax.text(
            style.ring_label_x,
            0.5 * (inner + outer),
            ring,
            ha="right",
            va="center",
            fontsize=style.label_font_size,
            fontweight="bold",
            color="black",
            zorder=8,
        )


# A wedge below this relative luminance needs light ink.  The reference coral
# sits just above the line, which is why its printed shares stay black.
_DARK_WEDGE_CUTOFF: Final[float] = 0.5


def _center_label_ink(color: str, style: ChartStyle) -> str:
    """The styled ink, giving way to a light one over a dark wedge."""

    contrast = readable_text_color(color, cutoff=_DARK_WEDGE_CUTOFF)
    if style.pie_label_color is None or contrast == "#FFFFFF":
        return contrast
    return style.pie_label_color


def _draw_center_pie(
    ax: Axes,
    data: PieRingData,
    palette: Palette,
    style: ChartStyle,
) -> None:
    shares = data.center_shares()
    colors = palette.take(len(data.parts))
    total = math.fsum(shares)
    # Wedges run clockwise from 12 o'clock, largest share first; equal shares
    # keep their order in ``parts``.
    order = sorted(range(len(shares)), key=lambda index: -shares[index])
    gap = 0.5 * style.pie_gap_degrees
    edges: list[float] = []
    cursor = 90.0

    for index in order:
        share = shares[index]
        start, end = cursor, cursor - 360.0 * share / total
        edges.append(start)
        cursor = end
        theta1, theta2 = end + gap, start - gap
        if theta2 <= theta1:
            theta1, theta2 = end, start
        color = colors[index]
        ax.add_patch(
            Wedge(
                (0.0, 0.0),
                style.pie_radius,
                theta1,
                theta2,
                facecolor=color,
                edgecolor="none",
                zorder=5,
            )
        )
        if share < style.min_pie_label_percent:
            continue
        theta = np.deg2rad(0.5 * (theta1 + theta2))
        radius = (0.62 if share >= 40.0 else 0.76) * style.pie_radius
        ax.text(
            radius * np.cos(theta),
            radius * np.sin(theta),
            data.center_format.format(share),
            ha="center",
            va="center",
            fontsize=style.pie_font_size,
            fontweight="bold",
            color=_center_label_ink(color, style),
            zorder=7,
        )

    for theta_deg in edges:
        _radial_line(
            ax,
            0.0,
            style.pie_radius,
            theta_deg,
            color=style.edge_color,
            linewidth=style.separator_width,
            zorder=6,
        )


def _draw_legend(
    ax: Axes,
    data: PieRingData,
    palette: Palette,
    style: ChartStyle,
) -> None:
    swatch_w, swatch_h = style.legend_swatch
    colors = palette.take(len(data.parts))
    for index, part in enumerate(data.parts):
        y = style.legend_top - index * style.legend_step
        ax.add_patch(
            Rectangle(
                (style.legend_x, y - 0.5 * swatch_h),
                swatch_w,
                swatch_h,
                facecolor=colors[index],
                edgecolor="none",
                zorder=8,
            )
        )
        ax.text(
            style.legend_x + swatch_w + style.legend_text_gap,
            y,
            part,
            ha="left",
            va="center",
            fontsize=style.legend_font_size,
            fontweight="bold",
            color="black",
            zorder=8,
        )


def _content_bounds(
    data: PieRingData, style: ChartStyle
) -> tuple[float, float, float, float]:
    """Bounding box ``(left, right, bottom, top)`` of every drawn anchor."""

    reach = style.scale_radius + 68.0
    left, right = -reach, reach
    bottom, top = -reach, reach

    swatch_width, swatch_height = style.legend_swatch
    rows = len(data.parts)
    left = min(left, style.legend_x)
    right = max(right, style.legend_x + swatch_width)
    top = max(top, style.legend_top + 0.5 * swatch_height)
    bottom = min(
        bottom,
        style.legend_top - (rows - 1) * style.legend_step - 0.5 * swatch_height,
    )
    if data.title:
        top = max(top, style.title_y)
    return left, right, bottom, top


def create_figure(
    palette: Palette = PALETTES[0],
    data: PieRingData = DEFAULT_DATA,
    style: ChartStyle = DEFAULT_STYLE,
) -> Figure:
    """Create the centre pie with concentric rings without writing it to disk."""

    data.validate()
    style.validate(categories=len(data.rings), series=len(data.parts))

    left, right, bottom, top = _content_bounds(data, style)
    x_limits = resolve_limits(
        style.x_limits, left, right, include_zero=False, snap=False
    )
    y_limits = resolve_limits(
        style.y_limits, bottom, top, include_zero=False, snap=False
    )

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
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        ax.set_aspect("equal", adjustable="box")
        ax.set_axis_off()
        ax.set_facecolor("white")

        _draw_rings(ax, data, palette, style)
        _draw_center_pie(ax, data, palette, style)
        _draw_axes(ax, data, style)
        _draw_scale(ax, data, style)
        _draw_legend(ax, data, palette, style)
        if data.title:
            ax.text(
                0.0,
                style.title_y,
                data.title,
                ha="center",
                va="bottom",
                fontsize=style.title_font_size,
                fontweight="bold",
                color="black",
                zorder=9,
            )

    return figure


def main(argv: Sequence[str] | None = None) -> int:
    return run_cli(sys.modules[__name__], argv)


if __name__ == "__main__":
    raise SystemExit(main())
