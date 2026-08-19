"""Pseudo-3D contribution pie: one shaded wedge per category.

``DEFAULT_DATA`` holds the fifteen environmental factors of a Xiaohongshu
carousel that repeats the same contribution pie 18 times, changing only the
palette.  The percentages were digitised from the labelled wedges; they sum to
100.2, so they are carried as ready-made percentages (``normalize=False``)
rather than being rescaled.  The source post does not name the underlying
table; replace ``DEFAULT_DATA`` when plotting another dataset.

Wedges run counterclockwise from ``start_angle`` (12 o'clock in the reference)
in ``categories`` order, and both the wedge angles and the printed labels are
read from :meth:`Pie3DData.shares`, so geometry and text cannot disagree.
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
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch, Polygon
from numpy.typing import NDArray

from ..contract import DataKind, Extent, Feature, Geometry, TemplateSpec
from ..palettes import Palette, readable_text_color
from ..render import resolve_limits, run_cli

SPEC: Final[TemplateSpec] = TemplateSpec(
    template_id="pie-3d",
    title="Pseudo-3D contribution pie",
    summary=(
        "A pie chart extruded into a shallow disc with shaded side walls, one "
        "wedge per category, labelled with each category's share of the total."
    ),
    kinds=(DataKind.PARTS_OF_WHOLE,),
    geometry=Geometry.CIRCULAR,
    categories=Extent(2, 20),
    series=Extent(1, 1),
    builder="Pie3DData.from_shares",
    data_contract=(
        "One non-negative value per category, read as a share of their total. "
        "Values may be raw magnitudes or percentages that already sum to 100."
    ),
    good_for=(
        "variance or contribution decompositions that sum to a whole",
        "presentation figures where the composition is the whole message",
        "a single composition with no grouping dimension",
    ),
    avoid_when=(
        "values can be negative, which has no share interpretation",
        "readers must compare similar slices precisely, where a bar chart wins",
        "more than about 20 categories, where wedges and labels collide",
    ),
    requires=(Feature.NON_NEGATIVE,),
    affinities=(
        (Feature.SUMS_TO_100, 13.0),
        (Feature.SINGLE_SERIES, 8.0),
        (Feature.MANY_CATEGORIES, -8.0),
    ),
    default_dpi=200,
    reference="Digitised from a Xiaohongshu carousel; percentages were read off the image.",
)


# Colours sampled from the left-hand legend swatches in carousel frames 1-18.
PALETTES: Final[tuple[Palette, ...]] = (
    Palette(
        "navy-peach-maroon",
        (
            "#2D4471",
            "#3E6AAF",
            "#518FC9",
            "#77B5DB",
            "#A5CEE3",
            "#CBDEF1",
            "#E1ECF7",
            "#F8FBFF",
            "#FEF3E5",
            "#F5D0BA",
            "#F1987D",
            "#EB5C4C",
            "#E13F36",
            "#CD2D2F",
            "#952222",
        ),
    ),
    Palette(
        "forest-mint-coral",
        (
            "#16542F",
            "#18783C",
            "#369556",
            "#50B26B",
            "#7FC981",
            "#A9DCA3",
            "#CBEBC4",
            "#E8F6E2",
            "#F7FCF6",
            "#FEF5F1",
            "#FDE3D6",
            "#FCC0A8",
            "#FB9C7D",
            "#FB775B",
            "#E03F39",
        ),
    ),
    Palette(
        "steel-cream-amber",
        (
            "#1D4277",
            "#1E60A6",
            "#337EBA",
            "#529BC9",
            "#78B5DB",
            "#A7CEE3",
            "#CBDEF1",
            "#E1ECF7",
            "#F8FBFF",
            "#FFF5ED",
            "#FEE8D2",
            "#FCD4A9",
            "#FCB56E",
            "#F27627",
            "#DB5717",
        ),
    ),
    Palette(
        "violet-mist-azure",
        (
            "#4F1688",
            "#633997",
            "#8178B6",
            "#A5A3CD",
            "#C2C2DF",
            "#DEDDEC",
            "#F0F0F5",
            "#FCFCFE",
            "#FEF8FC",
            "#EFE9F4",
            "#D4D5E8",
            "#ADC3DF",
            "#7FB0D4",
            "#479AC5",
            "#1B7CB6",
        ),
    ),
    Palette(
        "crimson-blush-cyan",
        (
            "#731622",
            "#AD242A",
            "#CD2D2F",
            "#EF4C3E",
            "#FB775B",
            "#FB9B7D",
            "#FCC0A9",
            "#FDE3D6",
            "#FEF5F2",
            "#F7FCF1",
            "#E3F5DE",
            "#D0ECC8",
            "#B1E0BC",
            "#87D0CA",
            "#5DB9D6",
        ),
    ),
    Palette(
        "graphite-lilac-magenta",
        (
            "#161616",
            "#383838",
            "#616161",
            "#7F7F7F",
            "#9F9F9F",
            "#C3C3C3",
            "#DCDCDC",
            "#F1F1F1",
            "#FFFFFF",
            "#F7F5FA",
            "#E9E3F0",
            "#D8BFDD",
            "#CE9DCC",
            "#E272B7",
            "#E93B94",
        ),
    ),
    Palette(
        "spectral-rainbow",
        (
            "#541661",
            "#573780",
            "#515592",
            "#466E97",
            "#3D8398",
            "#339B95",
            "#34B08F",
            "#54C47B",
            "#86D45F",
            "#C3E239",
            "#FDE837",
            "#FDD331",
            "#FDA72B",
            "#FE6817",
            "#FD1617",
        ),
    ),
    Palette(
        "twilight-sunset",
        (
            "#16161A",
            "#272045",
            "#4B247C",
            "#6F28A6",
            "#923576",
            "#B7465D",
            "#D75F45",
            "#EE841A",
            "#F7AE24",
            "#FBD346",
            "#FCFFAC",
            "#D5F2AC",
            "#F0FBBB",
            "#FEC996",
            "#FEA68C",
        ),
    ),
    Palette(
        "navy-ice-amber",
        (
            "#16345D",
            "#174572",
            "#165DA1",
            "#1972D0",
            "#46A1FD",
            "#72B9FC",
            "#A2D1FD",
            "#D0E7FE",
            "#FFFFFF",
            "#FEE7D0",
            "#FED0A2",
            "#FEB873",
            "#FDA246",
            "#D17218",
            "#A15B16",
        ),
    ),
    Palette(
        "violet-sand-ochre",
        (
            "#3E165A",
            "#633993",
            "#8B7FB0",
            "#B9B3D6",
            "#DBDCED",
            "#F8F8F8",
            "#FDE3BD",
            "#FCBE70",
            "#E28D28",
            "#BA661D",
            "#8A4D1E",
            "#624319",
            "#966020",
            "#C48C3F",
            "#E2C789",
        ),
    ),
    Palette(
        "sage-rose-sky",
        (
            "#2E5342",
            "#40775F",
            "#529A7A",
            "#60BD92",
            "#80CBA6",
            "#9ED8B9",
            "#BDE6CC",
            "#DAF4DF",
            "#F2FAF1",
            "#B0DDDF",
            "#5586A5",
            "#304665",
            "#E74A56",
            "#97CEE8",
            "#34A6C1",
        ),
    ),
    Palette(
        "ember-gold-umber",
        (
            "#181D30",
            "#481C2B",
            "#761A25",
            "#A5181C",
            "#D31616",
            "#DE4219",
            "#EA6A1C",
            "#F5961B",
            "#F9AB1C",
            "#FDBF1D",
            "#FEF5B6",
            "#456A73",
            "#A73C3D",
            "#612023",
            "#371E18",
        ),
    ),
    Palette(
        "mauve-stone",
        (
            "#35354B",
            "#5A5D77",
            "#A396A1",
            "#CEB4AF",
            "#F3EBE7",
            "#DAD0C7",
            "#EFEFEF",
            "#F6EDE4",
            "#E6D8CE",
            "#D8C3B7",
            "#BEBDAB",
            "#ADAD96",
            "#787C69",
            "#50524A",
            "#41423B",
        ),
    ),
    Palette(
        "cyan-taupe",
        (
            "#1683BC",
            "#179FCC",
            "#17B9DC",
            "#57CFE6",
            "#9BE2F0",
            "#B4EAF5",
            "#CEF2F9",
            "#EFEFEB",
            "#DAD0C7",
            "#F6ECE4",
            "#E5D9CE",
            "#DAC3B7",
            "#A47355",
            "#8A634A",
            "#4C3424",
        ),
    ),
    Palette(
        "qualitative-earth",
        (
            "#164158",
            "#DA3A3B",
            "#F68A18",
            "#FDC557",
            "#EBE4BE",
            "#98CEE6",
            "#32A7C2",
            "#184257",
            "#FEBD1B",
            "#FB8F19",
            "#F38F8D",
            "#F7CEC8",
            "#F8EFE4",
            "#8FADA7",
            "#F5C36D",
        ),
    ),
    Palette(
        "lime-gold",
        (
            "#165A37",
            "#177117",
            "#167F19",
            "#168B17",
            "#49B717",
            "#7DE218",
            "#A6F22E",
            "#D0FE44",
            "#FEFE51",
            "#FEF043",
            "#FEDD16",
            "#FEC916",
            "#FDB218",
            "#FE9916",
            "#FE8D18",
        ),
    ),
    Palette(
        "slate-crimson-gold",
        (
            "#3D3F52",
            "#97A2B7",
            "#EDF4F6",
            "#F0364C",
            "#DC1A3C",
            "#30384A",
            "#516983",
            "#8497B1",
            "#E3E4DF",
            "#171E29",
            "#16304D",
            "#184672",
            "#FDC918",
            "#FEDA1F",
            "#FFFFFF",
        ),
    ),
    Palette(
        "gray-rose",
        (
            "#34373B",
            "#464A52",
            "#595F64",
            "#798089",
            "#B5BBC3",
            "#E1E4E9",
            "#EAEEF0",
            "#F9FAFB",
            "#FFFFFF",
            "#FEF2F4",
            "#FFD0D9",
            "#FEB9C6",
            "#FD99AA",
            "#FE8099",
            "#FE5C7A",
        ),
    ),
)


@dataclass(frozen=True, slots=True)
class Pie3DData:
    """One non-negative value per category, read as a share of their total."""

    categories: tuple[str, ...]
    values: tuple[float, ...]
    value_label: str = "Share"
    value_format: str = "{:.1f}%"
    normalize: bool = True
    start_angle: float = 90.0
    title: str = ""

    @classmethod
    def from_shares(
        cls,
        *,
        categories: Sequence[str],
        values: Sequence[float | None],
        value_label: str = "Share",
        value_format: str = "{:.1f}%",
        normalize: bool = True,
        start_angle: float = 90.0,
        title: str = "",
    ) -> "Pie3DData":
        """Build from one value per category, in drawing order.

        ``normalize=True`` reads ``values`` as raw magnitudes and turns them
        into percentages of their total; ``normalize=False`` declares that they
        already are percentages summing to 100.  ``None`` entries become NaN and
        are rejected, because a composition cannot leave one share unstated.
        """

        built = cls(
            categories=tuple(str(name) for name in categories),
            values=tuple(
                float("nan") if value is None else float(value) for value in values
            ),
            value_label=value_label,
            value_format=value_format,
            normalize=bool(normalize),
            start_angle=float(start_angle),
            title=title,
        )
        built.validate()
        return built

    def validate(self) -> None:
        if len(self.categories) < 2:
            raise ValueError("need at least two categories")
        if len(set(self.categories)) != len(self.categories):
            raise ValueError("category labels must be unique")
        if len(self.values) != len(self.categories):
            raise ValueError(
                f"values has {len(self.values)} entries but there are "
                f"{len(self.categories)} categories"
            )
        for category, value in zip(self.categories, self.values, strict=True):
            if not math.isfinite(value):
                raise ValueError(f"values[{category!r}] must be a finite number")
            if value < 0.0:
                raise ValueError(
                    f"values[{category!r}] is negative; a share of a total has no "
                    "negative interpretation"
                )
        if not math.isfinite(self.start_angle):
            raise ValueError("start_angle must be finite")
        try:
            self.value_format.format(1.0)
        except (IndexError, KeyError, ValueError) as exc:
            raise ValueError(f"value_format {self.value_format!r} is not usable") from exc
        self.shares()

    def total(self) -> float:
        return math.fsum(self.values)

    def shares(self) -> tuple[float, ...]:
        """Percentages in ``categories`` order, driving angles and labels alike."""

        total = self.total()
        if total <= 0.0:
            raise ValueError("values must sum to a positive total")
        if self.normalize:
            return tuple(100.0 * value / total for value in self.values)
        if abs(total - 100.0) > 1.0:
            raise ValueError(
                f"normalize=False reads values as percentages, but they sum to "
                f"{total:.3f} rather than 100"
            )
        return tuple(float(value) for value in self.values)


_REFERENCE_CATEGORIES: Final[tuple[str, ...]] = (
    "GSR",
    "LST",
    "NDVI",
    "PRE",
    "ST",
    "ELEVATION",
    "SLOPE",
    "NDSI",
    "NIR",
    "NDWI",
    "WS",
    "NS",
    "NSC",
    "NSD",
    "ASPECT",
)

# Digitised from the labelled wedges of carousel frame 1; they sum to 100.2.
_REFERENCE_SHARES: Final[tuple[float, ...]] = (
    5.5,
    7.6,
    5.5,
    3.8,
    3.0,
    18.5,
    2.3,
    11.8,
    3.7,
    22.8,
    5.3,
    1.5,
    0.4,
    7.7,
    0.8,
)

DEFAULT_DATA: Final[Pie3DData] = Pie3DData.from_shares(
    categories=_REFERENCE_CATEGORIES,
    values=_REFERENCE_SHARES,
    value_label="Contribution",
    value_format="{:.1f}%",
    normalize=False,
    start_angle=90.0,
    title="HHH YSHJXLXXZTHDSMYCSKYA (3D)",
)


@dataclass(frozen=True, slots=True)
class ChartStyle:
    """Layout tuned to the 2019 x 1350 reference canvas.

    The legend grows downward by one ``legend_step`` per category, so the
    pinned limits are only kept while the whole drawing still fits inside
    them; a longer legend widens the view instead of being clipped.
    """

    figure_size: tuple[float, float] = (10.095, 6.75)
    x_limits: tuple[float, float] | None = (-1.72, 1.55)
    y_limits: tuple[float, float] | None = (-1.28, 1.42)
    center: tuple[float, float] = (0.24, 0.08)
    radius: float = 0.98
    aspect: float = 0.50
    height: float = 0.17
    explode: float = 0.010
    gap_degrees: float = 0.70
    side_shade: float = 0.86
    edge_width: float = 1.15
    edge_color: str = "#FFFFFF"
    label_radius: float = 1.13
    small_label_radius: float = 1.32
    small_slice_percent: float = 2.4
    label_color: str = "black"
    """Ink for labels parked outside the disc; on-wedge labels pick contrast."""

    title_x: float = 0.22
    title_y: float = 1.30
    title_font_size: float = 16.5
    label_font_size: float = 11.0
    legend_font_size: float = 11.5
    legend_x: float = -1.62
    legend_top: float = 1.08
    legend_step: float = 0.152
    legend_swatch: tuple[float, float] = (0.168, 0.058)
    legend_text_gap: float = 0.045
    show_value_label: bool = False
    """Print ``data.value_label`` as a heading above the legend column."""

    leader_color: str = "#B8B8B8"

    def validate(self, *, categories: int, series: int) -> None:
        if min(self.figure_size) <= 0:
            raise ValueError("figure_size must be positive")
        if self.radius <= 0 or self.aspect <= 0 or self.height < 0:
            raise ValueError("radius, aspect, and height must be positive")
        if not 0 <= self.explode < 0.5:
            raise ValueError("explode must be in [0, 0.5)")
        if self.legend_step <= 0:
            raise ValueError("legend_step must be positive")
        if series != 1:
            raise ValueError("a pie draws one composition, so series must be 1")
        if categories < 2:
            raise ValueError("need at least two categories")
        if self.gap_degrees * categories >= 360.0:
            raise ValueError(
                f"{categories} gaps of {self.gap_degrees:.2f} degrees leave no room "
                "for wedges; lower gap_degrees"
            )


DEFAULT_STYLE: Final[ChartStyle] = ChartStyle()


def _shade(color: str, factor: float) -> tuple[float, float, float]:
    red, green, blue = to_rgb(color)
    return (red * factor, green * factor, blue * factor)


def _slice_angles(shares: Sequence[float], start_angle: float) -> NDArray[np.float64]:
    """Monotonically increasing wedge edges, counterclockwise from ``start_angle``."""

    total = float(np.sum(shares))
    widths = 2.0 * np.pi * np.asarray(shares, dtype=float) / total
    start = float(np.radians(start_angle))
    return np.concatenate(([start], start + np.cumsum(widths)))


def _wedge_vertices(
    theta1: float,
    theta2: float,
    radius_x: float,
    radius_y: float,
    center: tuple[float, float],
    *,
    samples: int = 96,
) -> NDArray[np.float64]:
    thetas = np.linspace(theta1, theta2, samples)
    xs = np.concatenate(([center[0]], center[0] + radius_x * np.cos(thetas), [center[0]]))
    ys = np.concatenate(([center[1]], center[1] + radius_y * np.sin(thetas), [center[1]]))
    return np.column_stack((xs, ys))


def _front_arc(
    theta1: float,
    theta2: float,
    *,
    samples: int = 72,
    horizon: float = 0.18,
) -> NDArray[np.float64] | None:
    thetas = np.linspace(theta1, theta2, samples)
    front = thetas[np.sin(thetas) <= horizon]
    if front.size < 2:
        return None
    return front


def _draw_side_wall(
    ax: Axes,
    theta1: float,
    theta2: float,
    center: tuple[float, float],
    color: tuple[float, float, float],
    style: ChartStyle,
) -> None:
    arc = _front_arc(theta1, theta2)
    if arc is None:
        return
    cx, cy = center
    rx = style.radius
    ry = style.radius * style.aspect
    top_x = cx + rx * np.cos(arc)
    top_y = cy + ry * np.sin(arc)
    bot_x = top_x
    bot_y = top_y - style.height
    xs = np.concatenate([top_x, bot_x[::-1]])
    ys = np.concatenate([top_y, bot_y[::-1]])
    ax.add_patch(
        Polygon(
            np.column_stack((xs, ys)),
            closed=True,
            facecolor=color,
            edgecolor=style.edge_color,
            linewidth=0.55,
            joinstyle="round",
            zorder=2.0 + (1.0 - np.sin(0.5 * (theta1 + theta2))) * 0.1,
        )
    )


def _draw_radial_wall(
    ax: Axes,
    theta: float,
    center: tuple[float, float],
    color: tuple[float, float, float],
    style: ChartStyle,
) -> None:
    if np.sin(theta) > 0.25:
        return
    cx, cy = center
    rx = style.radius
    ry = style.radius * style.aspect
    outer = (cx + rx * np.cos(theta), cy + ry * np.sin(theta))
    inner = (cx, cy)
    verts = [
        inner,
        outer,
        (outer[0], outer[1] - style.height),
        (inner[0], inner[1] - style.height),
    ]
    ax.add_patch(
        Polygon(
            verts,
            closed=True,
            facecolor=color,
            edgecolor=style.edge_color,
            linewidth=0.4,
            zorder=1.8,
        )
    )


def _draw_top_face(
    ax: Axes,
    theta1: float,
    theta2: float,
    center: tuple[float, float],
    color: str,
    style: ChartStyle,
) -> None:
    vertices = _wedge_vertices(
        theta1,
        theta2,
        style.radius,
        style.radius * style.aspect,
        center,
    )
    ax.add_patch(
        Polygon(
            vertices,
            closed=True,
            facecolor=color,
            edgecolor=style.edge_color,
            linewidth=style.edge_width,
            joinstyle="round",
            zorder=3.2,
        )
    )


def _label_position(
    theta: float,
    share: float,
    center: tuple[float, float],
    style: ChartStyle,
) -> tuple[float, float, float]:
    """Return ``(x, y, radius_factor)`` for one wedge's label anchor."""

    factor = (
        style.small_label_radius
        if share < style.small_slice_percent
        else style.label_radius
    )
    cx, cy = style.center
    x = cx + factor * style.radius * np.cos(theta)
    y = cy + factor * style.radius * style.aspect * np.sin(theta)
    # Keep the pie's visual centre (including explode) out of the way.
    x += (center[0] - cx) * 0.35
    y += (center[1] - cy) * 0.35
    return x, y, factor


# A label only counts as sitting on a wedge once it is this far inside the
# silhouette; one straddling the edge keeps the styled ink so that the part of
# it hanging over the canvas stays visible.
_ON_DISC_MARGIN: Final[float] = 0.88


def _on_disc(
    x: float, y: float, center: tuple[float, float], style: ChartStyle
) -> bool:
    """True when a point lands well inside a wedge's top face or side wall."""

    limit = _ON_DISC_MARGIN * _ON_DISC_MARGIN
    dx = (x - center[0]) / style.radius
    ry = style.radius * style.aspect
    dy = (y - center[1]) / ry
    if dx * dx + dy * dy <= limit:
        return True
    dy_wall = (y - center[1] + style.height) / ry
    return dx * dx + dy_wall * dy_wall <= limit


def _label_ink(
    x: float,
    y: float,
    center: tuple[float, float],
    color: str,
    style: ChartStyle,
) -> str:
    """Ink that stays legible wherever the label ended up."""

    if not _on_disc(x, y, center, style):
        return style.label_color
    # On a wedge the fill decides: keep the styled ink unless it needs a light one.
    contrast = readable_text_color(color)
    return contrast if contrast == "#FFFFFF" else style.label_color


def _draw_labels(
    ax: Axes,
    slices: Sequence[tuple[float, tuple[float, float], float, str]],
    style: ChartStyle,
    *,
    value_format: str,
) -> None:
    """``slices`` is ``(mid_angle, wedge_center, share, color)``."""

    placed: list[tuple[float, float]] = []
    for mid, center, share, color in slices:
        x, y, factor = _label_position(mid, share, center, style)
        # Walk a crowded label outward until it stands clear of its neighbours.
        for _ in range(6):
            if not any(
                abs(x - px) < 0.16 and abs(y - py) < 0.10 for px, py in placed
            ):
                break
            y += 0.09 if y >= 0 else -0.09
            x += 0.04 * np.sign(np.cos(mid) or 1.0)
        placed.append((x, y))
        ink = _label_ink(x, y, center, color, style)
        ax.text(
            x,
            y,
            value_format.format(share),
            ha="center",
            va="center",
            fontsize=style.label_font_size,
            color=ink,
            zorder=5,
        )
        if factor > style.label_radius:
            edge_x = center[0] + style.radius * 0.98 * np.cos(mid)
            edge_y = center[1] + style.radius * style.aspect * 0.98 * np.sin(mid)
            ax.plot(
                [edge_x, x],
                [edge_y, y],
                color=style.leader_color,
                linewidth=0.7,
                solid_capstyle="round",
                zorder=4,
            )


def _draw_legend(
    ax: Axes,
    data: Pie3DData,
    palette: Palette,
    style: ChartStyle,
) -> None:
    width, height = style.legend_swatch
    colors = palette.take(len(data.categories))
    if style.show_value_label and data.value_label:
        ax.text(
            style.legend_x,
            style.legend_top + style.legend_step,
            data.value_label,
            ha="left",
            va="center",
            fontsize=style.legend_font_size,
            fontweight="bold",
            color="black",
            zorder=5,
        )
    for index, category in enumerate(data.categories):
        y = style.legend_top - index * style.legend_step
        ax.add_patch(
            FancyBboxPatch(
                (style.legend_x, y - 0.5 * height),
                width,
                height,
                boxstyle="square,pad=0",
                facecolor=colors[index],
                edgecolor="none",
                zorder=5,
            )
        )
        ax.text(
            style.legend_x + width + style.legend_text_gap,
            y,
            category,
            ha="left",
            va="center",
            fontsize=style.legend_font_size,
            color="black",
            zorder=5,
        )


def _draw_pie(
    ax: Axes,
    data: Pie3DData,
    palette: Palette,
    style: ChartStyle,
) -> None:
    shares = data.shares()
    colors = palette.take(len(data.categories))
    edges = _slice_angles(shares, data.start_angle)
    gap = np.deg2rad(style.gap_degrees)
    cx, cy = style.center
    slices: list[tuple[float, float, float, tuple[float, float], float, str]] = []

    for index, (share, color) in enumerate(zip(shares, colors, strict=True)):
        theta1 = edges[index] + 0.5 * gap
        theta2 = edges[index + 1] - 0.5 * gap
        if theta2 <= theta1:
            theta1, theta2 = edges[index], edges[index + 1]
        mid = 0.5 * (theta1 + theta2)
        shift = (
            cx + style.explode * np.cos(mid),
            cy + style.explode * style.aspect * np.sin(mid),
        )
        slices.append((mid, theta1, theta2, shift, share, color))

    painter = sorted(slices, key=lambda item: np.sin(item[0]), reverse=True)
    for _mid, theta1, theta2, center, _share, color in painter:
        wall = _shade(color, style.side_shade)
        _draw_radial_wall(ax, theta1, center, wall, style)
        _draw_radial_wall(ax, theta2, center, wall, style)
        _draw_side_wall(ax, theta1, theta2, center, wall, style)

    for _mid, theta1, theta2, center, _share, color in painter:
        _draw_top_face(ax, theta1, theta2, center, color, style)

    _draw_labels(
        ax,
        [(mid, center, share, color) for mid, _t1, _t2, center, share, color in slices],
        style,
        value_format=data.value_format,
    )


def _content_bounds(
    data: Pie3DData, style: ChartStyle
) -> tuple[float, float, float, float]:
    """Bounding box ``(left, right, bottom, top)`` of every drawn anchor."""

    cx, cy = style.center
    reach = max(style.label_radius, style.small_label_radius) * style.radius
    left = cx - style.explode - reach
    right = cx + style.explode + reach
    top = cy + (style.explode + reach) * style.aspect
    bottom = cy - (style.explode + reach) * style.aspect - style.height

    swatch_width, swatch_height = style.legend_swatch
    legend_top = style.legend_top + 0.5 * swatch_height
    if style.show_value_label and data.value_label:
        legend_top += style.legend_step
    rows = len(data.categories)
    left = min(left, style.legend_x)
    right = max(right, style.legend_x + swatch_width)
    top = max(top, legend_top)
    bottom = min(
        bottom,
        style.legend_top - (rows - 1) * style.legend_step - 0.5 * swatch_height,
    )
    if data.title:
        top = max(top, style.title_y)
    return left, right, bottom, top


def create_figure(
    palette: Palette = PALETTES[0],
    data: Pie3DData = DEFAULT_DATA,
    style: ChartStyle = DEFAULT_STYLE,
) -> Figure:
    """Create the pseudo-3D contribution pie without writing it to disk."""

    data.validate()
    style.validate(categories=len(data.categories), series=1)

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

        if data.title:
            ax.text(
                style.title_x,
                style.title_y,
                data.title,
                ha="center",
                va="center",
                fontsize=style.title_font_size,
                fontweight="bold",
                color="black",
                zorder=6,
            )
        _draw_legend(ax, data, palette, style)
        _draw_pie(ax, data, palette, style)

    return figure


def main(argv: Sequence[str] | None = None) -> int:
    return run_cli(sys.modules[__name__], argv)


if __name__ == "__main__":
    raise SystemExit(main())
