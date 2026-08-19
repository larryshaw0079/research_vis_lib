"""Reproduce the 15-factor 3D pie chart from the Xiaohongshu reference.

The carousel repeats the same contribution pie 18 times, changing only the
colour palette.  Slices run counterclockwise from 12 o'clock in
:data:`CATEGORIES` order.  Percentages were digitised from the labelled
wedges.  The source post does not name the underlying table; replace
:data:`DEFAULT_DATA` when using another dataset.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch, Polygon
from numpy.typing import NDArray


CATEGORIES: Final[tuple[str, ...]] = (
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


@dataclass(frozen=True, slots=True)
class Palette:
    """Fifteen slice colours in :data:`CATEGORIES` order."""

    name: str
    colors: tuple[str, ...]

    def for_category(self, category: str) -> str:
        try:
            return self.colors[CATEGORIES.index(category)]
        except ValueError as exc:
            raise KeyError(f"unsupported category: {category}") from exc

    def validate(self) -> None:
        if len(self.colors) != len(CATEGORIES):
            raise ValueError(
                f"palette {self.name!r} must have {len(CATEGORIES)} colours"
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
    """Category labels and contribution percentages."""

    categories: tuple[str, ...]
    percentages: Mapping[str, float]
    title: str = "HHH YSHJXLXXZTHDSMYCSKYA (3D)"

    def values(self) -> tuple[float, ...]:
        return tuple(self.percentages[name] for name in self.categories)

    def validate(self) -> None:
        if self.categories != CATEGORIES:
            raise ValueError("categories must match the canonical fifteen-factor order")
        if set(self.percentages) != set(self.categories):
            raise ValueError("percentages must contain one value per category")
        if any(value < 0 for value in self.percentages.values()):
            raise ValueError("percentages must be non-negative")
        if sum(self.percentages.values()) <= 0:
            raise ValueError("percentages must sum to a positive total")


@dataclass(frozen=True, slots=True)
class ChartStyle:
    """Layout tuned to the 2019 × 1350 reference canvas."""

    figure_size: tuple[float, float] = (10.095, 6.75)
    x_limits: tuple[float, float] = (-1.72, 1.55)
    y_limits: tuple[float, float] = (-1.28, 1.42)
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
    title_y: float = 1.30
    title_font_size: float = 16.5
    label_font_size: float = 11.0
    legend_font_size: float = 11.5
    legend_x: float = -1.62
    legend_top: float = 1.08
    legend_step: float = 0.152
    legend_swatch: tuple[float, float] = (0.168, 0.058)
    leader_color: str = "#B8B8B8"

    def validate(self) -> None:
        if min(self.figure_size) <= 0:
            raise ValueError("figure_size must be positive")
        if self.radius <= 0 or self.aspect <= 0 or self.height < 0:
            raise ValueError("radius, aspect, and height must be positive")
        if not 0 <= self.explode < 0.5:
            raise ValueError("explode must be in [0, 0.5)")


DEFAULT_DATA: Final[Pie3DData] = Pie3DData(
    categories=CATEGORIES,
    percentages={
        "GSR": 5.5,
        "LST": 7.6,
        "NDVI": 5.5,
        "PRE": 3.8,
        "ST": 3.0,
        "ELEVATION": 18.5,
        "SLOPE": 2.3,
        "NDSI": 11.8,
        "NIR": 3.7,
        "NDWI": 22.8,
        "WS": 5.3,
        "NS": 1.5,
        "NSC": 0.4,
        "NSD": 7.7,
        "ASPECT": 0.8,
    },
)

DEFAULT_STYLE: Final[ChartStyle] = ChartStyle()


def _shade(color: str, factor: float) -> tuple[float, float, float]:
    red, green, blue = to_rgb(color)
    return (red * factor, green * factor, blue * factor)


def _slice_angles(values: Sequence[float]) -> NDArray[np.float64]:
    """Return monotonically increasing edges, starting at +y and running CCW."""

    total = float(np.sum(values))
    widths = 2.0 * np.pi * np.asarray(values, dtype=float) / total
    edges = np.concatenate(([0.5 * np.pi], 0.5 * np.pi + np.cumsum(widths)))
    return edges


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
    percent: float,
    center: tuple[float, float],
    style: ChartStyle,
) -> tuple[float, float, bool]:
    radius = (
        style.small_label_radius
        if percent < style.small_slice_percent
        else style.label_radius
    )
    cx, cy = style.center
    x = cx + radius * style.radius * np.cos(theta)
    y = cy + radius * style.radius * style.aspect * np.sin(theta)
    # Keep the pie's visual centre (including explode) out of the way.
    x += (center[0] - cx) * 0.35
    y += (center[1] - cy) * 0.35
    return x, y, percent < style.small_slice_percent


def _draw_labels(
    ax: Axes,
    slices: Sequence[tuple[float, float, float, tuple[float, float], float]],
    style: ChartStyle,
) -> None:
    """``slices`` is ``(mid, theta1, theta2, center, percent)``."""

    placed: list[tuple[float, float]] = []
    for mid, _theta1, _theta2, center, percent in slices:
        x, y, is_small = _label_position(mid, percent, center, style)
        for px, py in placed:
            if abs(x - px) < 0.16 and abs(y - py) < 0.10:
                y += 0.09 if y >= 0 else -0.09
                x += 0.04 * np.sign(np.cos(mid) or 1.0)
        placed.append((x, y))
        label = f"{percent:.1f}%"
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=style.label_font_size,
            color="black",
            zorder=5,
        )
        if is_small:
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


def _draw_legend(ax: Axes, palette: Palette, style: ChartStyle) -> None:
    width, height = style.legend_swatch
    for index, category in enumerate(CATEGORIES):
        y = style.legend_top - index * style.legend_step
        ax.add_patch(
            FancyBboxPatch(
                (style.legend_x, y - 0.5 * height),
                width,
                height,
                boxstyle="square,pad=0",
                facecolor=palette.for_category(category),
                edgecolor="none",
                zorder=5,
            )
        )
        ax.text(
            style.legend_x + width + 0.045,
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
    values = data.values()
    colors = tuple(palette.for_category(name) for name in data.categories)
    edges = _slice_angles(values)
    gap = np.deg2rad(style.gap_degrees)
    cx, cy = style.center
    slices: list[tuple[float, float, float, tuple[float, float], float, str]] = []

    for index, (value, color) in enumerate(zip(values, colors)):
        theta1 = edges[index] + 0.5 * gap
        theta2 = edges[index + 1] - 0.5 * gap
        if theta2 <= theta1:
            theta1, theta2 = edges[index], edges[index + 1]
        mid = 0.5 * (theta1 + theta2)
        shift = (
            cx + style.explode * np.cos(mid),
            cy + style.explode * style.aspect * np.sin(mid),
        )
        slices.append((mid, theta1, theta2, shift, value, color))

    painter = sorted(slices, key=lambda item: np.sin(item[0]), reverse=True)
    for mid, theta1, theta2, center, _value, color in painter:
        wall = _shade(color, style.side_shade)
        _draw_radial_wall(ax, theta1, center, wall, style)
        _draw_radial_wall(ax, theta2, center, wall, style)
        _draw_side_wall(ax, theta1, theta2, center, wall, style)

    for mid, theta1, theta2, center, _value, color in painter:
        _draw_top_face(ax, theta1, theta2, center, color, style)

    _draw_labels(
        ax,
        [(mid, t1, t2, center, value) for mid, t1, t2, center, value, _color in slices],
        style,
    )


def create_figure(
    palette: Palette = PALETTES[0],
    data: Pie3DData = DEFAULT_DATA,
    style: ChartStyle = DEFAULT_STYLE,
) -> Figure:
    """Create the 3D pie chart without writing it to disk."""

    palette.validate()
    data.validate()
    style.validate()

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

        ax.text(
            0.22,
            style.title_y,
            data.title,
            ha="center",
            va="center",
            fontsize=style.title_font_size,
            fontweight="bold",
            color="black",
            zorder=6,
        )
        _draw_legend(ax, palette, style)
        _draw_pie(ax, data, palette, style)

    return figure


def palette_from_selector(selector: str | int) -> tuple[int, Palette]:
    """Resolve a one-based palette number or a palette name."""

    if isinstance(selector, int) or str(selector).isdigit():
        index = int(selector)
        if 1 <= index <= len(PALETTES):
            return index, PALETTES[index - 1]
        raise ValueError(f"palette number must be between 1 and {len(PALETTES)}")

    normalized = str(selector).strip().lower().replace("_", "-")
    for index, palette in enumerate(PALETTES, start=1):
        if palette.name == normalized:
            return index, palette
    choices = ", ".join(palette.name for palette in PALETTES)
    raise ValueError(f"unknown palette {selector!r}; choose from: {choices}")


def render_palette(
    index: int,
    palette: Palette,
    output_dir: Path,
    *,
    formats: Sequence[str] = ("png",),
    dpi: int = 200,
    data: Pie3DData = DEFAULT_DATA,
    style: ChartStyle = DEFAULT_STYLE,
) -> list[Path]:
    """Render one palette and return the written paths."""

    normalized_formats = tuple(format_name.lower().lstrip(".") for format_name in formats)
    unsupported = set(normalized_formats) - {"png", "svg", "pdf"}
    if unsupported:
        raise ValueError(f"unsupported output format(s): {', '.join(sorted(unsupported))}")
    if dpi <= 0:
        raise ValueError("dpi must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    figure = create_figure(palette=palette, data=data, style=style)
    paths: list[Path] = []
    stem = f"pie_3d_palette_{index:02d}_{palette.name.replace('-', '_')}"
    try:
        for format_name in normalized_formats:
            path = output_dir / f"{stem}.{format_name}"
            figure.savefig(
                path,
                format=format_name,
                dpi=dpi,
                facecolor="white",
                edgecolor="none",
            )
            paths.append(path)
    finally:
        plt.close(figure)
    return paths


def _selected_palettes(selector: str) -> Iterable[tuple[int, Palette]]:
    if selector.strip().lower() == "all":
        return enumerate(PALETTES, start=1)
    return (palette_from_selector(selector),)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render the 15-factor 3D contribution pie chart."
    )
    parser.add_argument(
        "--palette",
        default="1",
        help="palette number/name, or 'all' (default: 1)",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png"],
        metavar="FORMAT",
        help="one or more of: png svg pdf (default: png)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="raster DPI; 200 reproduces the 2019×1350 reference (default: 200)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/pie_3d"),
        help="destination directory (default: output/pie_3d)",
    )
    parser.add_argument(
        "--list-palettes",
        action="store_true",
        help="list palette names and colours, then exit",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if args.list_palettes:
        for index, palette in enumerate(PALETTES, start=1):
            print(f"{index:2d}  {palette.name:24s}  {' '.join(palette.colors)}")
        return 0

    try:
        selections = tuple(_selected_palettes(args.palette))
        written = [
            path
            for index, palette in selections
            for path in render_palette(
                index,
                palette,
                args.output_dir,
                formats=args.formats,
                dpi=args.dpi,
            )
        ]
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
