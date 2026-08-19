"""Reproduce the OTU flower / petal chart from the Xiaohongshu reference.

The carousel repeats the same 10-group flower plot 18 times, changing only the
colour palette.  Each petal is a rounded annular sector: radial sides, a circular inner
edge, and filleted outer tips.  The large number is the group's total OTU
count and the parenthetical number is the unique count.  The centre reports
OTUs shared by every group.

Counts were digitised from the labelled petals.  The source post does not name
the underlying table; replace :data:`DEFAULT_DATA` when using another dataset.
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
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MatplotlibPath
from numpy.typing import NDArray


GROUPS: Final[tuple[str, ...]] = (
    "CK",
    "Cu-L",
    "Cu-M",
    "Cu-H",
    "Cr-L",
    "Cr-M",
    "Cr-H",
    "Mix-L",
    "Mix-M",
    "Mix-H",
)


@dataclass(frozen=True, slots=True)
class Palette:
    """Ten petal colours in :data:`GROUPS` order."""

    name: str
    colors: tuple[str, str, str, str, str, str, str, str, str, str]

    def for_group(self, group: str) -> str:
        try:
            return self.colors[GROUPS.index(group)]
        except ValueError as exc:
            raise KeyError(f"unsupported group: {group}") from exc


# Palettes reconstructed from carousel images 1-18.  Named qualitative sets
# use canonical ColorBrewer / Tableau / matplotlib / Paul Tol / Okabe–Ito
# hex values; the rest are median colours sampled from the petal interiors.
PALETTES: Final[tuple[Palette, ...]] = (
    Palette(
        "gray-peach-orchid",
        (
            "#BDBDBD",
            "#F8CDAB",
            "#F09B88",
            "#EC655D",
            "#A1DFC8",
            "#6FCDD6",
            "#75A7EE",
            "#ADB8F5",
            "#C6A5E0",
            "#F6C3DF",
        ),
    ),
    Palette(
        "tab10",
        (
            "#1F77B4",
            "#FF7F0E",
            "#2CA02C",
            "#D62728",
            "#9467BD",
            "#8C564B",
            "#E377C2",
            "#7F7F7F",
            "#BCBD22",
            "#17BECF",
        ),
    ),
    Palette(
        "set3",
        (
            "#8DD3C7",
            "#FFFFB3",
            "#BEBADA",
            "#FB8072",
            "#80B1D3",
            "#FDB462",
            "#B3DE69",
            "#FCCDE5",
            "#D9D9D9",
            "#BC80BD",
        ),
    ),
    Palette(
        "paired",
        (
            "#A6CEE3",
            "#1F78B4",
            "#B2DF8A",
            "#33A02C",
            "#FB9A99",
            "#E31A1C",
            "#FDBF6F",
            "#FF7F00",
            "#CAB2D6",
            "#6A3D9A",
        ),
    ),
    Palette(
        "pastel1",
        (
            "#FBB4AE",
            "#B3CDE3",
            "#CCEBC5",
            "#DECBE4",
            "#FED9A6",
            "#FFFFCC",
            "#E5D8BD",
            "#FDDAEC",
            "#F2F2F2",
            "#B3B3B3",
        ),
    ),
    Palette(
        "set1",
        (
            "#E41A1C",
            "#377EB8",
            "#4DAF4A",
            "#984EA3",
            "#FF7F00",
            "#FFFF33",
            "#A65628",
            "#F781BF",
            "#999999",
            "#B15928",
        ),
    ),
    Palette(
        "dark2",
        (
            "#1B9E77",
            "#D95F02",
            "#7570B3",
            "#E7298A",
            "#66A61E",
            "#E6AB02",
            "#A6761D",
            "#666666",
            "#B3E2CD",
            "#999999",
        ),
    ),
    Palette(
        "accent",
        (
            "#7FC97F",
            "#BEAED4",
            "#FDC086",
            "#FFFF99",
            "#386CB0",
            "#F0027F",
            "#BF5B17",
            "#666666",
            "#FDCDAC",
            "#444444",
        ),
    ),
    Palette(
        "tableau10",
        (
            "#4E79A7",
            "#F28E2B",
            "#E15759",
            "#76B7B2",
            "#59A14F",
            "#EDC948",
            "#B07AA1",
            "#FF9D9A",
            "#9C755F",
            "#BAB0AC",
        ),
    ),
    Palette(
        "okabe-ito",
        (
            "#E69F00",
            "#56B4E9",
            "#009E73",
            "#F0E442",
            "#0072B2",
            "#D55E00",
            "#CC79A7",
            "#8DD3C7",
            "#737373",
            "#CCCCCC",
        ),
    ),
    Palette(
        "tol-muted",
        (
            "#332288",
            "#88CCEE",
            "#44AA99",
            "#117733",
            "#999933",
            "#DDCC77",
            "#CC6677",
            "#882255",
            "#AA4499",
            "#DDDDDD",
        ),
    ),
    Palette(
        "dust-rainbow",
        (
            "#88CDE5",
            "#B8D4A1",
            "#E7D893",
            "#EFA88E",
            "#D78994",
            "#B5759E",
            "#8D77A5",
            "#7A93B1",
            "#999999",
            "#555555",
        ),
    ),
    Palette(
        "coral-slate-gold",
        (
            "#FCA07C",
            "#20B1AA",
            "#778898",
            "#9470D8",
            "#3DB271",
            "#FB6349",
            "#4882B4",
            "#D8A523",
            "#FD69B2",
            "#CD5E5E",
        ),
    ),
    Palette(
        "pastel-candy",
        (
            "#FCB4BB",
            "#FEDFB9",
            "#FFFEB9",
            "#BAFDC9",
            "#BAE1FE",
            "#D7BFD7",
            "#FEBFCD",
            "#E5E6F7",
            "#EE807F",
            "#21B0A8",
        ),
    ),
    Palette(
        "atlantic-sunset",
        (
            "#264653",
            "#2A9D8F",
            "#E9C46A",
            "#F4A261",
            "#E76F51",
            "#8AB07D",
            "#BABB75",
            "#DDA06E",
            "#C16C5D",
            "#5B7470",
        ),
    ),
    Palette(
        "gold-teal-rust",
        (
            "#FCB706",
            "#045F73",
            "#0E9396",
            "#94D1BD",
            "#E9D7A5",
            "#EC9B01",
            "#CC6708",
            "#BB400A",
            "#AE2114",
            "#9B2329",
        ),
    ),
    Palette(
        "lime-navy",
        (
            "#D9ED93",
            "#B6E48C",
            "#99D98B",
            "#77C793",
            "#52B69A",
            "#349FA2",
            "#198AAD",
            "#1E769F",
            "#206092",
            "#1A4E78",
        ),
    ),
    Palette(
        "neon-rainbow",
        (
            "#FB595E",
            "#FDCA39",
            "#8BC927",
            "#1D82C4",
            "#6A4C92",
            "#EF5BB4",
            "#01F3D3",
            "#FBE442",
            "#00BBF8",
            "#9B5EE6",
        ),
    ),
)


@dataclass(frozen=True, slots=True)
class FlowerPlotData:
    """Core / total / unique OTU counts aligned with :data:`GROUPS`."""

    groups: tuple[str, ...]
    core: int
    totals: Mapping[str, int]
    uniques: Mapping[str, int]

    def validate(self) -> None:
        if self.groups != GROUPS:
            raise ValueError("groups must match the canonical ten-group order")
        if self.core < 0:
            raise ValueError("core count must be non-negative")
        if set(self.totals) != set(self.groups) or set(self.uniques) != set(self.groups):
            raise ValueError("totals and uniques must contain one value per group")
        for group in self.groups:
            total = self.totals[group]
            unique = self.uniques[group]
            if total < 0 or unique < 0:
                raise ValueError(f"{group} counts must be non-negative")
            if unique > total:
                raise ValueError(f"{group} unique count cannot exceed the total")


@dataclass(frozen=True, slots=True)
class ChartStyle:
    """Layout tuned to the 2560 × 2560 reference canvas."""

    figure_size: tuple[float, float] = (10.24, 10.24)
    x_limits: tuple[float, float] = (-1.0, 1.0)
    y_limits: tuple[float, float] = (-1.0, 1.0)
    inner_radius: float = 0.248
    outer_radius: float = 0.648
    label_radius: float = 0.755
    petal_width_degrees: float = 31.2
    inner_corner_radius: float = 0.012
    outer_corner_radius: float = 0.055
    core_font_size: float = 20.0
    core_line_gap: float = 0.058
    value_font_size: float = 13.0
    value_line_gap: float = 0.040
    label_font_size: float = 14.0
    luminance_cutoff: float = 0.22

    def validate(self) -> None:
        if min(self.figure_size) <= 0:
            raise ValueError("figure_size must be positive")
        if not 0 < self.inner_radius < self.outer_radius:
            raise ValueError("radii must satisfy 0 < inner < outer")
        if self.label_radius <= self.outer_radius:
            raise ValueError("label_radius must sit outside the petals")
        if not 0 < self.petal_width_degrees < 360.0 / len(GROUPS):
            raise ValueError("petal_width_degrees must leave a gap between petals")
        thickness = self.outer_radius - self.inner_radius
        if not 0 <= self.inner_corner_radius < 0.5 * thickness:
            raise ValueError("inner_corner_radius must be smaller than half the petal thickness")
        if not 0 <= self.outer_corner_radius < 0.5 * thickness:
            raise ValueError("outer_corner_radius must be smaller than half the petal thickness")


DEFAULT_DATA: Final[FlowerPlotData] = FlowerPlotData(
    groups=GROUPS,
    core=936,
    totals={
        "CK": 1755,
        "Cu-L": 2167,
        "Cu-M": 2182,
        "Cu-H": 2080,
        "Cr-L": 2308,
        "Cr-M": 1906,
        "Cr-H": 2014,
        "Mix-L": 2190,
        "Mix-M": 1877,
        "Mix-H": 1641,
    },
    uniques={
        "CK": 171,
        "Cu-L": 167,
        "Cu-M": 158,
        "Cu-H": 181,
        "Cr-L": 223,
        "Cr-M": 170,
        "Cr-H": 157,
        "Mix-L": 168,
        "Mix-M": 127,
        "Mix-H": 105,
    },
)

DEFAULT_STYLE: Final[ChartStyle] = ChartStyle()


def _petal_center_degrees(index: int) -> float:
    """Math angle (CCW from +x) of petal ``index``, clockwise from north."""

    return (90.0 - index * (360.0 / len(GROUPS))) % 360.0


def _contrasting_text_color(color: str, cutoff: float) -> str:
    red, green, blue = to_rgb(color)

    def linear(channel: float) -> float:
        if channel <= 0.04045:
            return channel / 12.92
        return ((channel + 0.055) / 1.055) ** 2.4

    luminance = (
        0.2126 * linear(red) + 0.7152 * linear(green) + 0.0722 * linear(blue)
    )
    return "#FFFFFF" if luminance < cutoff else "#000000"


def _label_rotation(theta_degrees: float) -> float:
    """Keep outer labels tangential and right-side up."""

    rotation = (theta_degrees - 90.0) % 360.0
    if 90.0 < rotation < 270.0:
        rotation += 180.0
    return (rotation + 180.0) % 360.0 - 180.0


def _unit(angle: float) -> NDArray[np.float64]:
    return np.array([np.cos(angle), np.sin(angle)], dtype=float)


def _arc_points(
    radius: float, start: float, stop: float, count: int
) -> NDArray[np.float64]:
    angles = np.linspace(start, stop, count)
    return radius * np.column_stack((np.cos(angles), np.sin(angles)))


def _circle_arc(
    center: NDArray[np.float64],
    start: NDArray[np.float64],
    stop: NDArray[np.float64],
    count: int,
) -> NDArray[np.float64]:
    """Short arc from ``start`` to ``stop`` on the circle around ``center``."""

    radius = float(np.linalg.norm(start - center))
    a0 = np.arctan2(start[1] - center[1], start[0] - center[0])
    a1 = np.arctan2(stop[1] - center[1], stop[0] - center[0])
    ccw = (a1 - a0) % (2.0 * np.pi)
    cw = (a0 - a1) % (2.0 * np.pi)
    angles = np.linspace(a0, a0 + ccw, count) if ccw <= cw else np.linspace(a0, a0 - cw, count)
    return center + radius * np.column_stack((np.cos(angles), np.sin(angles)))


def _clamped_corner_radius(
    requested: float,
    *,
    radial_room: float,
    half_width: float,
) -> float:
    return float(max(0.0, min(requested, 0.48 * radial_room, 0.90 * half_width)))


def _polar_fillet(
    theta: float,
    radius: float,
    corner_radius: float,
    *,
    inner: bool,
    toward_increasing: bool,
    count: int,
) -> tuple[NDArray[np.float64], float]:
    """Return a fillet at a polar-sector corner and the adjacent arc angle."""

    radial = _unit(theta)
    tangential = np.array([-np.sin(theta), np.cos(theta)], dtype=float)
    side = 1.0 if toward_increasing else -1.0
    if inner:
        center_radius = radius + corner_radius
        alpha = np.arcsin(
            np.clip(corner_radius / max(center_radius, 1e-9), 0.0, 0.99)
        )
        center = center_radius * _unit(theta + side * alpha)
        on_side = (radius + corner_radius) * radial
        on_arc = radius * _unit(theta + side * alpha)
    else:
        center_radius = radius - corner_radius
        alpha = np.arcsin(
            np.clip(corner_radius / max(center_radius, 1e-9), 0.0, 0.99)
        )
        center = center_radius * _unit(theta + side * alpha)
        on_side = (radius - corner_radius) * radial
        on_arc = radius * _unit(theta + side * alpha)
    return _circle_arc(center, on_side, on_arc, count), float(theta + side * alpha)


def rounded_petal_vertices(
    center_degrees: float,
    width_degrees: float,
    inner_radius: float,
    outer_radius: float,
    inner_corner: float,
    outer_corner: float,
    *,
    arc_points: int = 40,
    fillet_points: int = 16,
) -> NDArray[np.float64]:
    """Vertices of a rounded annular sector, walking counter-clockwise."""

    theta1 = np.deg2rad(center_degrees - 0.5 * width_degrees)
    theta2 = np.deg2rad(center_degrees + 0.5 * width_degrees)
    half_angle = 0.5 * (theta2 - theta1)
    radial_room = outer_radius - inner_radius
    rho_out = _clamped_corner_radius(
        outer_corner,
        radial_room=radial_room,
        half_width=outer_radius * np.sin(half_angle),
    )
    rho_in = _clamped_corner_radius(
        inner_corner,
        radial_room=radial_room,
        half_width=inner_radius * np.sin(half_angle),
    )

    inner1, inner_a1 = _polar_fillet(
        theta1,
        inner_radius,
        rho_in,
        inner=True,
        toward_increasing=True,
        count=fillet_points,
    )
    inner2, inner_a2 = _polar_fillet(
        theta2,
        inner_radius,
        rho_in,
        inner=True,
        toward_increasing=False,
        count=fillet_points,
    )
    outer2, outer_a2 = _polar_fillet(
        theta2,
        outer_radius,
        rho_out,
        inner=False,
        toward_increasing=False,
        count=fillet_points,
    )
    outer1, outer_a1 = _polar_fillet(
        theta1,
        outer_radius,
        rho_out,
        inner=False,
        toward_increasing=True,
        count=fillet_points,
    )

    # CCW around the petal: inner θ1 → inner arc → inner θ2 → outer θ2 →
    # outer arc (decreasing θ) → outer θ1.  Reverse fillets that were built
    # from the radial side toward the arc when the walk hits the arc first.
    return np.vstack(
        (
            inner1,
            _arc_points(inner_radius, inner_a1, inner_a2, arc_points),
            inner2[::-1],
            outer2,
            _arc_points(outer_radius, outer_a2, outer_a1, arc_points),
            outer1[::-1],
        )
    )


def _draw_petals(
    ax: Axes,
    data: FlowerPlotData,
    palette: Palette,
    style: ChartStyle,
) -> None:
    value_radius = 0.5 * (style.inner_radius + style.outer_radius)
    for index, group in enumerate(data.groups):
        center = _petal_center_degrees(index)
        color = palette.for_group(group)
        vertices = rounded_petal_vertices(
            center,
            style.petal_width_degrees,
            style.inner_radius,
            style.outer_radius,
            style.inner_corner_radius,
            style.outer_corner_radius,
        )
        codes = np.full(len(vertices) + 1, MatplotlibPath.LINETO, dtype=np.uint8)
        codes[0] = MatplotlibPath.MOVETO
        codes[-1] = MatplotlibPath.CLOSEPOLY
        path = MatplotlibPath(np.vstack((vertices, vertices[0])), codes)
        ax.add_patch(
            PathPatch(
                path,
                facecolor=color,
                edgecolor=color,
                linewidth=0.35,
                joinstyle="round",
                zorder=2,
            )
        )
        theta = np.deg2rad(center)
        text_color = _contrasting_text_color(color, style.luminance_cutoff)
        x = value_radius * np.cos(theta)
        y = value_radius * np.sin(theta)
        ax.text(
            x,
            y + 0.5 * style.value_line_gap,
            str(data.totals[group]),
            ha="center",
            va="center",
            fontsize=style.value_font_size,
            color=text_color,
            zorder=4,
        )
        ax.text(
            x,
            y - 0.5 * style.value_line_gap,
            f"({data.uniques[group]})",
            ha="center",
            va="center",
            fontsize=style.value_font_size,
            color=text_color,
            zorder=4,
        )


def _draw_labels(ax: Axes, style: ChartStyle) -> None:
    for index, group in enumerate(GROUPS):
        center = _petal_center_degrees(index)
        theta = np.deg2rad(center)
        ax.text(
            style.label_radius * np.cos(theta),
            style.label_radius * np.sin(theta),
            group,
            ha="center",
            va="center",
            rotation=_label_rotation(center),
            rotation_mode="anchor",
            fontsize=style.label_font_size,
            color="black",
            clip_on=False,
            zorder=5,
        )


def _draw_core(ax: Axes, data: FlowerPlotData, style: ChartStyle) -> None:
    ax.text(
        0.0,
        0.5 * style.core_line_gap,
        "Core",
        ha="center",
        va="center",
        fontsize=style.core_font_size,
        fontweight="bold",
        color="black",
        zorder=6,
    )
    ax.text(
        0.0,
        -0.5 * style.core_line_gap,
        str(data.core),
        ha="center",
        va="center",
        fontsize=style.core_font_size,
        fontweight="bold",
        color="black",
        zorder=6,
    )


def create_figure(
    palette: Palette = PALETTES[0],
    data: FlowerPlotData = DEFAULT_DATA,
    style: ChartStyle = DEFAULT_STYLE,
) -> Figure:
    """Create the flower plot without writing it to disk."""

    data.validate()
    style.validate()

    with plt.rc_context(
        {
            "font.family": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
            "font.size": style.value_font_size,
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

        _draw_petals(ax, data, palette, style)
        _draw_labels(ax, style)
        _draw_core(ax, data, style)

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
    dpi: int = 250,
    data: FlowerPlotData = DEFAULT_DATA,
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
    stem = f"flower_plot_palette_{index:02d}_{palette.name.replace('-', '_')}"
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
        description="Render the 10-group OTU flower / petal chart."
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
        default=250,
        help="raster DPI; 250 reproduces the 2560×2560 reference (default: 250)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/flower_plot"),
        help="destination directory (default: output/flower_plot)",
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
            print(f"{index:2d}  {palette.name:22s}  {' '.join(palette.colors)}")
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
