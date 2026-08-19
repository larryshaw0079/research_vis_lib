"""Reproduce the BTH 2030 pie-plus-annular stacked energy chart.

The Xiaohongshu carousel repeats the same 2030 Beijing–Tianjin–Hebei terminal
energy figure 18 times, changing only the seven-colour palette.  A labelled
pie in the centre shows the overall fuel mix.  Six concentric 270° tracks
around it give sector totals in EJ, stacked by fuel from the 12 o'clock
origin clockwise.

Ring values were digitised from the visible arc lengths on a 0–8 EJ scale
(270° clockwise from 12 o'clock).  Pie labels follow the printed 52 / 26 /
13 percent wedges; the four remaining fuels share the leftover 9 percent.
The source post does not publish the underlying table; replace
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
from matplotlib.figure import Figure
from matplotlib.patches import Arc, Rectangle, Wedge


FUELS: Final[tuple[str, ...]] = (
    "Oil",
    "Gas",
    "Coal",
    "Biomass",
    "Wind",
    "Solar",
    "Electricity",
)

# Innermost track to outermost track.
SECTORS: Final[tuple[str, ...]] = (
    "Non-road",
    "Residential fuel",
    "Buildings",
    "Road",
    "Industrial",
    "Electricity",
)

# Clockwise pie order from 12 o'clock, matching the labelled wedges.
PIE_ORDER: Final[tuple[str, ...]] = (
    "Coal",
    "Oil",
    "Gas",
    "Biomass",
    "Wind",
    "Solar",
    "Electricity",
)


@dataclass(frozen=True, slots=True)
class Palette:
    """Seven fuel colours in :data:`FUELS` order."""

    name: str
    colors: tuple[str, str, str, str, str, str, str]

    def for_fuel(self, fuel: str) -> str:
        try:
            return self.colors[FUELS.index(fuel)]
        except ValueError as exc:
            raise KeyError(f"unsupported fuel: {fuel}") from exc

    def validate(self) -> None:
        if len(self.colors) != len(FUELS):
            raise ValueError(f"palette {self.name!r} must have {len(FUELS)} colours")


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
    """Sector-by-fuel energy in EJ plus the labelled centre-pie mix."""

    sectors: tuple[str, ...]
    fuels: tuple[str, ...]
    energy: Mapping[str, Mapping[str, float]]
    pie_percentages: Mapping[str, float]
    title: str = "2030-BTH"
    energy_unit: str = "Energy: EJ"

    def sector_total(self, sector: str) -> float:
        return float(sum(self.energy[sector].get(fuel, 0.0) for fuel in self.fuels))

    def stacked(self, sector: str) -> tuple[tuple[str, float], ...]:
        return tuple(
            (fuel, float(self.energy[sector][fuel]))
            for fuel in self.fuels
            if self.energy[sector].get(fuel, 0.0) > 0.0
        )

    def pie_values(self) -> tuple[float, ...]:
        return tuple(self.pie_percentages[name] for name in PIE_ORDER)

    def validate(self) -> None:
        if self.sectors != SECTORS:
            raise ValueError("sectors must match the canonical six-sector order")
        if self.fuels != FUELS:
            raise ValueError("fuels must match the canonical seven-fuel order")
        if set(self.energy) != set(self.sectors):
            raise ValueError("energy must contain exactly one mapping per sector")
        for sector, values in self.energy.items():
            extra = set(values) - set(self.fuels)
            if extra:
                raise ValueError(f"energy[{sector!r}] has unknown fuels: {sorted(extra)}")
            for fuel, value in values.items():
                if not np.isfinite(value) or value < 0:
                    raise ValueError(
                        f"energy[{sector!r}][{fuel!r}] must be finite and non-negative"
                    )
            if self.sector_total(sector) <= 0:
                raise ValueError(f"energy[{sector!r}] must sum to a positive total")
        if set(self.pie_percentages) != set(self.fuels):
            raise ValueError("pie_percentages must contain one value per fuel")
        if any(value < 0 for value in self.pie_percentages.values()):
            raise ValueError("pie_percentages must be non-negative")
        if sum(self.pie_percentages.values()) <= 0:
            raise ValueError("pie_percentages must sum to a positive total")


# Digitised from carousel frame 1; values fall on a 0.1 EJ grid.
DEFAULT_DATA: Final[PieRingData] = PieRingData(
    sectors=SECTORS,
    fuels=FUELS,
    energy={
        "Non-road": {"Electricity": 0.1},
        "Residential fuel": {"Oil": 0.2, "Coal": 0.5},
        "Buildings": {"Oil": 1.1, "Gas": 0.8, "Coal": 0.4, "Electricity": 0.4},
        "Road": {"Oil": 2.8, "Gas": 0.2, "Electricity": 1.2},
        "Industrial": {"Gas": 1.0, "Coal": 4.1, "Electricity": 0.9},
        "Electricity": {
            "Gas": 0.5,
            "Coal": 4.0,
            "Biomass": 0.3,
            "Wind": 0.8,
            "Solar": 0.5,
            "Electricity": 1.9,
        },
    },
    pie_percentages={
        "Coal": 52.0,
        "Oil": 26.0,
        "Gas": 13.0,
        "Biomass": 2.0,
        "Wind": 3.0,
        "Solar": 2.0,
        "Electricity": 2.0,
    },
)


@dataclass(frozen=True, slots=True)
class ChartStyle:
    """Layout tuned to the 2370 × 2490 reference canvas."""

    figure_size: tuple[float, float] = (11.85, 12.45)
    x_limits: tuple[float, float] = (-1185.0, 1185.0)
    y_limits: tuple[float, float] = (-1187.0, 1303.0)
    pie_radius: float = 331.0
    first_ring_inner: float = 475.0
    ring_width: float = 72.0
    ring_gap: float = 24.0
    scale_radius: float = 1052.0
    max_energy: float = 8.0
    span_degrees: float = 270.0
    pie_gap_degrees: float = 0.0
    separator_width: float = 2.2
    track_color: str = "#F0F0F0"
    axis_color: str = "#A0A0A0"
    scale_color: str = "#A0A0A0"
    edge_color: str = "#FFFFFF"
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
    sector_label_x: float = -48.0
    min_pie_label_percent: float = 10.0

    def ring_inner(self, index: int) -> float:
        return self.first_ring_inner + index * (self.ring_width + self.ring_gap)

    def ring_outer(self, index: int) -> float:
        return self.ring_inner(index) + self.ring_width

    def energy_to_theta(self, energy: float) -> float:
        """Matplotlib wedge angle (CCW from +x) for a clockwise EJ value."""

        return 90.0 - energy / self.max_energy * self.span_degrees

    def validate(self) -> None:
        if min(self.figure_size) <= 0:
            raise ValueError("figure_size must be positive")
        if self.pie_radius <= 0 or self.ring_width <= 0:
            raise ValueError("pie_radius and ring_width must be positive")
        if self.max_energy <= 0 or self.span_degrees <= 0:
            raise ValueError("max_energy and span_degrees must be positive")


DEFAULT_STYLE: Final[ChartStyle] = ChartStyle()


def _wedge_thetas(start: float, end: float, style: ChartStyle) -> tuple[float, float]:
    """Return (theta1, theta2) so matplotlib draws the clockwise EJ interval."""

    return style.energy_to_theta(end), style.energy_to_theta(start)


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


def _draw_axes(ax: Axes, style: ChartStyle) -> None:
    """Draw only the 12 o'clock (0 EJ) and 9 o'clock (8 EJ) reference rays."""

    outer = style.scale_radius + 18.0
    inner = style.pie_radius + 3.0
    _radial_line(
        ax,
        inner,
        outer,
        90.0,
        color=style.axis_color,
        linewidth=1.2,
        zorder=4,
    )
    ax.plot(
        [-outer, -inner],
        [0.0, 0.0],
        color=style.axis_color,
        linewidth=1.2,
        solid_capstyle="butt",
        zorder=4,
    )


def _draw_scale(ax: Axes, data: PieRingData, style: ChartStyle) -> None:
    radius = style.scale_radius
    ax.add_patch(
        Arc(
            (0.0, 0.0),
            2.0 * radius,
            2.0 * radius,
            theta1=style.energy_to_theta(style.max_energy),
            theta2=style.energy_to_theta(0.0),
            color=style.scale_color,
            linewidth=1.35,
            zorder=5,
        )
    )
    ticks = (0.0, 2.0, 4.0, 6.0, 8.0)
    for energy in ticks:
        theta = np.deg2rad(style.energy_to_theta(energy))
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
        if energy == 0.0:
            ha, va = "center", "bottom"
            y += 4.0
        elif energy == style.max_energy:
            ha, va = "right", "center"
        else:
            ha, va = "center", "center"
        ax.text(
            x,
            y,
            f"{energy:.0f}",
            ha=ha,
            va=va,
            fontsize=style.tick_font_size,
            fontweight="bold",
            color="black",
            zorder=8,
        )

    ax.text(
        -718.0,
        0.0,
        data.energy_unit,
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
    for index, sector in enumerate(data.sectors):
        inner = style.ring_inner(index)
        outer = style.ring_outer(index)
        theta1, theta2 = _wedge_thetas(0.0, style.max_energy, style)
        ax.add_patch(
            Wedge(
                (0.0, 0.0),
                outer,
                theta1,
                theta2,
                width=style.ring_width,
                facecolor=style.track_color,
                edgecolor="none",
                zorder=2,
            )
        )
        cursor = 0.0
        joints: list[float] = []
        for fuel, value in data.stacked(sector):
            start, end = cursor, cursor + value
            t1, t2 = _wedge_thetas(start, end, style)
            ax.add_patch(
                Wedge(
                    (0.0, 0.0),
                    outer,
                    t1,
                    t2,
                    width=style.ring_width,
                    facecolor=palette.for_fuel(fuel),
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
                style.energy_to_theta(joint),
                color=style.edge_color,
                linewidth=style.separator_width,
                zorder=4,
            )
        mid_r = 0.5 * (inner + outer)
        ax.text(
            style.sector_label_x,
            mid_r,
            sector,
            ha="right",
            va="center",
            fontsize=style.label_font_size,
            fontweight="bold",
            color="black",
            zorder=8,
        )


def _draw_pie(
    ax: Axes,
    data: PieRingData,
    palette: Palette,
    style: ChartStyle,
) -> None:
    values = np.asarray(data.pie_values(), dtype=float)
    total = float(values.sum())
    widths = 360.0 * values / total
    edges = np.concatenate(([90.0], 90.0 - np.cumsum(widths)))
    gap = 0.5 * style.pie_gap_degrees
    for fuel, theta_start, theta_end, value in zip(
        PIE_ORDER, edges[:-1], edges[1:], values, strict=True
    ):
        t1 = float(theta_end) + gap
        t2 = float(theta_start) - gap
        if t2 <= t1:
            t1, t2 = float(theta_end), float(theta_start)
        ax.add_patch(
            Wedge(
                (0.0, 0.0),
                style.pie_radius,
                t1,
                t2,
                facecolor=palette.for_fuel(fuel),
                edgecolor="none",
                zorder=5,
            )
        )
        if value < style.min_pie_label_percent:
            continue
        mid = 0.5 * (t1 + t2)
        theta = np.deg2rad(mid)
        radius = (0.62 if value >= 40.0 else 0.76) * style.pie_radius
        ax.text(
            radius * np.cos(theta),
            radius * np.sin(theta),
            f"{value:.0f}%",
            ha="center",
            va="center",
            fontsize=style.pie_font_size,
            fontweight="bold",
            color="black",
            zorder=7,
        )
    for theta_deg in edges[:-1]:
        _radial_line(
            ax,
            0.0,
            style.pie_radius,
            float(theta_deg),
            color=style.edge_color,
            linewidth=style.separator_width,
            zorder=6,
        )


def _draw_legend(ax: Axes, palette: Palette, style: ChartStyle) -> None:
    swatch_w, swatch_h = style.legend_swatch
    for index, fuel in enumerate(FUELS):
        y = style.legend_top - index * style.legend_step
        ax.add_patch(
            Rectangle(
                (style.legend_x, y - 0.5 * swatch_h),
                swatch_w,
                swatch_h,
                facecolor=palette.for_fuel(fuel),
                edgecolor="none",
                zorder=8,
            )
        )
        ax.text(
            style.legend_x + swatch_w + style.legend_text_gap,
            y,
            fuel,
            ha="left",
            va="center",
            fontsize=style.legend_font_size,
            fontweight="bold",
            color="black",
            zorder=8,
        )


def create_figure(
    palette: Palette = PALETTES[0],
    data: PieRingData = DEFAULT_DATA,
    style: ChartStyle = DEFAULT_STYLE,
) -> Figure:
    """Create the pie-plus-annular stacked chart without writing it to disk."""

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

        _draw_rings(ax, data, palette, style)
        _draw_pie(ax, data, palette, style)
        _draw_axes(ax, style)
        _draw_scale(ax, data, style)
        _draw_legend(ax, palette, style)
        ax.text(
            0.0,
            1191.0,
            data.title,
            ha="center",
            va="bottom",
            fontsize=style.title_font_size,
            fontweight="bold",
            color="black",
            zorder=9,
        )

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
    data: PieRingData = DEFAULT_DATA,
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
    stem = f"pie_ring_palette_{index:02d}_{palette.name.replace('-', '_')}"
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
        description="Render the BTH 2030 pie-plus-annular stacked energy chart."
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
        help="raster DPI; 200 reproduces the 2370×2490 reference (default: 200)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/pie_ring"),
        help="destination directory (default: output/pie_ring)",
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
