"""Reproduce the annular radar-bubble chart from the Xiaohongshu reference.

The chart combines two encodings for the same 31 regions:

* smoothed filled profiles encode grain yield for 2020, 2010, and 2000;
* bubble area on three fixed-radius rings encodes planting area.

All geometry and the 18 palettes were reconstructed from the reference images.
The renderer deliberately uses Cartesian coordinates rather than a polar axis so
the annular mask, labels, ticks, and two legends can be positioned precisely.
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
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle
from numpy.typing import NDArray


YEARS: Final[tuple[int, int, int]] = (2020, 2010, 2000)

REGIONS: Final[tuple[str, ...]] = (
    "NX",
    "QH",
    "GS",
    "SN",
    "XZ",
    "YN",
    "GZ",
    "SC",
    "CQ",
    "HI",
    "GX",
    "GD",
    "HN",
    "HB",
    "HA",
    "SD",
    "JX",
    "FJ",
    "AH",
    "ZJ",
    "JS",
    "SH",
    "HL",
    "JL",
    "LN",
    "IM",
    "SX",
    "HE",
    "TJ",
    "BJ",
    "XJ",
)


@dataclass(frozen=True, slots=True)
class Palette:
    """Three year colors in 2020, 2010, 2000 order."""

    name: str
    colors: tuple[str, str, str]

    def for_year(self, year: int) -> str:
        try:
            return self.colors[YEARS.index(year)]
        except ValueError as exc:
            raise KeyError(f"unsupported year: {year}") from exc


# Colors are sampled from the solid legend swatches in images 1-18.
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
    """Values aligned with ``labels`` for every year."""

    labels: tuple[str, ...]
    grain_yield: Mapping[int, NDArray[np.float64]]
    planting_area: Mapping[int, NDArray[np.float64]]

    def validate(self) -> None:
        count = len(self.labels)
        if count < 3:
            raise ValueError("at least three labels are required")
        for field_name, values_by_year in (
            ("grain_yield", self.grain_yield),
            ("planting_area", self.planting_area),
        ):
            if set(values_by_year) != set(YEARS):
                raise ValueError(f"{field_name} must contain exactly {YEARS}")
            for year, values in values_by_year.items():
                values = np.asarray(values, dtype=float)
                if values.shape != (count,):
                    raise ValueError(
                        f"{field_name}[{year}] has shape {values.shape}; "
                        f"expected ({count},)"
                    )
                if not np.all(np.isfinite(values)) or np.any(values < 0):
                    raise ValueError(f"{field_name}[{year}] must be finite and nonnegative")


# The source post publishes the chart but not the underlying table. These values
# are digitized from the visible curve radii and bubble sizes so that the example
# reproduces the reference geometry; replace them with real data when available.
DEFAULT_DATA: Final[RadarBubbleData] = RadarBubbleData(
    labels=REGIONS,
    grain_yield={
        2020: np.array(
            [
                3740, 3580, 3460, 3350, 3270, 3150, 3050, 2950,
                2830, 2750, 2590, 4960, 5120, 4690, 4630, 4490,
                4370, 4270, 4170, 4070, 3960, 3880, 2560, 2400,
                2320, 2200, 2100, 1980, 1820, 1940, 3760,
            ],
            dtype=float,
        ),
        2010: np.array(
            [
                3150, 2200, 2160, 2060, 4190, 4010, 4010, 3860,
                3800, 3660, 3600, 3420, 3360, 3210, 3210, 3050,
                2970, 4880, 4610, 4650, 4450, 4390, 4370, 2690,
                2750, 2590, 2570, 2440, 1900, 1820, 4210,
            ],
            dtype=float,
        ),
        2000: np.array(
            [
                4930, 5030, 2380, 2260, 2120, 2000, 1900, 1770,
                3600, 3480, 3400, 3270, 3190, 3070, 2970, 2850,
                2750, 2630, 2560, 2380, 4030, 4330, 4450, 4390,
                4290, 4190, 4070, 3970, 3860, 3760, 5010,
            ],
            dtype=float,
        ),
    },
    planting_area={
        2020: np.array(
            [
                3900, 3500, 4000, 4200, 4100, 3300, 3000, 2900,
                2600, 3000, 2800, 6000, 5600, 4600, 4800, 5200,
                5400, 4700, 5600, 4500, 4300, 4400, 2700, 2400,
                2500, 2300, 2700, 2200, 2100, 2000, 4100,
            ],
            dtype=float,
        ),
        2010: np.array(
            [
                3000, 2400, 2500, 2200, 3700, 3400, 3200, 3000,
                2900, 2800, 2600, 4400, 4200, 3500, 3700, 4100,
                3900, 5200, 5000, 5600, 5300, 4800, 4300, 3100,
                3000, 2800, 2700, 2500, 2200, 2100, 3600,
            ],
            dtype=float,
        ),
        2000: np.array(
            [
                2900, 2700, 2600, 2300, 2200, 2100, 2000, 1900,
                2700, 2600, 2500, 3500, 3300, 3000, 3100, 3400,
                3200, 2800, 3000, 2700, 3900, 4200, 4500, 4100,
                3800, 3600, 3400, 3300, 3100, 3000, 3700,
            ],
            dtype=float,
        ),
    },
)


@dataclass(frozen=True, slots=True)
class ChartStyle:
    """Geometry and typography tuned to the 1196 x 1080 reference."""

    figure_size: tuple[float, float] = (11.96, 10.8)
    x_limits: tuple[float, float] = (-1.333, 1.620)
    y_limits: tuple[float, float] = (-1.333, 1.333)
    inner_radius: float = 0.30
    outer_radius: float = 1.00
    yield_offset: float = 0.125
    yield_span: float = 8000.0
    bubble_rings: tuple[float, float, float] = (0.90, 0.70, 0.50)
    fill_alpha: float = 0.46
    curve_width: float = 2.2
    grid_color: str = "#B9B9B9"
    grid_width: float = 1.0
    outer_width: float = 2.4
    label_radius: float = 1.105
    tick_outer_radius: float = 1.052
    samples_per_segment: int = 18


DEFAULT_STYLE: Final[ChartStyle] = ChartStyle()


def _angles(count: int) -> NDArray[np.float64]:
    """Return category angles starting at 12 o'clock and moving clockwise."""

    return np.pi / 2.0 - np.arange(count, dtype=float) * 2.0 * np.pi / count


def _yield_to_radius(
    values: NDArray[np.float64], style: ChartStyle
) -> NDArray[np.float64]:
    return style.yield_offset + np.asarray(values, dtype=float) / style.yield_span


def _bubble_size(values: float | NDArray[np.float64]) -> float | NDArray[np.float64]:
    """Map planting area to Matplotlib marker area in points squared."""

    return 80.0 + 0.065 * np.asarray(values, dtype=float)


def _smooth_periodic_radii(
    values: NDArray[np.float64], samples_per_segment: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Interpolate cyclic samples with a closed Catmull-Rom spline."""

    values = np.asarray(values, dtype=float)
    count = len(values)
    u = np.linspace(0.0, 1.0, samples_per_segment, endpoint=False)
    pieces: list[NDArray[np.float64]] = []
    parameters: list[NDArray[np.float64]] = []

    for index in range(count):
        p0 = values[(index - 1) % count]
        p1 = values[index]
        p2 = values[(index + 1) % count]
        p3 = values[(index + 2) % count]
        curve = 0.5 * (
            2.0 * p1
            + (-p0 + p2) * u
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * u**2
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * u**3
        )
        pieces.append(curve)
        parameters.append(index + u)

    radii = np.concatenate(pieces)
    parameter = np.concatenate(parameters)
    theta = np.pi / 2.0 - 2.0 * np.pi * parameter / count
    return (
        np.append(theta, theta[0] - 2.0 * np.pi),
        np.append(radii, radii[0]),
    )


def _draw_grid(ax: Axes, count: int, style: ChartStyle) -> None:
    theta = _angles(count)
    for angle in theta:
        cosine, sine = np.cos(angle), np.sin(angle)
        ax.plot(
            [style.inner_radius * cosine, style.outer_radius * cosine],
            [style.inner_radius * sine, style.outer_radius * sine],
            color=style.grid_color,
            linewidth=style.grid_width,
            zorder=1,
        )

    for tick in (2000, 4000, 6000):
        radius = style.yield_offset + tick / style.yield_span
        ax.add_patch(
            Circle(
                (0.0, 0.0),
                radius,
                facecolor="none",
                edgecolor=style.grid_color,
                linewidth=style.grid_width,
                zorder=1,
            )
        )


def _draw_profiles(
    ax: Axes,
    data: RadarBubbleData,
    palette: Palette,
    style: ChartStyle,
) -> None:
    # Oldest first gives overlap colors and edge visibility closest to the source.
    for year in reversed(YEARS):
        radii = _yield_to_radius(data.grain_yield[year], style)
        theta, smooth_radii = _smooth_periodic_radii(
            radii, style.samples_per_segment
        )
        x = smooth_radii * np.cos(theta)
        y = smooth_radii * np.sin(theta)
        ax.fill(
            x,
            y,
            facecolor=palette.for_year(year),
            edgecolor="none",
            alpha=style.fill_alpha,
            zorder=3,
        )

    for year in reversed(YEARS):
        radii = _yield_to_radius(data.grain_yield[year], style)
        theta, smooth_radii = _smooth_periodic_radii(
            radii, style.samples_per_segment
        )
        ax.plot(
            smooth_radii * np.cos(theta),
            smooth_radii * np.sin(theta),
            color=palette.for_year(year),
            linewidth=style.curve_width,
            solid_joinstyle="round",
            solid_capstyle="round",
            zorder=5,
        )


def _draw_bubbles(
    ax: Axes,
    data: RadarBubbleData,
    palette: Palette,
    style: ChartStyle,
) -> None:
    theta = _angles(len(data.labels))
    for year, radius in zip(YEARS, style.bubble_rings, strict=True):
        color = palette.for_year(year)
        ax.scatter(
            radius * np.cos(theta),
            radius * np.sin(theta),
            s=_bubble_size(data.planting_area[year]),
            facecolors=[to_rgba(color, 0.56)],
            edgecolors=[to_rgba(color, 0.92)],
            linewidths=1.0,
            zorder=7,
        )


def _draw_outer_labels(
    ax: Axes, labels: Sequence[str], style: ChartStyle
) -> None:
    theta = _angles(len(labels))
    for label, angle in zip(labels, theta, strict=True):
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
            fontsize=12.5,
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


def _draw_radial_axis(ax: Axes, style: ChartStyle) -> None:
    ax.plot(
        [0.0, 0.0],
        [style.inner_radius, style.tick_outer_radius],
        color="black",
        linewidth=1.8,
        zorder=16,
    )
    for tick in (2000, 4000, 6000):
        radius = style.yield_offset + tick / style.yield_span
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
            str(tick),
            ha="left",
            va="center",
            fontsize=10.5,
            zorder=18,
        )


def _draw_center(ax: Axes, style: ChartStyle) -> None:
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
        "Grain Yield\n(10k tons)",
        ha="center",
        va="center",
        fontsize=17,
        linespacing=1.05,
        zorder=25,
    )


def _draw_legends(ax: Axes, palette: Palette) -> None:
    ax.text(1.21, 0.46, "Planting Area (Kha)", fontsize=11.5, ha="left")
    legend_values = np.array([2000.0, 4000.0, 6000.0])
    legend_y = np.array([0.35, 0.245, 0.135])
    ax.scatter(
        np.full_like(legend_y, 1.33),
        legend_y,
        s=_bubble_size(legend_values),
        facecolors="white",
        edgecolors="black",
        linewidths=1.2,
        zorder=30,
    )
    for value, y in zip(legend_values.astype(int), legend_y, strict=True):
        ax.text(1.42, y, str(value), fontsize=11, ha="left", va="center")

    ax.text(1.24, -0.065, "Year", fontsize=11.5, ha="left")
    year_y = (-0.145, -0.23, -0.315)
    for year, y in zip(YEARS, year_y, strict=True):
        ax.add_patch(
            Rectangle(
                (1.21, y - 0.018),
                0.09,
                0.036,
                facecolor=palette.for_year(year),
                edgecolor="none",
                zorder=30,
            )
        )
        ax.text(1.40, y, str(year), fontsize=11, ha="left", va="center")


def create_figure(
    palette: Palette = PALETTES[0],
    data: RadarBubbleData = DEFAULT_DATA,
    style: ChartStyle = DEFAULT_STYLE,
) -> Figure:
    """Build one chart and return the Matplotlib figure."""

    data.validate()
    plt.rcParams.update(
        {
            "font.family": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
            "font.size": 11,
            "axes.linewidth": 1.5,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )

    figure = plt.figure(figsize=style.figure_size, facecolor="white")
    ax = figure.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_xlim(*style.x_limits)
    ax.set_ylim(*style.y_limits)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()

    _draw_grid(ax, len(data.labels), style)
    _draw_profiles(ax, data, palette, style)
    _draw_bubbles(ax, data, palette, style)
    _draw_outer_labels(ax, data.labels, style)
    _draw_radial_axis(ax, style)
    _draw_center(ax, style)
    _draw_legends(ax, palette)
    return figure


def palette_from_selector(selector: str | int) -> tuple[int, Palette]:
    """Resolve a one-based number or a palette name."""

    if isinstance(selector, int) or str(selector).isdigit():
        index = int(selector) - 1
        if not 0 <= index < len(PALETTES):
            raise ValueError(f"palette number must be between 1 and {len(PALETTES)}")
        return index + 1, PALETTES[index]

    normalized = str(selector).strip().lower().replace("_", "-")
    for index, palette in enumerate(PALETTES, start=1):
        if palette.name == normalized:
            return index, palette
    raise ValueError(f"unknown palette: {selector!r}")


def render_palette(
    palette_number: int,
    palette: Palette,
    output_dir: Path,
    formats: Iterable[str] = ("png",),
    dpi: int = 100,
    data: RadarBubbleData = DEFAULT_DATA,
    style: ChartStyle = DEFAULT_STYLE,
) -> list[Path]:
    """Render one palette to one or more file formats."""

    output_dir.mkdir(parents=True, exist_ok=True)
    figure = create_figure(palette=palette, data=data, style=style)
    stem = f"radar_bubble_{palette_number:02d}_{palette.name}"
    paths: list[Path] = []
    try:
        for output_format in formats:
            output_format = output_format.lower().lstrip(".")
            if output_format not in {"png", "svg", "pdf"}:
                raise ValueError(f"unsupported output format: {output_format}")
            path = output_dir / f"{stem}.{output_format}"
            figure.savefig(
                path,
                dpi=dpi,
                facecolor="white",
                edgecolor="none",
                bbox_inches=None,
                pad_inches=0,
            )
            paths.append(path)
    finally:
        plt.close(figure)
    return paths


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render the 31-region annular radar-bubble chart."
    )
    parser.add_argument(
        "--palette",
        default="1",
        help="palette number 1-18, palette name, or 'all' (default: 1)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="directory for generated figures (default: output)",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=("png",),
        choices=("png", "svg", "pdf"),
        help="one or more export formats (default: png)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=100,
        help="raster DPI; 100 reproduces the 1196x1080 reference (default: 100)",
    )
    parser.add_argument(
        "--list-palettes",
        action="store_true",
        help="print the available palette numbers and names, then exit",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.dpi <= 0:
        raise SystemExit("--dpi must be positive")

    if args.list_palettes:
        for index, palette in enumerate(PALETTES, start=1):
            print(f"{index:02d}  {palette.name:26s}  {' '.join(palette.colors)}")
        return 0

    if args.palette.lower() == "all":
        selections = list(enumerate(PALETTES, start=1))
    else:
        try:
            selections = [palette_from_selector(args.palette)]
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    generated: list[Path] = []
    for number, palette in selections:
        generated.extend(
            render_palette(
                palette_number=number,
                palette=palette,
                output_dir=args.output_dir,
                formats=args.formats,
                dpi=args.dpi,
            )
        )

    for path in generated:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
