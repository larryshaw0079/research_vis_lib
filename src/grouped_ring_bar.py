"""Reproduce the grouped annular bar chart from the Xiaohongshu reference.

The chart compares five long-term forecasting models on eight datasets. Bar
length is an inverted, within-dataset min–max encoding of MSE so that a lower
score yields a longer bar and the best/worst models in each dataset span a
shared radial range. The Xiaohongshu carousel repeats the same geometry with
18 five-colour palettes.

Values were digitised from the labelled bars. They follow Table 2 (horizon
L=720, MSE) of Ma et al., "MoFo: Empowering Long-term Time Series Forecasting
with Periodic Pattern Modeling", NeurIPS 2025
(https://openreview.net/forum?id=sbvLts2HqR), with a few entries that differ
from the camera-ready table and match the published figure instead.
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
from matplotlib.patches import Arc, FancyBboxPatch, Rectangle, Wedge
from numpy.typing import NDArray


MODELS: Final[tuple[str, ...]] = (
    "iTransformer",
    "Pathformer",
    "PatchTST",
    "PDF",
    "MoFo (Ours)",
)

DATASETS: Final[tuple[str, ...]] = (
    "Traffic",
    "ETTh1",
    "ETTh2",
    "ETTm1",
    "ETTm2",
    "Weather",
    "Electricity",
    "Solar",
)


@dataclass(frozen=True, slots=True)
class Palette:
    """Five model colours in :data:`MODELS` order."""

    name: str
    colors: tuple[str, str, str, str, str]

    def for_model(self, model: str) -> str:
        try:
            return self.colors[MODELS.index(model)]
        except ValueError as exc:
            raise KeyError(f"unsupported model: {model}") from exc


# Colours are sampled from the legend swatches in carousel images 1-18.
# Several match widely used Coolors / Tailwind five-colour palettes.
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
    """MSE values aligned with :data:`DATASETS` and :data:`MODELS`."""

    datasets: tuple[str, ...]
    models: tuple[str, ...]
    mse: Mapping[str, Mapping[str, float]]

    def validate(self) -> None:
        if self.datasets != DATASETS:
            raise ValueError("datasets must match the canonical eight-dataset order")
        if self.models != MODELS:
            raise ValueError("models must match the canonical five-model order")
        if set(self.mse) != set(self.datasets):
            raise ValueError("mse must contain exactly one mapping per dataset")
        for dataset, values in self.mse.items():
            if set(values) != set(self.models):
                raise ValueError(f"mse[{dataset!r}] must contain exactly {self.models}")
            for model, value in values.items():
                if not np.isfinite(value) or value <= 0:
                    raise ValueError(
                        f"mse[{dataset!r}][{model!r}] must be finite and positive"
                    )


# Horizon-720 MSE digitised from the labelled bars in the reference carousel.
DEFAULT_DATA: Final[GroupedRingBarData] = GroupedRingBarData(
    datasets=DATASETS,
    models=MODELS,
    mse={
        "Traffic": {
            "iTransformer": 0.445,
            "Pathformer": 0.452,
            "PatchTST": 0.435,
            "PDF": 0.438,
            "MoFo (Ours)": 0.424,
        },
        "ETTh1": {
            "iTransformer": 0.495,
            "Pathformer": 0.450,
            "PatchTST": 0.457,
            "PDF": 0.456,
            "MoFo (Ours)": 0.447,
        },
        "ETTh2": {
            "iTransformer": 0.424,
            "Pathformer": 0.413,
            "PatchTST": 0.406,
            "PDF": 0.398,
            "MoFo (Ours)": 0.379,
        },
        "ETTm1": {
            "iTransformer": 0.429,
            "Pathformer": 0.428,
            "PatchTST": 0.416,
            "PDF": 0.408,
            "MoFo (Ours)": 0.388,
        },
        "ETTm2": {
            "iTransformer": 0.375,
            "Pathformer": 0.361,
            "PatchTST": 0.362,
            "PDF": 0.349,
            "MoFo (Ours)": 0.342,
        },
        "Weather": {
            "iTransformer": 0.320,
            "Pathformer": 0.318,
            "PatchTST": 0.312,
            "PDF": 0.323,
            "MoFo (Ours)": 0.312,
        },
        "Electricity": {
            "iTransformer": 0.214,
            "Pathformer": 0.211,
            "PatchTST": 0.214,
            "PDF": 0.199,
            "MoFo (Ours)": 0.191,
        },
        "Solar": {
            "iTransformer": 0.223,
            "Pathformer": 0.208,
            "PatchTST": 0.215,
            "PDF": 0.212,
            "MoFo (Ours)": 0.193,
        },
    },
)


@dataclass(frozen=True, slots=True)
class ChartStyle:
    """Geometry and typography tuned to the 2601 × 2601 reference image."""

    figure_size: tuple[float, float] = (10.404, 10.404)
    x_limits: tuple[float, float] = (-1300.0, 1300.0)
    y_limits: tuple[float, float] = (-1300.0, 1300.0)
    group_degrees: float = 45.0
    group_start_offset: float = 0.50
    bar_width_degrees: float = 6.20
    bar_gap_degrees: float = 1.00
    inner_radius: float = 447.5
    min_outer_radius: float = 671.2
    max_outer_radius: float = 1231.1
    arc_radius: float = 436.0
    arc_width: float = 3.0
    label_radius: float = 366.0
    value_inset: float = 88.0
    dataset_font_size: float = 22.0
    value_font_size: float = 17.0
    legend_font_size: float = 15.0
    highlight_color: str = "#C00000"

    def validate(self) -> None:
        n_models = len(MODELS)
        occupied = (
            n_models * self.bar_width_degrees
            + (n_models - 1) * self.bar_gap_degrees
            + self.group_start_offset
        )
        if occupied >= self.group_degrees:
            raise ValueError("bars plus gaps overflow a 45-degree dataset sector")
        if not 0 < self.inner_radius < self.min_outer_radius < self.max_outer_radius:
            raise ValueError("radii must satisfy 0 < inner < min_outer < max_outer")
        if self.value_inset <= 0:
            raise ValueError("value_inset must be positive")


DEFAULT_STYLE: Final[ChartStyle] = ChartStyle()


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


def _bar_starts(style: ChartStyle) -> NDArray[np.float64]:
    pitch = style.bar_width_degrees + style.bar_gap_degrees
    index = np.arange(len(MODELS), dtype=float)
    starts = []
    for group in range(len(DATASETS)):
        origin = group * style.group_degrees + style.group_start_offset
        starts.append(origin + index * pitch)
    return np.concatenate(starts)


def _inverted_outers(
    values: Sequence[float], style: ChartStyle
) -> NDArray[np.float64]:
    """Map lower scores to longer bars, stretched to the group's min/max."""

    array = np.asarray(values, dtype=float)
    vmin = float(array.min())
    vmax = float(array.max())
    if np.isclose(vmax, vmin):
        return np.full(array.shape, 0.5 * (style.min_outer_radius + style.max_outer_radius))
    span = style.max_outer_radius - style.min_outer_radius
    return style.min_outer_radius + span * (vmax - array) / (vmax - vmin)


def _draw_bars(
    ax: Axes,
    data: GroupedRingBarData,
    palette: Palette,
    style: ChartStyle,
) -> None:
    starts = _bar_starts(style).reshape(len(DATASETS), len(MODELS))
    for dataset, group_starts in zip(DATASETS, starts, strict=True):
        values = [data.mse[dataset][model] for model in MODELS]
        outers = _inverted_outers(values, style)
        for model, start, outer, value in zip(
            MODELS, group_starts, outers, values, strict=True
        ):
            theta1, theta2 = _wedge_thetas(float(start), style.bar_width_degrees)
            color = palette.for_model(model)
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
            bar_len = float(outer) - style.inner_radius
            label_radius = min(
                float(outer) - style.value_inset,
                style.inner_radius + 0.62 * bar_len,
            )
            label_radius = max(label_radius, style.inner_radius + 0.42 * bar_len)
            mid_angle = float(start) + 0.5 * style.bar_width_degrees
            x, y = _xy(mid_angle, label_radius)
            ax.text(
                float(x),
                float(y),
                f"{value:.3f}",
                ha="center",
                va="center",
                rotation=90.0 - mid_angle,
                rotation_mode="anchor",
                fontsize=style.value_font_size,
                color="white",
                fontweight="bold",
                zorder=6,
            )


def _draw_group_arcs(ax: Axes, style: ChartStyle) -> None:
    occupied = (
        len(MODELS) * style.bar_width_degrees
        + (len(MODELS) - 1) * style.bar_gap_degrees
    )
    for group in range(len(DATASETS)):
        start = group * style.group_degrees + style.group_start_offset
        theta1, theta2 = _wedge_thetas(start, occupied)
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


def _draw_dataset_labels(ax: Axes, style: ChartStyle) -> None:
    occupied = (
        len(MODELS) * style.bar_width_degrees
        + (len(MODELS) - 1) * style.bar_gap_degrees
    )
    for index, name in enumerate(DATASETS):
        start = index * style.group_degrees + style.group_start_offset
        end = start + occupied
        pad = 0.07 * (end - start)
        angles = np.linspace(start + pad, end - pad, len(name))
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
                fontsize=style.dataset_font_size,
                fontweight="bold",
                color="black",
                zorder=7,
            )


def _draw_legend(ax: Axes, palette: Palette, style: ChartStyle) -> None:
    box_width = 500.0
    box_height = 455.0
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

    swatch_width = 101.0
    swatch_height = 42.0
    row_pitch = 77.0
    first_y = 160.5
    swatch_x = -220.0
    text_x = -100.0
    for index, model in enumerate(MODELS):
        y = first_y - index * row_pitch
        ax.add_patch(
            Rectangle(
                (swatch_x, y - 0.5 * swatch_height),
                swatch_width,
                swatch_height,
                facecolor=palette.for_model(model),
                edgecolor="none",
                zorder=21,
            )
        )
        highlighted = model == "MoFo (Ours)"
        ax.text(
            text_x,
            y,
            model,
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
    style.validate()

    with plt.rc_context(
        {
            "font.family": [
                "Times New Roman",
                "Times",
                "DejaVu Serif",
                "serif",
            ],
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

        _draw_group_arcs(ax, style)
        _draw_bars(ax, data, palette, style)
        _draw_dataset_labels(ax, style)
        _draw_legend(ax, palette, style)

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
    data: GroupedRingBarData = DEFAULT_DATA,
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
    stem = f"grouped_ring_bar_palette_{index:02d}_{palette.name.replace('-', '_')}"
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
        description="Render the grouped annular bar chart of forecasting MSE."
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
        help="raster DPI; 250 reproduces the 2601×2601 reference (default: 250)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/grouped_ring_bar"),
        help="destination directory (default: output/grouped_ring_bar)",
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
