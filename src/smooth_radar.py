"""Reproduce the smooth-curve radar chart from the Xiaohongshu reference.

The carousel repeats the same 31-task pathology AUROC radar 18 times, changing
only the colour palette.  Five models are drawn as periodic splines; task names
sit in two-line rounded boxes whose colours mark Morphology, Biomarkers, or
Prognosis.

Task names and categories follow Figure 1c of Neidlinger et al. (2026),
Nature Communications, CC BY 4.0.  The source post does not release a table;
:data:`DEFAULT_DATA` holds values digitised from the labelled radii.  Replace
those arrays when using the renderer with another dataset.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Iterable, Mapping, Sequence

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


MODELS: Final[tuple[str, ...]] = (
    "EAGLE",
    "CHIEF",
    "GigaPath",
    "CTransPath",
    "Virchow2",
)

CATEGORIES: Final[tuple[str, ...]] = ("Morphology", "Biomarkers", "Prognosis")

TASKS: Final[tuple[str, ...]] = (
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

TASK_CATEGORY: Final[Mapping[str, str]] = {
    "CPTAC CRC Sidedness": "Morphology",
    "CPTAC NSCLC Subtyping": "Morphology",
    "KIEL STAD LAUREN": "Morphology",
    "BERN STAD LAUREN": "Morphology",
    "DACHS CRC Sidedness": "Morphology",
    "CPTAC CRC KRAS": "Biomarkers",
    "CPTAC CRC PIK3CA": "Biomarkers",
    "CPTAC LUAD KRAS": "Biomarkers",
    "DACHS CRC KRAS": "Biomarkers",
    "CPTAC CRC MSI": "Biomarkers",
    "CPTAC BRCA ESR1": "Biomarkers",
    "DACHS CRC MSI": "Biomarkers",
    "KIEL STAD EBV": "Biomarkers",
    "BERN STAD MSI": "Biomarkers",
    "KIEL STAD MSI": "Biomarkers",
    "DACHS CRC BRAF": "Biomarkers",
    "CPTAC BRCA PGR": "Biomarkers",
    "CPTAC CRC BRAF": "Biomarkers",
    "CPTAC LUAD STK11": "Biomarkers",
    "CPTAC LUAD TP53": "Biomarkers",
    "CPTAC LUAD EGFR": "Biomarkers",
    "CPTAC BRCA ERBB2": "Biomarkers",
    "DACHS CRC CIMP": "Biomarkers",
    "CPTAC BRCA PIK3CA": "Biomarkers",
    "BERN STAD N-STATUS": "Prognosis",
    "CPTAC CRC N-STATUS": "Prognosis",
    "IEO BRCA N-STATUS": "Prognosis",
    "KIEL STAD M-STATUS": "Prognosis",
    "DACHS CRC M-STATUS": "Prognosis",
    "DACHS CRC N-STATUS": "Prognosis",
    "KIEL STAD N-STATUS": "Prognosis",
}


@dataclass(frozen=True, slots=True)
class Palette:
    """Five model colours plus three category colours."""

    name: str
    models: tuple[str, str, str, str, str]
    categories: tuple[str, str, str]

    def for_model(self, model: str) -> str:
        try:
            return self.models[MODELS.index(model)]
        except ValueError as exc:
            raise KeyError(f"unsupported model: {model}") from exc

    def for_category(self, category: str) -> str:
        try:
            return self.categories[CATEGORIES.index(category)]
        except ValueError as exc:
            raise KeyError(f"unsupported category: {category}") from exc


# Model swatches sampled from carousel legends 1-18.  Named qualitative sets
# use canonical ColorBrewer / Tableau / Paul Tol hex values.
PALETTES: Final[tuple[Palette, ...]] = (
    Palette(
        "crimson-peach-ice",
        ("#C83E4D", "#F4B38C", "#D0E1F9", "#85C1E9", "#2E86C1"),
        ("#4B2C7A", "#C24B7A", "#8B3A2F"),
    ),
    Palette(
        "set1",
        ("#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00"),
        ("#FFFF33", "#A65628", "#F781BF"),
    ),
    Palette(
        "dark2",
        ("#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#66A61E"),
        ("#E6AB02", "#A6761D", "#666666"),
    ),
    Palette(
        "set2",
        ("#FC8D62", "#8DA0CB", "#E78AC3", "#A6D854", "#FFD92F"),
        ("#66C2A5", "#E5C494", "#B3B3B3"),
    ),
    Palette(
        "tableau10",
        ("#4E79A7", "#F28E2C", "#E15759", "#76B7B2", "#59A14F"),
        ("#EDC948", "#B07AA1", "#9C755F"),
    ),
    Palette(
        "material-bold",
        ("#D81B60", "#1E88E5", "#FFC107", "#004D40", "#8E24AA"),
        ("#4CAF50", "#AB2F26", "#6D4C41"),
    ),
    Palette(
        "neon-rainbow",
        ("#FF595E", "#FFCA3A", "#8AC926", "#1982C4", "#6A4C93"),
        ("#A7182A", "#C45C26", "#4A4A4A"),
    ),
    Palette(
        "sunset-teal",
        ("#F94144", "#F3722C", "#90BE6D", "#43AA8B", "#577590"),
        ("#F8961E", "#F9C74F", "#AE8B37"),
    ),
    Palette(
        "atlantic-sunset",
        ("#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51"),
        ("#8AB17D", "#81859B", "#5B7470"),
    ),
    Palette(
        "teal-cream-coral",
        ("#0081A7", "#00AFB9", "#FDFCDC", "#FED9B7", "#F07167"),
        ("#D4E09B", "#C3D6B5", "#F4A261"),
    ),
    Palette(
        "tol-muted",
        ("#332288", "#117733", "#44AA99", "#88CCEE", "#DDCC77"),
        ("#CC6677", "#882255", "#AA4499"),
    ),
    Palette(
        "tol-wine",
        ("#CC6677", "#AA4499", "#882255", "#332288", "#DDCC77"),
        ("#117733", "#44AA99", "#30776B"),
    ),
    Palette(
        "venice",
        ("#E63946", "#F1FAEE", "#A8DADC", "#457B9D", "#1D3557"),
        ("#F4A261", "#A24E39", "#1D3557"),
    ),
    Palette(
        "space-red",
        ("#2B2D42", "#8D99AE", "#EDF2F4", "#EF233C", "#D90429"),
        ("#EDF2F4", "#AD5900", "#003049"),
    ),
    Palette(
        "forest-wine",
        ("#386641", "#6A994E", "#A7C957", "#F2E8CF", "#BC4749"),
        ("#847DB2", "#BC4749", "#386641"),
    ),
    Palette(
        "spectral",
        ("#D53E4F", "#FC8D59", "#FEE08B", "#E6F598", "#99D594"),
        ("#3288BD", "#5E4FA2", "#423771"),
    ),
    Palette(
        "set3",
        ("#8DD3C7", "#FFFFB3", "#BEBADA", "#FB8072", "#80B1D3"),
        ("#B3DE69", "#FDB462", "#FCCDE5"),
    ),
    Palette(
        "paired",
        ("#A6CEE3", "#1F78B4", "#B2DF8A", "#33A02C", "#FB9A99"),
        ("#E31A1C", "#FDBF6F", "#6A3D9A"),
    ),
)


@dataclass(frozen=True, slots=True)
class SmoothRadarData:
    """AUROC values aligned with :data:`TASKS` for every model."""

    tasks: tuple[str, ...]
    auroc: Mapping[str, NDArray[np.float64]]

    def validate(self) -> None:
        if self.tasks != TASKS:
            raise ValueError("tasks must match the canonical 31-task order")
        if set(self.auroc) != set(MODELS):
            raise ValueError(f"auroc must contain exactly {MODELS}")
        count = len(self.tasks)
        for model, values in self.auroc.items():
            values = np.asarray(values, dtype=float)
            if values.shape != (count,):
                raise ValueError(
                    f"auroc[{model}] has shape {values.shape}; expected ({count},)"
                )
            if not np.all(np.isfinite(values)):
                raise ValueError(f"auroc[{model}] must be finite")
            if np.any(values < 0.0) or np.any(values > 1.0):
                raise ValueError(f"auroc[{model}] must lie in [0, 1]")


# Digitised from the labelled radii on the carousel.  The published EAGLE
# source spreadsheet uses a different per-task normalisation, so those raw
# fold means do not match the printed two-decimal labels.
DEFAULT_DATA: Final[SmoothRadarData] = SmoothRadarData(
    tasks=TASKS,
    auroc={
        "EAGLE": np.array(
            [
                0.81, 0.83, 0.67, 0.77, 0.72, 0.73, 0.77, 0.89, 0.67, 0.76,
                0.86, 0.85, 0.87, 0.67, 0.91, 0.88, 0.91, 0.76, 0.83, 0.80,
                0.81, 0.74, 0.73, 0.64, 0.76, 0.74, 0.84, 0.82, 0.84, 0.69, 0.67,
            ],
            dtype=float,
        ),
        "CHIEF": np.array(
            [
                0.84, 0.66, 0.71, 0.85, 0.68, 0.66, 0.70, 0.89, 0.82, 0.80,
                0.67, 0.77, 0.77, 0.73, 0.91, 0.66, 0.92, 0.64, 0.69, 0.74,
                0.59, 0.54, 0.65, 0.66, 0.59, 0.81, 0.66, 0.61, 0.77, 0.58, 0.68,
            ],
            dtype=float,
        ),
        "GigaPath": np.array(
            [
                0.57, 0.52, 0.50, 0.51, 0.48, 0.46, 0.52, 0.61, 0.52, 0.54,
                0.52, 0.45, 0.40, 0.61, 0.46, 0.58, 0.62, 0.53, 0.60, 0.53,
                0.36, 0.54, 0.50, 0.50, 0.50, 0.48, 0.57, 0.41, 0.36, 0.58, 0.48,
            ],
            dtype=float,
        ),
        "CTransPath": np.array(
            [
                0.59, 0.53, 0.54, 0.42, 0.48, 0.44, 0.38, 0.65, 0.49, 0.69,
                0.50, 0.47, 0.40, 0.60, 0.54, 0.58, 0.60, 0.52, 0.57, 0.68,
                0.35, 0.56, 0.40, 0.48, 0.49, 0.51, 0.60, 0.40, 0.35, 0.60, 0.57,
            ],
            dtype=float,
        ),
        "Virchow2": np.array(
            [
                0.73, 0.73, 0.64, 0.70, 0.63, 0.62, 0.62, 0.67, 0.78, 0.69,
                0.77, 0.81, 0.71, 0.63, 0.83, 0.82, 0.70, 0.84, 0.69, 0.67,
                0.83, 0.81, 0.76, 0.76, 0.65, 0.84, 0.89, 0.69, 0.84, 0.69, 0.61,
            ],
            dtype=float,
        ),
    },
)


@dataclass(frozen=True, slots=True)
class ChartStyle:
    """Layout tuned to the 2560 × 2243 reference canvas."""

    figure_size: tuple[float, float] = (10.24, 8.972)
    x_limits: tuple[float, float] = (-1.38, 1.770)
    y_limits: tuple[float, float] = (-1.38, 1.38)
    outer_radius: float = 1.00
    center_radius: float = 0.038
    label_radius: float = 1.20
    grid_rings: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0)
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
    category_font_size: float = 9.0


DEFAULT_STYLE: Final[ChartStyle] = ChartStyle()


def split_task_label(task: str) -> tuple[str, str]:
    """Split ``COHORT CANCER TASK`` into two display lines."""

    parts = task.split()
    if len(parts) < 3:
        raise ValueError(f"task label must have at least three tokens: {task!r}")
    return " ".join(parts[:2]), " ".join(parts[2:])


def _angles(count: int) -> NDArray[np.float64]:
    """Return task angles starting at 12 o'clock and moving clockwise."""

    return np.pi / 2.0 - np.arange(count, dtype=float) * 2.0 * np.pi / count


def _smooth_closed_curve(
    values: NDArray[np.float64],
    samples_per_segment: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Periodic cubic spline through the 31 task radii."""

    values = np.asarray(values, dtype=float)
    count = len(values)
    nodes = np.arange(count + 1, dtype=float)
    closed = np.append(values, values[0])
    spline = make_interp_spline(nodes, closed, k=3, bc_type="periodic")
    dense = np.linspace(0.0, float(count), count * samples_per_segment + 1)
    radii = spline(dense)
    theta = np.pi / 2.0 - 2.0 * np.pi * dense / count
    return theta, radii


def _darker(color: str, factor: float = 0.62) -> tuple[float, float, float]:
    return tuple(max(0.0, min(1.0, channel * factor)) for channel in to_rgb(color))


def _draw_grid(ax: Axes, count: int, style: ChartStyle) -> None:
    theta = _angles(count)
    for angle in theta:
        ax.plot(
            [style.center_radius * np.cos(angle), style.outer_radius * np.cos(angle)],
            [style.center_radius * np.sin(angle), style.outer_radius * np.sin(angle)],
            color=style.grid_color,
            linewidth=style.grid_width,
            zorder=1,
        )
    for radius in style.grid_rings:
        width = style.outer_width if radius == style.outer_radius else style.grid_width
        color = style.outer_color if radius == style.outer_radius else style.grid_color
        ax.add_patch(
            Circle(
                (0.0, 0.0),
                radius,
                facecolor="none",
                edgecolor=color,
                linewidth=width,
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
    style: ChartStyle,
) -> None:
    for model in MODELS:
        theta, radii = _smooth_closed_curve(
            data.auroc[model], style.samples_per_segment
        )
        ax.plot(
            radii * np.cos(theta),
            radii * np.sin(theta),
            color=palette.for_model(model),
            linewidth=style.curve_width,
            solid_joinstyle="round",
            solid_capstyle="round",
            zorder=5,
        )


def _draw_values(
    ax: Axes,
    data: SmoothRadarData,
    palette: Palette,
    style: ChartStyle,
) -> None:
    theta = _angles(len(data.tasks))
    halo = [patheffects.withStroke(linewidth=3.2, foreground="white")]
    # Slight angular jitter so stacked labels on one spoke stay readable.
    offsets = {
        "EAGLE": (0.012, 0.0),
        "CHIEF": (-0.010, 0.012),
        "GigaPath": (0.014, -0.010),
        "CTransPath": (-0.016, -0.008),
        "Virchow2": (0.000, 0.014),
    }
    for model in MODELS:
        color = palette.for_model(model)
        dx, dy = offsets[model]
        for angle, value in zip(theta, data.auroc[model], strict=True):
            ax.text(
                value * np.cos(angle) + dx,
                value * np.sin(angle) + dy,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=style.value_font_size,
                color=color,
                fontweight="bold",
                zorder=8,
                path_effects=halo,
            )


def _draw_task_labels(ax: Axes, palette: Palette, style: ChartStyle) -> None:
    theta = _angles(len(TASKS))
    for task, angle in zip(TASKS, theta, strict=True):
        category = TASK_CATEGORY[task]
        color = palette.for_category(category)
        line1, line2 = split_task_label(task)
        ax.text(
            style.label_radius * np.cos(angle),
            style.label_radius * np.sin(angle),
            f"{line1}\n{line2}",
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


def _draw_model_legend(ax: Axes, palette: Palette, style: ChartStyle) -> None:
    handles = [
        Line2D(
            [0],
            [0],
            color=palette.for_model(model),
            linewidth=4.2,
            solid_capstyle="butt",
            label=model,
        )
        for model in MODELS
    ]
    legend = ax.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.995, 0.995),
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


def _draw_category_legend(ax: Axes, palette: Palette, style: ChartStyle) -> None:
    # Data-coordinate stack in the lower-right corner.
    width, height, gap = 0.30, 0.072, 0.018
    x = style.x_limits[1] - width - 0.04
    y0 = style.y_limits[0] + 0.055
    for index, category in enumerate(reversed(CATEGORIES)):
        color = palette.for_category(category)
        y = y0 + index * (height + gap)
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
            category,
            ha="center",
            va="center",
            fontsize=style.category_font_size,
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
    if len(data.tasks) < 3:
        raise ValueError("at least three tasks are required")

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

        _draw_grid(ax, len(data.tasks), style)
        _draw_curves(ax, data, palette, style)
        _draw_values(ax, data, palette, style)
        _draw_task_labels(ax, palette, style)
        _draw_model_legend(ax, palette, style)
        _draw_category_legend(ax, palette, style)

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
    data: SmoothRadarData = DEFAULT_DATA,
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
    stem = f"smooth_radar_palette_{index:02d}_{palette.name.replace('-', '_')}"
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
        description="Render the 31-task smooth-curve pathology radar chart."
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
        help="raster DPI; 250 reproduces the 2560×2243 reference (default: 250)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/smooth_radar"),
        help="destination directory (default: output/smooth_radar)",
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
            models = " ".join(palette.models)
            cats = " ".join(palette.categories)
            print(f"{index:2d}  {palette.name:20s}  {models}  |  {cats}")
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
