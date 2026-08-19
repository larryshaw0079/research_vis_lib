"""Reproduce the grouped gradient histogram with normal-fit overlays.

The Xiaohongshu carousel repeats the same residual-strain histogram 18
times, changing only the three-colour palette.  Each panel has:

* grouped frequency bars for Zhang, Kioumarsi, and Xue, with a vertical
  colour-to-white gradient on every bar;
* dashed normal-density curves scaled to the frequency axis;
* a top-left legend that prints each group's mean and standard deviation.

The source post publishes the chart but not the residual table.
``DEFAULT_DATA`` reconstructs one residual per visible bar count; the
printed means and standard deviations are taken from the legend.
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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MultipleLocator
from numpy.typing import NDArray
from scipy.stats import norm


GROUPS: Final[tuple[str, ...]] = ("Zhang et al.", "Kioumarsi et al.", "Xue et al.")

BIN_EDGES: Final[tuple[float, ...]] = (
    -106.8,
    -91.6,
    -76.3,
    -61.1,
    -45.8,
    -30.5,
    -15.3,
    0.0,
    15.3,
    30.5,
    45.8,
    61.1,
    76.3,
    91.6,
    106.8,
)

TARGET_STATS: Final[Mapping[str, tuple[float, float]]] = {
    "Zhang et al.": (-4.9, 14.4),
    "Kioumarsi et al.": (7.6, 25.1),
    "Xue et al.": (-3.3, 10.6),
}

# Digitised from the visible bar heights in the first carousel frame.
TARGET_COUNTS: Final[Mapping[str, tuple[int, ...]]] = {
    "Zhang et al.": (0, 0, 0, 0, 3, 15, 34, 21, 6, 1, 0, 0, 0, 0),
    "Kioumarsi et al.": (0, 0, 1, 2, 6, 8, 23, 23, 23, 10, 5, 1, 1, 0),
    "Xue et al.": (0, 0, 0, 0, 0, 11, 33, 23, 3, 0, 0, 0, 0, 0),
}

GROUP_SIZES: Final[Mapping[str, int]] = {
    group: int(sum(counts)) for group, counts in TARGET_COUNTS.items()
}


@dataclass(frozen=True, slots=True)
class Palette:
    """Three group colours in :data:`GROUPS` order."""

    name: str
    colors: tuple[str, str, str]

    def for_group(self, group: str) -> str:
        try:
            return self.colors[GROUPS.index(group)]
        except ValueError as exc:
            raise KeyError(f"unsupported group: {group}") from exc


# Colours reconstructed from the legend swatches in carousel images 1-18.
# JPEG compression was snapped to nearby saturated hex values.
PALETTES: Final[tuple[Palette, ...]] = (
    Palette("coral-navy-teal", ("#E74D40", "#3B4A6B", "#1D8C85")),
    Palette("watermelon-azure-lime", ("#FE5A5F", "#1A82C4", "#8CC929")),
    Palette("apricot-plum-sea", ("#FE9351", "#5E4D7C", "#55A788")),
    Palette("crimson-navy-amber", ("#D62828", "#002F49", "#F67F04")),
    Palette("rose-cyan-mint", ("#EE4970", "#108AB3", "#07D6A0")),
    Palette("burgundy-pine-wine", ("#990520", "#114C5C", "#601040")),
    Palette("terracotta-slate-sage", ("#DF7B60", "#3C405D", "#81B39A")),
    Palette("scarlet-navy-steel", ("#E33A47", "#1D3557", "#457B9D")),
    Palette("coral-charcoal-steel", ("#EE6C50", "#293241", "#3D5A80")),
    Palette("tomato-lake-sea", ("#F94245", "#277CA1", "#45AA8C")),
    Palette("brick-cobalt-forest", ("#C23027", "#023F92", "#0E7D59")),
    Palette("peach-teal-mint", ("#E2967B", "#026D77", "#85C4BE")),
    Palette("crimson-ink-slate", ("#D8062A", "#2B2D42", "#8D99AE")),
    Palette("cardinal-navy-sky", ("#BF1420", "#003049", "#679BBB")),
    Palette("red-navy-olive", ("#D5293A", "#202D5A", "#5A8157")),
    Palette("salmon-petrol-teal", ("#DA514D", "#094C63", "#187E89")),
    Palette("coral-pine-sea", ("#FE715C", "#254343", "#45AA8C")),
    Palette("wine-azure-emerald", ("#B43A51", "#2274A4", "#34936E")),
)


@dataclass(frozen=True, slots=True)
class GroupedHistData:
    """Residual samples aligned with :data:`GROUPS`."""

    groups: tuple[str, ...]
    values: Mapping[str, NDArray[np.float64]]
    means: Mapping[str, float] | None = None
    stds: Mapping[str, float] | None = None

    def validate(self) -> None:
        if self.groups != GROUPS:
            raise ValueError("groups must match the canonical three-study order")
        if set(self.values) != set(self.groups):
            raise ValueError("values must contain one array per group")
        if self.means is not None and set(self.means) != set(self.groups):
            raise ValueError("means must contain one value per group")
        if self.stds is not None and set(self.stds) != set(self.groups):
            raise ValueError("stds must contain one value per group")
        for group in self.groups:
            sample = np.asarray(self.values[group], dtype=float)
            if sample.ndim != 1 or sample.size < 3:
                raise ValueError(f"{group} values must be 1-D and of length >= 3")
            if not np.all(np.isfinite(sample)):
                raise ValueError(f"{group} values must be finite")
            if self.stds is not None and self.stds[group] <= 0:
                raise ValueError(f"{group} standard deviation must be positive")

    def sample(self, group: str) -> NDArray[np.float64]:
        return np.asarray(self.values[group], dtype=float)

    def stats(self, group: str) -> tuple[float, float]:
        if self.means is not None and self.stds is not None:
            return float(self.means[group]), float(self.stds[group])
        sample = self.sample(group)
        return float(sample.mean()), float(sample.std(ddof=0))


@dataclass(frozen=True, slots=True)
class ChartStyle:
    """Layout tuned to the 2121 × 1762 reference image."""

    figure_size: tuple[float, float] = (8.484, 7.048)
    axes_bounds: tuple[float, float, float, float] = (0.118, 0.168, 0.842, 0.768)
    y_limits: tuple[float, float] = (0.0, 50.0)
    bar_width_fraction: float = 0.26
    bar_gap_fraction: float = 0.04
    gradient_steps: int = 256
    line_width: float = 2.15
    spine_width: float = 1.55
    tick_length: float = 5.2
    minor_tick_length: float = 2.8
    label_font_size: float = 16.0
    tick_font_size: float = 10.0
    legend_font_size: float = 10.5
    panel_font_size: float = 20.0
    x_label_pad: float = 10.0
    y_label_pad: float = 8.0

    def validate(self) -> None:
        if min(self.figure_size) <= 0:
            raise ValueError("figure_size must be positive")
        if self.y_limits[1] <= self.y_limits[0]:
            raise ValueError("y_limits must be increasing")
        if not 0 < self.bar_width_fraction < 0.4:
            raise ValueError("bar_width_fraction must sit in (0, 0.4)")
        if self.gradient_steps < 16:
            raise ValueError("gradient_steps must be at least 16")


DEFAULT_STYLE: Final[ChartStyle] = ChartStyle()


def _samples_from_counts(
    counts: Sequence[int],
    edges: Sequence[float],
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Place the digitised bin counts uniformly inside each interval."""

    pieces: list[NDArray[np.float64]] = []
    for count, left, right in zip(counts, edges[:-1], edges[1:], strict=True):
        if count <= 0:
            continue
        pieces.append(rng.uniform(left, right, count))
    if not pieces:
        raise ValueError("at least one bin must have a positive count")
    return np.concatenate(pieces)


def _build_default_data(seed: int = 20260819) -> GroupedHistData:
    rng = np.random.default_rng(seed)
    values: dict[str, NDArray[np.float64]] = {}
    for group in GROUPS:
        values[group] = _samples_from_counts(TARGET_COUNTS[group], BIN_EDGES, rng)
    return GroupedHistData(
        groups=GROUPS,
        values=values,
        means={group: stats[0] for group, stats in TARGET_STATS.items()},
        stds={group: stats[1] for group, stats in TARGET_STATS.items()},
    )


DEFAULT_DATA: Final[GroupedHistData] = _build_default_data()

BIN_LABELS: Final[tuple[str, ...]] = tuple(
    f"[{BIN_EDGES[index]:.1f}, {BIN_EDGES[index + 1]:.1f})"
    for index in range(len(BIN_EDGES) - 1)
)


def histogram_counts(
    values: NDArray[np.float64],
    edges: Sequence[float] = BIN_EDGES,
) -> NDArray[np.int64]:
    counts, _ = np.histogram(values, bins=np.asarray(edges, dtype=float))
    return counts.astype(np.int64)


def format_legend_label(group: str, mean: float, std: float) -> str:
    return f"{group} ($\\mu = {mean:.1f}$, $\\sigma = {std:.1f}$)"


def _bin_width(edges: Sequence[float] = BIN_EDGES) -> float:
    array = np.asarray(edges, dtype=float)
    return float(np.mean(np.diff(array)))


def _gradient_cmap(color: str) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "bar_gradient",
        [(0.0, "#FFFFFF"), (0.18, "#FFFFFF"), (1.0, color)],
        N=256,
    )


def _draw_gradient_bar(
    ax: Axes,
    x: float,
    height: float,
    width: float,
    color: str,
    style: ChartStyle,
) -> None:
    if height <= 0:
        return
    gradient = np.linspace(0.0, 1.0, style.gradient_steps).reshape(-1, 1)
    image = ax.imshow(
        gradient,
        extent=(x - 0.5 * width, x + 0.5 * width, 0.0, height),
        origin="lower",
        aspect="auto",
        cmap=_gradient_cmap(color),
        interpolation="bicubic",
        zorder=2,
        clip_on=True,
    )
    image.set_clip_path(ax.patch)
    ax.add_patch(
        Rectangle(
            (x - 0.5 * width, 0.0),
            width,
            height,
            fill=False,
            edgecolor=color,
            linewidth=0.35,
            zorder=3,
        )
    )


def _style_axis(ax: Axes, style: ChartStyle) -> None:
    ax.set_ylim(*style.y_limits)
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(2))
    ax.tick_params(
        which="major",
        direction="in",
        top=True,
        right=True,
        length=style.tick_length,
        width=style.spine_width,
        labelsize=style.tick_font_size,
        pad=4.0,
    )
    ax.tick_params(
        which="minor",
        direction="in",
        top=True,
        right=True,
        left=True,
        length=style.minor_tick_length,
        width=style.spine_width * 0.75,
    )
    ax.set_xlabel(
        r"Residual Group ($\mu\epsilon$)",
        fontsize=style.label_font_size,
        fontweight="bold",
        labelpad=style.x_label_pad,
    )
    ax.set_ylabel(
        "Frequency",
        fontsize=style.label_font_size,
        fontweight="bold",
        labelpad=style.y_label_pad,
    )
    for label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        label.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(style.spine_width)
        spine.set_color("black")
    ax.set_axisbelow(False)
    ax.grid(False)


def _draw_bars(
    ax: Axes,
    data: GroupedHistData,
    palette: Palette,
    style: ChartStyle,
) -> Mapping[str, NDArray[np.int64]]:
    centers = np.arange(len(BIN_LABELS), dtype=float)
    width = style.bar_width_fraction
    gap = style.bar_gap_fraction
    offsets = (-(width + gap), 0.0, width + gap)
    counts: dict[str, NDArray[np.int64]] = {}
    for group, offset in zip(GROUPS, offsets, strict=True):
        group_counts = histogram_counts(data.sample(group))
        counts[group] = group_counts
        color = palette.for_group(group)
        for x, height in zip(centers, group_counts, strict=True):
            _draw_gradient_bar(ax, x + offset, float(height), width, color, style)
    return counts


def _draw_fits(
    ax: Axes,
    data: GroupedHistData,
    palette: Palette,
    style: ChartStyle,
) -> None:
    residual_grid = np.linspace(BIN_EDGES[0], BIN_EDGES[-1], 400)
    axis_grid = np.interp(residual_grid, BIN_EDGES, np.arange(len(BIN_EDGES), dtype=float))
    bin_width = _bin_width()
    for group in GROUPS:
        sample = data.sample(group)
        mean, std = data.stats(group)
        density = norm.pdf(residual_grid, loc=mean, scale=std)
        ax.plot(
            axis_grid,
            density * sample.size * bin_width,
            linestyle=(0, (4.6, 2.8)),
            color=palette.for_group(group),
            linewidth=style.line_width,
            solid_capstyle="round",
            dash_capstyle="round",
            zorder=4,
        )


def _draw_legend(ax: Axes, data: GroupedHistData, palette: Palette, style: ChartStyle) -> None:
    handles: list[Patch | Line2D] = []
    labels: list[str] = []
    for group in GROUPS:
        mean, std = data.stats(group)
        handles.append(Patch(facecolor=palette.for_group(group), edgecolor="none"))
        labels.append(format_legend_label(group, mean, std))
    handles.append(
        Line2D(
            [0],
            [0],
            color="#7A7A7A",
            linestyle=(0, (4.6, 2.8)),
            linewidth=style.line_width,
        )
    )
    labels.append("Normal fits")
    legend = ax.legend(
        handles,
        labels,
        loc="upper left",
        frameon=True,
        fancybox=False,
        edgecolor="black",
        facecolor="white",
        framealpha=1.0,
        borderpad=0.55,
        labelspacing=0.45,
        handlelength=1.35,
        handletextpad=0.55,
        fontsize=style.legend_font_size,
    )
    legend.get_frame().set_linewidth(0.9)
    for text in legend.get_texts():
        text.set_fontweight("bold")


def create_figure(
    palette: Palette = PALETTES[0],
    data: GroupedHistData = DEFAULT_DATA,
    style: ChartStyle = DEFAULT_STYLE,
) -> Figure:
    """Create the grouped gradient histogram without writing it to disk."""

    data.validate()
    style.validate()

    with plt.rc_context(
        {
            "font.family": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
            "font.size": style.tick_font_size,
            "axes.linewidth": style.spine_width,
            "mathtext.fontset": "stix",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    ):
        figure = plt.figure(figsize=style.figure_size, facecolor="white")
        ax = figure.add_axes(style.axes_bounds)
        ax.set_facecolor("white")
        _style_axis(ax, style)
        _draw_bars(ax, data, palette, style)
        _draw_fits(ax, data, palette, style)
        _draw_legend(ax, data, palette, style)

        centers = np.arange(len(BIN_LABELS), dtype=float)
        ax.set_xlim(-0.75, len(BIN_LABELS) - 0.25)
        ax.set_xticks(centers)
        ax.set_xticklabels(BIN_LABELS, rotation=45, ha="right", rotation_mode="anchor")
        for label in ax.get_xticklabels():
            label.set_fontweight("bold")
            label.set_fontsize(style.tick_font_size)
        ax.text(
            0.985,
            0.965,
            "(a)",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=style.panel_font_size,
            fontweight="bold",
            zorder=6,
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
    dpi: int = 250,
    data: GroupedHistData = DEFAULT_DATA,
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
    stem = f"grouped_gradient_hist_palette_{index:02d}_{palette.name.replace('-', '_')}"
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
        description="Render the grouped gradient histogram with normal-fit overlays."
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
        help="raster DPI; 250 reproduces the 2121×1762 reference (default: 250)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/grouped_gradient_hist"),
        help="destination directory (default: output/grouped_gradient_hist)",
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
