"""Reproduce the cartoon stacked-bar policy-mix chart.

The Xiaohongshu carousel repeats the same nine-bar figure 16 times, changing
only the colour palette.  Each bar is a stack of capsule segments with white
gaps: three one-policy bars, three dual-policy (A ∪ B) bars, and three
multi-policy (A ∪ B ∪ C) bars.  Trade-offs and synergies are the extreme
cases called out in the title; the Unrelated bars are the additive baseline
without the A∩B overlap.

The source post publishes the chart but not the underlying table.
``DEFAULT_DATA`` stores integer segment heights that sum to the labelled
totals (48, 54, 60, 64, 78, 86, 90, 114, 130).  Replace that mapping when
using the renderer with another dataset.
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
from matplotlib.patches import FancyBboxPatch


SERIES: Final[tuple[str, ...]] = (
    "policy_a",
    "policy_b",
    "policy_c",
    "overlap_cyan",
    "overlap_purple",
    "overlap_yellow",
    "overlap_green",
    "tradeoffs_missing",
    "synergy_bonus",
)

SERIES_LABELS: Final[Mapping[str, str]] = {
    "policy_a": "Policy A Base",
    "policy_b": "Policy B Base",
    "policy_c": "Policy C Base",
    "overlap_cyan": "Overlap (Cyan)",
    "overlap_green": "Overlap (Green)",
    "overlap_purple": "Overlap (Purple)",
    "overlap_yellow": "Overlap (Yellow)",
    "tradeoffs_missing": "Trade-offs Missing",
    "synergy_bonus": "Synergy Bonus",
}

# Row-major 3×3 order matching the carousel legend.
LEGEND_ORDER: Final[tuple[str, ...]] = (
    "policy_a",
    "overlap_cyan",
    "overlap_purple",
    "overlap_green",
    "policy_b",
    "overlap_yellow",
    "policy_c",
    "tradeoffs_missing",
    "synergy_bonus",
)

BAR_IDS: Final[tuple[str, ...]] = (
    "policy_a",
    "policy_b",
    "policy_c",
    "dual_tradeoffs",
    "dual_unrelated",
    "dual_synergies",
    "multi_tradeoffs",
    "multi_unrelated",
    "multi_synergies",
)

BAR_LABELS: Final[Mapping[str, str]] = {
    "policy_a": "Policy A",
    "policy_b": "Policy B",
    "policy_c": "Policy C",
    "dual_tradeoffs": "Trade-offs",
    "dual_unrelated": "Unrelated",
    "dual_synergies": "Synergies",
    "multi_tradeoffs": "Trade-offs",
    "multi_unrelated": "Unrelated",
    "multi_synergies": "Synergies",
}

GROUPS: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    ("One-Policy (A/B/C)", ("policy_a", "policy_b", "policy_c")),
    ("Dual-Policies (A ∪ B)", ("dual_tradeoffs", "dual_unrelated", "dual_synergies")),
    (
        "Multi-Policies (A ∪ B ∪ C)",
        ("multi_tradeoffs", "multi_unrelated", "multi_synergies"),
    ),
)

# Exclusive bases plus four Venn overlaps.  Dual Unrelated omits A∩B;
# Dual Synergies adds it back; Multi Trade-offs keeps only the three bases
# and draws the missing overlaps as a hollow dashed capsule.
TARGET_TOTALS: Final[Mapping[str, int]] = {
    "policy_a": 48,
    "policy_b": 54,
    "policy_c": 60,
    "dual_tradeoffs": 64,
    "dual_unrelated": 78,
    "dual_synergies": 86,
    "multi_tradeoffs": 90,
    "multi_unrelated": 114,
    "multi_synergies": 130,
}

STACK_ORDER: Final[tuple[str, ...]] = (
    "policy_a",
    "policy_b",
    "policy_c",
    "overlap_cyan",
    "overlap_purple",
    "overlap_yellow",
    "overlap_green",
    "tradeoffs_missing",
    "synergy_bonus",
)


@dataclass(frozen=True, slots=True)
class Palette:
    """Seven fill colours plus the Trade-offs Missing dash colour.

    Order: Policy A, Policy B, Policy C, Overlap (Cyan), Overlap (Green),
    Overlap (Purple), Overlap (Yellow), missing-edge.  Synergy Bonus reuses
    the Policy A colour.
    """

    name: str
    colors: tuple[str, str, str, str, str, str, str, str]

    def for_series(self, series: str) -> str:
        mapping = {
            "policy_a": self.colors[0],
            "policy_b": self.colors[1],
            "policy_c": self.colors[2],
            "overlap_cyan": self.colors[3],
            "overlap_green": self.colors[4],
            "overlap_purple": self.colors[5],
            "overlap_yellow": self.colors[6],
            "synergy_bonus": self.colors[0],
            "tradeoffs_missing": self.colors[7],
        }
        try:
            return mapping[series]
        except KeyError as exc:
            raise KeyError(f"unsupported series: {series}") from exc


# Colours sampled from the 3×3 legend pills in carousel frames 1-16.
# Named qualitative sets use canonical ColorBrewer / Coolors hex values;
# the rest are median fills snapped to nearby saturated hex codes.
PALETTES: Final[tuple[Palette, ...]] = (
    Palette(
        "sky-cream-mint",
        ("#5CA2FA", "#FDF6C2", "#B8E9C9", "#90C3A5", "#ABE5E9", "#FE9F9B", "#CACAFC", "#3FB882"),
    ),
    Palette(
        "pastel1",
        ("#FBB4AE", "#FED9A6", "#E5D8BD", "#B3CDE3", "#DECBE4", "#CCEBC5", "#FFFFCC", "#F1B6D0"),
    ),
    Palette(
        "mint-peach-pastel",
        ("#B3E3CD", "#E5F4C9", "#F2E2CB", "#EAC9AE", "#F5CAE5", "#CAD5E7", "#FFF2AE", "#E8B48A"),
    ),
    Palette(
        "brbg",
        ("#8C510A", "#C7EAE5", "#35978F", "#BF812D", "#F6E8C3", "#DFC27D", "#80CDC1", "#01665E"),
    ),
    Palette(
        "rdbu",
        ("#B2182B", "#D1E5F0", "#4393C3", "#D6604D", "#FDDBC7", "#F4A582", "#92C5DE", "#2166AC"),
    ),
    Palette(
        "piyg",
        ("#C51B7D", "#A1D99B", "#F1B6DA", "#E9A3C9", "#E6F5D0", "#FDE0EF", "#4D9221", "#4D9221"),
    ),
    Palette(
        "navy-magenta-orange",
        ("#003F5C", "#D45087", "#FF7C43", "#2F4B7C", "#A05195", "#665191", "#F95D6A", "#FFA600"),
    ),
    Palette(
        "navy-cream-coral",
        ("#01429E", "#FFF6C8", "#F4777D", "#4A71B1", "#A5D5D9", "#74A2C6", "#FFBCAF", "#C24566"),
    ),
    Palette(
        "ink-cream-aqua",
        ("#22223A", "#F2E9E4", "#0DA4AF", "#4B4D68", "#CAADA7", "#9A8C9B", "#B5E1FA", "#E6DC9F"),
    ),
    Palette(
        "taupe-olive",
        ("#CB997E", "#A5A58D", "#3F423B", "#DCBEAA", "#B8B7A3", "#FFE7D6", "#6B705C", "#D0AB8F"),
    ),
    Palette(
        "pacific-sunset",
        ("#001219", "#E9D8A6", "#CA6702", "#005F73", "#94D2BD", "#0A9396", "#EE9B00", "#BB3E03"),
    ),
    Palette(
        "navy-sky-amber",
        ("#8ECAE6", "#FB8500", "#023047", "#219EBC", "#FFB703", "#3D405B", "#E76F51", "#8AB5A0"),
    ),
    Palette(
        "candy-pastel",
        ("#CDB4DB", "#A2D2FF", "#FDFFB6", "#E7C2D4", "#BDE0FE", "#FFAFCC", "#FFD7A6", "#B8E8B0"),
    ),
    Palette(
        "lime-teal",
        ("#D9ED92", "#52B69A", "#168AAD", "#B5E48C", "#76C893", "#99D98C", "#34A0A4", "#1A759F"),
    ),
    Palette(
        "mediterranean",
        ("#264653", "#E76F51", "#457B9D", "#2A9D8F", "#F4A261", "#E9C46A", "#1D3557", "#80CBC4"),
    ),
    Palette(
        "olive-rust-rose",
        ("#5F6C37", "#BC6C25", "#E3989B", "#283618", "#DCA15D", "#FFF6D6", "#B5838E", "#E8A598"),
    ),
)


@dataclass(frozen=True, slots=True)
class CartoonBarData:
    """Segment heights for the nine labelled bars."""

    bars: Mapping[str, Mapping[str, float]]

    def validate(self) -> None:
        if tuple(self.bars) != BAR_IDS:
            raise ValueError("bars must use the canonical nine-bar order")
        for bar_id, segments in self.bars.items():
            unknown = set(segments) - set(SERIES)
            if unknown:
                raise ValueError(f"{bar_id} has unknown series: {sorted(unknown)}")
            if any(value < 0 or not np.isfinite(value) for value in segments.values()):
                raise ValueError(f"{bar_id} values must be finite and non-negative")
            if self.labelled_total(bar_id) <= 0:
                raise ValueError(f"{bar_id} must have a positive labelled total")

    def labelled_total(self, bar_id: str) -> float:
        return float(
            sum(
                value
                for series, value in self.bars[bar_id].items()
                if series != "tradeoffs_missing" and value > 0
            )
        )

    def stack(self, bar_id: str) -> list[tuple[str, float]]:
        segments = self.bars[bar_id]
        return [
            (series, float(segments[series]))
            for series in STACK_ORDER
            if series in segments and segments[series] > 0
        ]


DEFAULT_DATA: Final[CartoonBarData] = CartoonBarData(
    bars={
        "policy_a": {
            "policy_a": 24,
            "overlap_cyan": 8,
            "overlap_purple": 8,
            "overlap_green": 8,
        },
        "policy_b": {
            "policy_b": 30,
            "overlap_cyan": 8,
            "overlap_yellow": 8,
            "overlap_green": 8,
        },
        "policy_c": {
            "policy_c": 36,
            "overlap_purple": 8,
            "overlap_yellow": 8,
            "overlap_green": 8,
        },
        "dual_tradeoffs": {
            "policy_a": 20,
            "policy_b": 26,
            "overlap_purple": 6,
            "overlap_yellow": 6,
            "overlap_green": 6,
        },
        "dual_unrelated": {
            "policy_a": 24,
            "policy_b": 30,
            "overlap_purple": 8,
            "overlap_yellow": 8,
            "overlap_green": 8,
        },
        "dual_synergies": {
            "policy_a": 24,
            "policy_b": 30,
            "overlap_cyan": 8,
            "overlap_purple": 8,
            "overlap_yellow": 8,
            "overlap_green": 8,
        },
        "multi_tradeoffs": {
            "policy_a": 24,
            "policy_b": 30,
            "policy_c": 36,
            "tradeoffs_missing": 24,
        },
        "multi_unrelated": {
            "policy_a": 24,
            "policy_b": 30,
            "policy_c": 36,
            "overlap_purple": 8,
            "overlap_yellow": 8,
            "overlap_green": 8,
        },
        "multi_synergies": {
            "policy_a": 24,
            "policy_b": 30,
            "policy_c": 36,
            "overlap_cyan": 8,
            "overlap_purple": 8,
            "overlap_yellow": 8,
            "overlap_green": 8,
            "synergy_bonus": 8,
        },
    }
)


@dataclass(frozen=True, slots=True)
class ChartStyle:
    """Layout tuned to the 1916 × 1080 reference image."""

    figure_size: tuple[float, float] = (12.774, 7.2)
    axes_bounds: tuple[float, float, float, float] = (0.074, 0.138, 0.912, 0.792)
    x_limits: tuple[float, float] = (-0.85, 11.45)
    y_limits: tuple[float, float] = (-18.0, 168.0)
    y_max: float = 150.0
    bar_width: float = 0.62
    bar_gap: float = 1.25
    title: str = "Samples"
    title_note: str = "(Synergy and trade-offs only list the most extreme cases)"
    y_label: str = "Effect Score"
    title_font_size: float = 16.5
    note_font_size: float = 13.0
    label_font_size: float = 12.5
    tick_font_size: float = 11.5
    legend_font_size: float = 10.8
    total_font_size: float = 13.5
    group_font_size: float = 13.0
    frame_color: str = "#B0B0B0"
    grid_color: str = "#C8C8C8"
    shadow_color: str = "#D6D6D6"
    missing_fill: str = "#FFFFFF"

    def validate(self) -> None:
        if min(self.figure_size) <= 0:
            raise ValueError("figure_size must be positive")
        if self.x_limits[1] <= self.x_limits[0] or self.y_limits[1] <= self.y_limits[0]:
            raise ValueError("axis limits must be increasing")
        if self.y_max <= 0:
            raise ValueError("y_max must be positive")
        if self.bar_width <= 0:
            raise ValueError("bar_width must be positive")


DEFAULT_STYLE: Final[ChartStyle] = ChartStyle()

_BAR_X: Final[tuple[float, ...]] = (0.0, 1.15, 2.30, 4.15, 5.30, 6.45, 8.30, 9.45, 10.60)


def _bar_x(bar_id: str) -> float:
    return _BAR_X[BAR_IDS.index(bar_id)]


def _mutation_aspect(ax: Axes) -> float:
    bbox = ax.get_window_extent()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    if bbox.height <= 0:
        return 1.0
    return (bbox.width / (xlim[1] - xlim[0])) / (bbox.height / (ylim[1] - ylim[0]))


def _pill(
    ax: Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    facecolor: str,
    edgecolor: str,
    linestyle: str | tuple[int, tuple[int, ...]] = "solid",
    linewidth: float = 0.0,
    zorder: float = 3,
    shadow: bool = False,
    style: ChartStyle,
) -> None:
    aspect = _mutation_aspect(ax)
    rounding = min(0.48 * width, 0.48 * height / max(aspect, 1e-6))
    boxstyle = f"round,pad=0,rounding_size={rounding}"
    if shadow:
        for offset, alpha_color in (
            (1.05, "#E4E4E4"),
            (0.70, "#D8D8D8"),
        ):
            ax.add_patch(
                FancyBboxPatch(
                    (x + 0.028, y - offset),
                    width,
                    height,
                    boxstyle=boxstyle,
                    mutation_aspect=aspect,
                    facecolor=alpha_color,
                    edgecolor="none",
                    linewidth=0,
                    zorder=zorder - 0.4,
                )
            )
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle=boxstyle,
            mutation_aspect=aspect,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linestyle=linestyle,
            linewidth=linewidth,
            zorder=zorder,
        )
    )


def _style_axis(ax: Axes, style: ChartStyle) -> None:
    ax.set_xlim(*style.x_limits)
    ax.set_ylim(*style.y_limits)
    ax.set_yticks(np.arange(0.0, style.y_max + 1.0, 30.0))
    ax.set_xticks([])
    ax.tick_params(axis="y", length=0, labelsize=style.tick_font_size, colors="#333333", pad=4)
    ax.set_ylabel(
        style.y_label,
        fontsize=style.label_font_size,
        fontweight="bold",
        labelpad=10,
        color="#222222",
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.yaxis.grid(True, linestyle=(0, (3.2, 3.4)), color=style.grid_color, linewidth=0.95, zorder=0)
    ax.set_axisbelow(True)
    ax.set_facecolor("white")
    for label in ax.get_yticklabels():
        label.set_color("#333333")


def _draw_frame(ax: Axes, style: ChartStyle) -> None:
    aspect = _mutation_aspect(ax)
    ax.add_patch(
        FancyBboxPatch(
            (style.x_limits[0] + 0.08, -6.5),
            style.x_limits[1] - style.x_limits[0] - 0.18,
            style.y_max + 20.5,
            boxstyle="round,pad=0,rounding_size=0.18",
            mutation_aspect=aspect,
            facecolor="none",
            edgecolor=style.frame_color,
            linestyle=(0, (4.5, 3.6)),
            linewidth=1.15,
            zorder=0.5,
            clip_on=False,
        )
    )


def _draw_title(ax: Axes, style: ChartStyle) -> None:
    ax.text(
        -0.35,
        158.5,
        style.title,
        ha="left",
        va="center",
        fontsize=style.title_font_size,
        fontweight="bold",
        color="#111111",
        zorder=6,
        clip_on=False,
    )
    ax.text(
        1.22,
        158.5,
        style.title_note,
        ha="left",
        va="center",
        fontsize=style.note_font_size,
        fontweight="normal",
        color="#222222",
        zorder=6,
        clip_on=False,
    )


def _draw_legend(ax: Axes, palette: Palette, style: ChartStyle) -> None:
    left, bottom, width, height = -0.38, 109.5, 5.35, 40.0
    aspect = _mutation_aspect(ax)
    ax.add_patch(
        FancyBboxPatch(
            (left, bottom),
            width,
            height,
            boxstyle="round,pad=0,rounding_size=0.16",
            mutation_aspect=aspect,
            facecolor="white",
            edgecolor="#9A9A9A",
            linewidth=1.05,
            zorder=5,
        )
    )
    col_x = (left + 0.16, left + 1.88, left + 3.60)
    row_y = (bottom + 29.2, bottom + 16.8, bottom + 4.5)
    pill_w, pill_h = 0.40, 7.0
    for index, series in enumerate(LEGEND_ORDER):
        column, row = divmod(index, 3)
        x = col_x[row]
        y = row_y[column]
        color = palette.for_series(series)
        if series == "tradeoffs_missing":
            _pill(
                ax,
                x,
                y,
                pill_w,
                pill_h,
                facecolor=style.missing_fill,
                edgecolor=color,
                linestyle=(0, (2.6, 1.8)),
                linewidth=1.55,
                zorder=6,
                style=style,
            )
        else:
            _pill(
                ax,
                x,
                y,
                pill_w,
                pill_h,
                facecolor=color,
                edgecolor="none",
                zorder=6,
                style=style,
            )
        ax.text(
            x + pill_w + 0.10,
            y + 0.5 * pill_h,
            SERIES_LABELS[series],
            ha="left",
            va="center",
            fontsize=style.legend_font_size,
            color="#222222",
            zorder=6,
        )


def _draw_bars(
    ax: Axes,
    data: CartoonBarData,
    palette: Palette,
    style: ChartStyle,
) -> None:
    width = style.bar_width
    for bar_id in BAR_IDS:
        x = _bar_x(bar_id) - 0.5 * width
        bottom = 0.0
        stack = data.stack(bar_id)
        for series, value in stack:
            if series == "tradeoffs_missing":
                _pill(
                    ax,
                    x,
                    bottom,
                    width,
                    value,
                    facecolor=style.missing_fill,
                    edgecolor=palette.for_series(series),
                    linestyle=(0, (3.4, 2.4)),
                    linewidth=1.65,
                    zorder=3.2,
                    shadow=False,
                    style=style,
                )
            else:
                _pill(
                    ax,
                    x,
                    bottom,
                    width,
                    value,
                    facecolor=palette.for_series(series),
                    edgecolor="#FFFFFF",
                    linewidth=0.55,
                    zorder=3,
                    shadow=True,
                    style=style,
                )
            bottom += value + style.bar_gap
        labelled = data.labelled_total(bar_id)
        top = bottom - (style.bar_gap if stack else 0.0)
        ax.text(
            x + 0.5 * width,
            top + 4.6,
            f"{labelled:.0f}",
            ha="center",
            va="bottom",
            fontsize=style.total_font_size,
            fontweight="bold",
            color="#111111",
            zorder=6,
        )


def _draw_x_labels(ax: Axes, style: ChartStyle) -> None:
    for bar_id in BAR_IDS:
        ax.text(
            _bar_x(bar_id),
            -7.2,
            BAR_LABELS[bar_id],
            ha="center",
            va="top",
            fontsize=style.tick_font_size,
            fontweight="normal",
            color="#222222",
            clip_on=False,
        )
    for group_name, members in GROUPS:
        xs = [_bar_x(bar_id) for bar_id in members]
        ax.text(
            0.5 * (xs[0] + xs[-1]),
            -14.8,
            group_name,
            ha="center",
            va="top",
            fontsize=style.group_font_size,
            fontweight="bold",
            color="#222222",
            clip_on=False,
        )


def create_figure(
    palette: Palette = PALETTES[0],
    data: CartoonBarData = DEFAULT_DATA,
    style: ChartStyle = DEFAULT_STYLE,
) -> Figure:
    """Create the cartoon stacked-bar figure without writing it to disk."""

    data.validate()
    style.validate()

    with plt.rc_context(
        {
            "font.family": [
                "Comic Sans MS",
                "Chalkboard SE",
                "Arial Rounded MT Bold",
                "DejaVu Sans",
                "sans-serif",
            ],
            "font.size": style.tick_font_size,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    ):
        figure = plt.figure(figsize=style.figure_size, facecolor="white")
        ax = figure.add_axes(style.axes_bounds)
        _style_axis(ax, style)
        figure.canvas.draw()
        _draw_frame(ax, style)
        _draw_bars(ax, data, palette, style)
        _draw_legend(ax, palette, style)
        _draw_title(ax, style)
        _draw_x_labels(ax, style)

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
    dpi: int = 150,
    data: CartoonBarData = DEFAULT_DATA,
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
    stem = f"cartoon_stacked_bar_palette_{index:02d}_{palette.name.replace('-', '_')}"
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
        description="Render the cartoon stacked-bar policy-mix chart."
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
        default=150,
        help="raster DPI; 150 reproduces the 1916×1080 reference (default: 150)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/cartoon_stacked_bar"),
        help="destination directory (default: output/cartoon_stacked_bar)",
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
