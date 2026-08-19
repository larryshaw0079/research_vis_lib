"""Reproduce the patient-level concordance UpSet variant.

The Xiaohongshu carousel repeats the same eight-combination chart 18 times,
changing only the three group colours.  Bars sit above a presence/absence
matrix of check marks and crosses for plasma-specific, tissue-specific, and
shared variants.

Counts and grouping follow Figure 4b of Zhang et al. (2026), npj Precision
Oncology, CC BY 4.0.  The carousel prints those patient totals on the bars;
replace :data:`DEFAULT_DATA` when using the renderer with another table.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import to_hex, to_rgb
from matplotlib.figure import Figure
from matplotlib.ticker import FixedLocator, MultipleLocator
from numpy.typing import NDArray


GROUPS: Final[tuple[str, ...]] = (
    "Disconcordant",
    "Complete concordant",
    "Partially concordant",
)

VARIANT_ROWS: Final[tuple[str, ...]] = (
    "With plasma-specific variants",
    "With tissue-specific variants",
    "With shared variants",
)

ROW_LABEL_COLORS: Final[tuple[str, str, str]] = (
    "#E17D65",
    "#276E8F",
    "#515151",
)

CHECK_MARK: Final[str] = "✓"
CROSS_MARK: Final[str] = "×"
CROSS_COLOR: Final[str] = "#85A3B8"
CHECK_COLOR: Final[str] = "#111111"


@dataclass(frozen=True, slots=True)
class Combination:
    """One UpSet column: variant membership, patient count, and group."""

    plasma: bool
    tissue: bool
    shared: bool
    count: int
    group: str

    @property
    def membership(self) -> tuple[bool, bool, bool]:
        return (self.plasma, self.tissue, self.shared)


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


# Named qualitative sets use canonical ColorBrewer / Tableau / Paul Tol hex
# values.  The remaining trios were sampled from bar interiors in frames 1-4.
PALETTES: Final[tuple[Palette, ...]] = (
    Palette("teal-coral-sand", ("#1F7598", "#DE6344", "#FBE2C4")),
    Palette("navy-crimson-ivory", ("#2B4B7C", "#DA534F", "#FFFFFF")),
    Palette("forest-crimson-gold", ("#4A7C59", "#C83E4D", "#F4D35F")),
    Palette("navy-magenta-amber", ("#003F5C", "#BC5090", "#FFA600")),
    Palette("tab10", ("#1F77B4", "#FF7F0E", "#2CA02C")),
    Palette("tab10-earth", ("#8C564B", "#E377C2", "#7F7F7F")),
    Palette("set1", ("#E41A1C", "#377EB8", "#4DAF4A")),
    Palette("set2", ("#66C2A5", "#FC8D62", "#8DA0CB")),
    Palette("dark2", ("#1B9E77", "#D95F02", "#7570B3")),
    Palette("paired", ("#A6CEE3", "#1F78B4", "#B2DF8A")),
    Palette("pastel1", ("#FBB4AE", "#B3CDE3", "#CCEBC5")),
    Palette("pastel2", ("#B3E2CD", "#FDCDAC", "#CBD5E8")),
    Palette("set3", ("#8DD3C7", "#FFFFB3", "#BEBADA")),
    Palette("set3-warm", ("#FB8072", "#80B1D3", "#FDB462")),
    Palette("set1-orchid-amber", ("#984EA3", "#FF7F00", "#FFFF33")),
    Palette("set1-umber", ("#A65628", "#F781BF", "#999999")),
    Palette("tol-muted", ("#332288", "#88CCEE", "#44AA99")),
    Palette("set2-bloom", ("#E78AC3", "#A6D854", "#FFD92F")),
)


@dataclass(frozen=True, slots=True)
class ConcordanceUpsetData:
    """Eight combination columns aligned with the reference matrix."""

    combinations: tuple[Combination, ...]

    def validate(self) -> None:
        if len(self.combinations) != 8:
            raise ValueError("combinations must contain the eight reference columns")
        if any(item.count < 0 for item in self.combinations):
            raise ValueError("patient counts must be non-negative")
        groups = tuple(item.group for item in self.combinations)
        expected = (
            GROUPS[0],
            GROUPS[0],
            GROUPS[0],
            GROUPS[1],
            GROUPS[1],
            GROUPS[2],
            GROUPS[2],
            GROUPS[2],
        )
        if groups != expected:
            raise ValueError("combination groups must follow Disconcordant / complete / partial order")
        if len({item.membership for item in self.combinations}) != 8:
            raise ValueError("each combination column must be unique")
        unknown = set(groups) - set(GROUPS)
        if unknown:
            raise ValueError(f"unsupported groups: {sorted(unknown)}")

    def counts(self) -> tuple[int, ...]:
        return tuple(item.count for item in self.combinations)

    def membership_matrix(self) -> NDArray[np.bool_]:
        return np.asarray(
            [item.membership for item in self.combinations],
            dtype=bool,
        ).T

    def total_patients(self) -> int:
        return int(sum(item.count for item in self.combinations))


DEFAULT_DATA: Final[ConcordanceUpsetData] = ConcordanceUpsetData(
    combinations=(
        Combination(False, True, False, 420, GROUPS[0]),
        Combination(True, False, False, 7, GROUPS[0]),
        Combination(True, True, False, 76, GROUPS[0]),
        Combination(False, False, False, 8, GROUPS[1]),
        Combination(False, False, True, 42, GROUPS[1]),
        Combination(True, False, True, 99, GROUPS[2]),
        Combination(False, True, True, 209, GROUPS[2]),
        Combination(True, True, True, 250, GROUPS[2]),
    )
)


@dataclass(frozen=True, slots=True)
class ChartStyle:
    """Layout tuned to the 2431 × 1603 reference image."""

    figure_size: tuple[float, float] = (9.724, 6.412)
    subplot_left: float = 0.358
    subplot_right: float = 0.986
    subplot_top: float = 0.855
    subplot_bottom: float = 0.048
    height_ratios: tuple[float, float] = (3.28, 1.0)
    hspace: float = 0.035
    y_limits: tuple[float, float] = (0.0, 478.0)
    y_ticks: tuple[int, ...] = (0, 150, 300, 450)
    y_minor_tick: float = 50.0
    bar_width: float = 0.62
    value_pad: float = 7.0
    group_label_pad: float = 22.0
    spine_width: float = 1.45
    tick_length: float = 5.0
    bar_edge_width: float = 0.85
    title_font_size: float = 18.0
    panel_font_size: float = 20.0
    label_font_size: float = 15.0
    tick_font_size: float = 12.0
    value_font_size: float = 12.5
    group_font_size: float = 13.5
    row_font_size: float = 12.5
    mark_font_size: float = 17.0
    luminance_cutoff: float = 0.93

    def validate(self) -> None:
        if min(self.figure_size) <= 0:
            raise ValueError("figure_size must be positive")
        if self.y_limits[1] <= self.y_limits[0]:
            raise ValueError("y_limits must be increasing")
        if not 0 < self.bar_width < 1.0:
            raise ValueError("bar_width must sit in (0, 1)")
        if self.subplot_right <= self.subplot_left:
            raise ValueError("subplot x-bounds must be increasing")
        if self.subplot_top <= self.subplot_bottom:
            raise ValueError("subplot y-bounds must be increasing")


DEFAULT_STYLE: Final[ChartStyle] = ChartStyle()


def _relative_luminance(color: str) -> float:
    red, green, blue = to_rgb(color)
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def group_label_color(color: str, cutoff: float = DEFAULT_STYLE.luminance_cutoff) -> str:
    """Darken a near-white fill so the group title stays readable."""

    if _relative_luminance(color) <= cutoff:
        return color
    red, green, blue = to_rgb(color)
    return to_hex(tuple(channel * 0.55 for channel in (red, green, blue)))


def _style_bar_axis(ax: Axes, style: ChartStyle) -> None:
    ax.set_ylim(*style.y_limits)
    ax.yaxis.set_major_locator(FixedLocator(style.y_ticks))
    ax.yaxis.set_minor_locator(MultipleLocator(style.y_minor_tick))
    ax.tick_params(
        axis="y",
        which="major",
        direction="out",
        left=True,
        right=False,
        length=style.tick_length,
        width=style.spine_width,
        labelsize=style.tick_font_size,
        pad=5.0,
    )
    ax.tick_params(
        axis="y",
        which="minor",
        direction="out",
        left=True,
        right=False,
        length=style.tick_length * 0.55,
        width=style.spine_width,
    )
    ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    ax.set_ylabel(
        "Number of patients",
        fontsize=style.label_font_size,
        labelpad=10.0,
    )
    ax.set_title(
        "Patient-Level Concordance",
        fontsize=style.title_font_size,
        pad=10.0,
    )
    for spine_name, spine in ax.spines.items():
        spine.set_visible(spine_name in {"left", "bottom"})
        spine.set_linewidth(style.spine_width)
        spine.set_color("black")
    ax.set_facecolor("white")
    ax.grid(False)


def _draw_bars(
    ax: Axes,
    data: ConcordanceUpsetData,
    palette: Palette,
    style: ChartStyle,
) -> NDArray[np.float64]:
    centers = np.arange(len(data.combinations), dtype=float)
    colors = [palette.for_group(item.group) for item in data.combinations]
    ax.bar(
        centers,
        data.counts(),
        width=style.bar_width,
        color=colors,
        edgecolor="black",
        linewidth=style.bar_edge_width,
        zorder=2,
    )
    for x, item in zip(centers, data.combinations, strict=True):
        ax.text(
            x,
            item.count + style.value_pad,
            f"{item.count}",
            ha="center",
            va="bottom",
            fontsize=style.value_font_size,
            color="black",
            zorder=3,
        )
    return centers


def _group_label_text(group: str) -> str:
    if " " in group:
        first, rest = group.split(" ", 1)
        return f"{first}\n{rest}"
    return group


def _draw_group_labels(
    ax: Axes,
    data: ConcordanceUpsetData,
    palette: Palette,
    style: ChartStyle,
) -> None:
    centers = np.arange(len(data.combinations), dtype=float)
    for group in GROUPS:
        paired = [
            (x, item)
            for x, item in zip(centers, data.combinations, strict=True)
            if item.group == group
        ]
        if not paired:
            continue
        xs = [x for x, _ in paired]
        peak = max(item.count for _, item in paired)
        color = group_label_color(palette.for_group(group), style.luminance_cutoff)
        ax.text(
            float(np.mean(xs)),
            peak + style.value_pad + style.group_label_pad,
            _group_label_text(group),
            ha="center",
            va="bottom",
            fontsize=style.group_font_size,
            fontweight="bold",
            color=color,
            linespacing=1.05,
            zorder=4,
            clip_on=False,
        )


def _draw_matrix(
    ax: Axes,
    data: ConcordanceUpsetData,
    style: ChartStyle,
) -> None:
    matrix = data.membership_matrix()
    n_rows, n_cols = matrix.shape
    ax.set_ylim(-0.55, n_rows - 0.45)
    ax.set_xlim(-0.65, n_cols - 0.35)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("white")
    ax.set_axis_off()

    for row, present_row in enumerate(matrix):
        y = n_rows - 1 - row
        for x, present in enumerate(present_row):
            ax.text(
                x,
                y,
                CHECK_MARK if present else CROSS_MARK,
                ha="center",
                va="center",
                fontsize=style.mark_font_size,
                color=CHECK_COLOR if present else CROSS_COLOR,
                fontfamily="DejaVu Sans",
                fontweight="bold",
                zorder=2,
            )
        ax.text(
            -0.08,
            y,
            VARIANT_ROWS[row],
            ha="right",
            va="center",
            fontsize=style.row_font_size,
            fontweight="bold",
            color=ROW_LABEL_COLORS[row],
            transform=ax.get_yaxis_transform(),
            clip_on=False,
            zorder=3,
        )


def create_figure(
    palette: Palette = PALETTES[0],
    data: ConcordanceUpsetData = DEFAULT_DATA,
    style: ChartStyle = DEFAULT_STYLE,
) -> Figure:
    """Create the concordance UpSet chart without writing it to disk."""

    data.validate()
    style.validate()

    with plt.rc_context(
        {
            "font.family": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
            "font.size": style.tick_font_size,
            "axes.linewidth": style.spine_width,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    ):
        figure = plt.figure(figsize=style.figure_size, facecolor="white")
        grid = figure.add_gridspec(
            2,
            1,
            height_ratios=list(style.height_ratios),
            left=style.subplot_left,
            right=style.subplot_right,
            top=style.subplot_top,
            bottom=style.subplot_bottom,
            hspace=style.hspace,
        )
        ax_bar = figure.add_subplot(grid[0])
        ax_mat = figure.add_subplot(grid[1], sharex=ax_bar)

        _style_bar_axis(ax_bar, style)
        centers = _draw_bars(ax_bar, data, palette, style)
        _draw_group_labels(ax_bar, data, palette, style)
        ax_bar.set_xlim(-0.65, len(centers) - 0.35)
        _draw_matrix(ax_mat, data, style)

        figure.text(
            0.028,
            0.955,
            "b",
            fontsize=style.panel_font_size,
            fontweight="bold",
            ha="left",
            va="top",
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
    data: ConcordanceUpsetData = DEFAULT_DATA,
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
    stem = f"concordance_upset_palette_{index:02d}_{palette.name.replace('-', '_')}"
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
        description="Render the patient-level concordance UpSet variant."
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
        help="raster DPI; 250 reproduces the 2431×1603 reference (default: 250)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/concordance_upset"),
        help="destination directory (default: output/concordance_upset)",
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
