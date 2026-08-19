"""Reproduce the stacked N2O bar-plus-line chart from the Xiaohongshu reference.

The carousel repeats the same two-panel figure 18 times, changing only the
LF / HF colour pair.  The top panel is a horizontal bar chart of cumulative
N2O emission with replicate points, SEM bars, and a significance bracket.
The bottom panel is the matching flux time series, with error bars and a
three-period background split by two dashed guides.

The source post publishes the chart but not the underlying table, and notes
that the top-panel bars were a naive sum of the flux points rather than a
true cumulative flux.  ``DEFAULT_DATA`` stores the digitised series and
keeps that same shortcut so the rendered bars match the carousel.  Replace
the flux arrays and/or the cumulative mapping when using another dataset.
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
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator
from numpy.typing import NDArray


GROUPS: Final[tuple[str, ...]] = ("LF", "HF")

DATES: Final[tuple[str, ...]] = (
    "20 Oct.",
    "26 Oct.",
    "31 Oct.",
    "05 Nov.",
    "13 Nov.",
    "31 Dec.",
    "12 Mar.",
    "20 Mar.",
    "28 Mar.",
    "05 Apr.",
    "15 Apr.",
)

# Digitised from the visible markers in carousel frame 2 (orange / green).
TARGET_FLUX: Final[Mapping[str, tuple[float, ...]]] = {
    "LF": (16.0, 48.0, 26.0, 9.0, 6.0, 8.0, 10.0, 23.0, 85.0, 58.0, 14.0),
    "HF": (26.0, 45.0, 34.0, 25.0, 6.0, 12.0, 34.0, 39.0, 152.0, 79.0, 15.0),
}

TARGET_FLUX_ERR: Final[Mapping[str, tuple[float, ...]]] = {
    "LF": (3.0, 6.0, 4.0, 2.0, 2.0, 2.0, 3.0, 4.0, 8.0, 6.0, 3.0),
    "HF": (4.0, 6.0, 4.0, 4.0, 2.0, 3.0, 4.0, 5.0, 12.0, 8.0, 3.0),
}

# Four visible replicates clustered at the bar tips in the reference frames.
TARGET_REPLICATES: Final[Mapping[str, tuple[float, ...]]] = {
    "LF": (271.0, 292.0, 311.0, 338.0),
    "HF": (428.0, 451.0, 478.0, 511.0),
}

# First dashed guide sits on 13 Nov.; the second sits on 12 Mar.
PERIOD_BOUNDARIES: Final[tuple[float, float]] = (4.0, 6.0)


@dataclass(frozen=True, slots=True)
class Palette:
    """LF then HF colours."""

    name: str
    colors: tuple[str, str]

    def for_group(self, group: str) -> str:
        try:
            return self.colors[GROUPS.index(group)]
        except ValueError as exc:
            raise KeyError(f"unsupported group: {group}") from exc


# Colours sampled from the LF / HF bar interiors in carousel frames 2-18.
# Frame 1 is a 3x3 preview sheet; its unused rose-sage pair is palette 18.
PALETTES: Final[tuple[Palette, ...]] = (
    Palette("orange-green", ("#FA882F", "#338227")),
    Palette("coral-steel", ("#DA5D47", "#538CB9")),
    Palette("orange-seafoam", ("#FA882F", "#73B7A0")),
    Palette("rose-teal", ("#CD89A0", "#539780")),
    Palette("salmon-lime", ("#FF9D87", "#93E287")),
    Palette("gold-teal", ("#EDBF4F", "#539780")),
    Palette("peach-seafoam", ("#EDB487", "#73B7A0")),
    Palette("pink-leaf", ("#EDA9C0", "#53A247")),
    Palette("orange-mauve", ("#FA882F", "#AD6980")),
    Palette("burgundy-wine", ("#4D0920", "#9A1240")),
    Palette("navy-amber", ("#336C99", "#FA9300")),
    Palette("gray-silver", ("#808080", "#C0C0C0")),
    Palette("navy-crimson", ("#336C99", "#E72627")),
    Palette("charcoal-coral", ("#404040", "#DA5D47")),
    Palette("navy-sky", ("#336C99", "#93CCF9")),
    Palette("crimson-amber", ("#E72627", "#FA9300")),
    Palette("azure-peach", ("#0678D1", "#EDB487")),
    Palette("rose-sage", ("#E8A0B4", "#A8C86C")),
)


@dataclass(frozen=True, slots=True)
class BarLineData:
    """Flux series and cumulative bars aligned with :data:`GROUPS`."""

    groups: tuple[str, ...]
    dates: tuple[str, ...]
    flux: Mapping[str, NDArray[np.float64]]
    flux_err: Mapping[str, NDArray[np.float64]]
    replicates: Mapping[str, NDArray[np.float64]]
    cumulative: Mapping[str, float] | None = None
    cumulative_err: Mapping[str, float] | None = None

    def validate(self) -> None:
        if self.groups != GROUPS:
            raise ValueError("groups must match the canonical LF / HF order")
        if self.dates != DATES:
            raise ValueError("dates must match the canonical eleven-date order")
        expected = set(self.groups)
        for mapping_name, mapping in (
            ("flux", self.flux),
            ("flux_err", self.flux_err),
            ("replicates", self.replicates),
        ):
            if set(mapping) != expected:
                raise ValueError(f"{mapping_name} must contain one array per group")
        if self.cumulative is not None and set(self.cumulative) != expected:
            raise ValueError("cumulative must contain one value per group")
        if self.cumulative_err is not None and set(self.cumulative_err) != expected:
            raise ValueError("cumulative_err must contain one value per group")
        n_dates = len(self.dates)
        for group in self.groups:
            flux = np.asarray(self.flux[group], dtype=float)
            err = np.asarray(self.flux_err[group], dtype=float)
            reps = np.asarray(self.replicates[group], dtype=float)
            if flux.shape != (n_dates,) or err.shape != (n_dates,):
                raise ValueError(f"{group} flux arrays must have length {n_dates}")
            if reps.ndim != 1 or reps.size < 2:
                raise ValueError(f"{group} replicates must be 1-D and of length >= 2")
            if not np.all(np.isfinite(flux)) or not np.all(np.isfinite(err)):
                raise ValueError(f"{group} flux values must be finite")
            if not np.all(np.isfinite(reps)):
                raise ValueError(f"{group} replicates must be finite")
            if np.any(err < 0):
                raise ValueError(f"{group} flux errors must be non-negative")
            if self.cumulative_err is not None and self.cumulative_err[group] < 0:
                raise ValueError(f"{group} cumulative error must be non-negative")

    def flux_series(self, group: str) -> NDArray[np.float64]:
        return np.asarray(self.flux[group], dtype=float)

    def flux_errors(self, group: str) -> NDArray[np.float64]:
        return np.asarray(self.flux_err[group], dtype=float)

    def replicate_values(self, group: str) -> NDArray[np.float64]:
        return np.asarray(self.replicates[group], dtype=float)

    def bar_stats(self, group: str) -> tuple[float, float]:
        if self.cumulative is not None and self.cumulative_err is not None:
            return float(self.cumulative[group]), float(self.cumulative_err[group])
        sample = self.replicate_values(group)
        return float(sample.mean()), float(sample.std(ddof=1) / np.sqrt(sample.size))


@dataclass(frozen=True, slots=True)
class ChartStyle:
    """Layout tuned to the 1260 x 1080 reference image."""

    figure_size: tuple[float, float] = (8.4, 7.2)
    bar_bounds: tuple[float, float, float, float] = (0.168, 0.705, 0.762, 0.198)
    line_bounds: tuple[float, float, float, float] = (0.168, 0.168, 0.762, 0.428)
    bar_x_limits: tuple[float, float] = (0.0, 640.0)
    line_y_limits: tuple[float, float] = (0.0, 165.0)
    bar_height: float = 0.46
    line_width: float = 1.85
    marker_size: float = 7.2
    spine_width: float = 1.45
    tick_length: float = 4.8
    label_font_size: float = 13.0
    tick_font_size: float = 10.5
    panel_font_size: float = 16.0
    significance_font_size: float = 13.0
    panel_color: str = "#1F4E8C"
    x_label_pad: float = 8.0
    y_label_pad: float = 6.0

    def validate(self) -> None:
        if min(self.figure_size) <= 0:
            raise ValueError("figure_size must be positive")
        if self.bar_x_limits[1] <= self.bar_x_limits[0]:
            raise ValueError("bar_x_limits must be increasing")
        if self.line_y_limits[1] <= self.line_y_limits[0]:
            raise ValueError("line_y_limits must be increasing")
        if not 0.15 < self.bar_height < 0.8:
            raise ValueError("bar_height must sit in (0.15, 0.8)")


DEFAULT_STYLE: Final[ChartStyle] = ChartStyle()


def naive_cumulative(flux: Sequence[float]) -> float:
    """Sum flux points the way the source post built the top bars."""

    values = np.asarray(flux, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("flux must be a non-empty 1-D sequence")
    if not np.all(np.isfinite(values)):
        raise ValueError("flux values must be finite")
    return float(values.sum())


def replicate_sem(values: Sequence[float]) -> float:
    sample = np.asarray(values, dtype=float)
    if sample.size < 2:
        raise ValueError("need at least two replicates to compute SEM")
    return float(sample.std(ddof=1) / np.sqrt(sample.size))


def _build_default_data() -> BarLineData:
    flux = {group: np.asarray(TARGET_FLUX[group], dtype=float) for group in GROUPS}
    return BarLineData(
        groups=GROUPS,
        dates=DATES,
        flux=flux,
        flux_err={group: np.asarray(TARGET_FLUX_ERR[group], dtype=float) for group in GROUPS},
        replicates={
            group: np.asarray(TARGET_REPLICATES[group], dtype=float) for group in GROUPS
        },
        cumulative={group: naive_cumulative(flux[group]) for group in GROUPS},
        cumulative_err={
            group: replicate_sem(TARGET_REPLICATES[group]) for group in GROUPS
        },
    )


DEFAULT_DATA: Final[BarLineData] = _build_default_data()


def _period_cmap(color: str) -> LinearSegmentedColormap:
    r, g, b = to_rgb(color)
    return LinearSegmentedColormap.from_list(
        "period",
        (
            (0.0, (r, g, b, 0.03)),
            (0.45, (r, g, b, 0.14)),
            (1.0, (r, g, b, 0.04)),
        ),
        N=256,
    )


def _style_bar_axis(ax: Axes, style: ChartStyle) -> None:
    ax.set_xlim(*style.bar_x_limits)
    ax.set_ylim(-0.72, 1.72)
    ax.set_yticks((1.0, 0.0), labels=list(GROUPS))
    ax.set_xticks((0, 150, 300, 450, 600))
    ax.tick_params(
        axis="x",
        which="major",
        direction="in",
        top=False,
        bottom=True,
        length=style.tick_length,
        width=style.spine_width,
        labelsize=style.tick_font_size,
        pad=3.5,
    )
    ax.tick_params(
        axis="y",
        which="major",
        direction="in",
        left=True,
        right=False,
        length=style.tick_length,
        width=style.spine_width,
        labelsize=style.tick_font_size,
        pad=4.0,
    )
    ax.set_xlabel(
        r"$N_2O$ cumulative emission ($kg \cdot N_2O \cdot ha^{-1}$)",
        fontsize=style.label_font_size,
        fontweight="bold",
        labelpad=style.x_label_pad,
    )
    for label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        label.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(style.spine_width)
        spine.set_color("black")
    ax.set_axisbelow(False)
    ax.grid(False)


def _style_line_axis(ax: Axes, style: ChartStyle) -> None:
    ax.set_xlim(-0.55, 10.55)
    ax.set_ylim(*style.line_y_limits)
    ax.yaxis.set_major_locator(MultipleLocator(50))
    ax.set_xticks(np.arange(len(DATES), dtype=float))
    ax.set_xticklabels(DATES, rotation=90, ha="center", va="top")
    ax.tick_params(
        which="major",
        direction="in",
        top=True,
        right=True,
        length=style.tick_length,
        width=style.spine_width,
        labelsize=style.tick_font_size,
        pad=3.5,
    )
    ax.set_ylabel(
        r"$N_2O$ emission flux ($\mu g \cdot m^{-2} \cdot h^{-1}$)",
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
    data: BarLineData,
    palette: Palette,
    style: ChartStyle,
) -> None:
    rng = np.random.default_rng(20260819)
    for group, y in zip(GROUPS, (1.0, 0.0), strict=True):
        color = palette.for_group(group)
        mean, sem = data.bar_stats(group)
        ax.barh(
            y,
            mean,
            height=style.bar_height,
            color=color,
            edgecolor="black",
            linewidth=1.15,
            zorder=2,
        )
        ax.errorbar(
            mean,
            y,
            xerr=sem,
            fmt="none",
            ecolor="black",
            elinewidth=1.25,
            capsize=3.8,
            capthick=1.25,
            zorder=4,
        )
        replicates = data.replicate_values(group)
        jitter = rng.uniform(-0.13, 0.13, size=replicates.size)
        ax.scatter(
            replicates,
            y + jitter,
            s=36,
            facecolors=color,
            edgecolors="black",
            linewidths=0.85,
            zorder=5,
        )

    left, right = data.bar_stats("LF")[0], data.bar_stats("HF")[0]
    sig_x = max(left, right) + max(
        data.bar_stats("LF")[1], data.bar_stats("HF")[1]
    ) + 28.0
    ax.plot(
        [sig_x, sig_x],
        [0.0, 1.0],
        color="black",
        linewidth=1.35,
        solid_capstyle="butt",
        zorder=6,
    )
    ax.text(
        sig_x + 10.0,
        0.5,
        "**",
        ha="left",
        va="center",
        fontsize=style.significance_font_size,
        fontweight="bold",
        zorder=6,
    )


def _draw_periods(ax: Axes, palette: Palette, style: ChartStyle) -> None:
    y0, y1 = style.line_y_limits
    left_end, right_start = PERIOD_BOUNDARIES
    bands = (
        (-0.55, left_end, palette.for_group("LF")),
        (right_start, 10.55, palette.for_group("HF")),
    )
    for x0, x1, color in bands:
        gradient = np.linspace(0.0, 1.0, 256).reshape(1, -1)
        image = ax.imshow(
            gradient,
            extent=(x0, x1, y0, y1),
            origin="lower",
            aspect="auto",
            cmap=_period_cmap(color),
            interpolation="bicubic",
            zorder=0,
        )
        image.set_clip_path(ax.patch)
    for x in PERIOD_BOUNDARIES:
        ax.axvline(
            x,
            linestyle=(0, (4.2, 3.2)),
            color="#8D8D8D",
            linewidth=1.15,
            zorder=1,
        )


def _draw_lines(
    ax: Axes,
    data: BarLineData,
    palette: Palette,
    style: ChartStyle,
) -> None:
    x = np.arange(len(data.dates), dtype=float)
    markers = {"LF": "s", "HF": "o"}
    for group in GROUPS:
        color = palette.for_group(group)
        ax.errorbar(
            x,
            data.flux_series(group),
            yerr=data.flux_errors(group),
            fmt=f"-{markers[group]}",
            color=color,
            ecolor=color,
            linewidth=style.line_width,
            markersize=style.marker_size,
            markerfacecolor=color,
            markeredgecolor=color,
            markeredgewidth=0.6,
            capsize=3.2,
            capthick=1.15,
            elinewidth=1.15,
            zorder=3,
        )


def create_figure(
    palette: Palette = PALETTES[0],
    data: BarLineData = DEFAULT_DATA,
    style: ChartStyle = DEFAULT_STYLE,
) -> Figure:
    """Create the stacked bar-plus-line chart without writing it to disk."""

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
        ax_bar = figure.add_axes(style.bar_bounds)
        ax_line = figure.add_axes(style.line_bounds)
        ax_bar.set_facecolor("white")
        ax_line.set_facecolor("#FBFCF8")
        _style_bar_axis(ax_bar, style)
        _style_line_axis(ax_line, style)
        _draw_bars(ax_bar, data, palette, style)
        _draw_periods(ax_line, palette, style)
        _draw_lines(ax_line, data, palette, style)
        figure.text(
            0.042,
            0.955,
            "(a)",
            ha="left",
            va="top",
            fontsize=style.panel_font_size,
            fontweight="bold",
            color=style.panel_color,
            zorder=7,
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
    dpi: int = 150,
    data: BarLineData = DEFAULT_DATA,
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
    stem = f"bar_line_palette_{index:02d}_{palette.name.replace('-', '_')}"
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
        description="Render the stacked N2O bar-plus-line chart."
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
        help="raster DPI; 150 reproduces the 1260x1080 reference (default: 150)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/bar_line"),
        help="destination directory (default: output/bar_line)",
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
            print(f"{index:2d}  {palette.name:18s}  {' '.join(palette.colors)}")
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
