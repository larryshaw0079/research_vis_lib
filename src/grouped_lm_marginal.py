"""Reproduce the grouped linear-fit scatter plot with marginal histograms.

The Xiaohongshu carousel shows the same GST–LST land-cover comparison 16
times, changing only the four-colour palette.  Each panel has:

* a joint scatter with per-group dashed OLS fits and 95% confidence bands;
* stacked histograms plus dashed KDE curves on the top and right margins;
* colour-matched regression equations, R², and p-values in the joint panel.

The source post does not publish the point table.  ``DEFAULT_DATA`` is a
synthetic 25-point sample per group whose OLS slope, intercept, R², and
p-value round to the printed annotations.
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
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
from numpy.typing import NDArray
from scipy.stats import gaussian_kde, t


GROUPS: Final[tuple[str, ...]] = ("Grass", "Land", "Water", "Urban")

TARGET_FITS: Final[Mapping[str, tuple[float, float, float]]] = {
    "Grass": (1.69, 2.09, 0.845),
    "Land": (1.24, 4.17, 0.717),
    "Water": (0.91, 3.16, 0.579),
    "Urban": (0.44, 8.33, 0.250),
}


@dataclass(frozen=True, slots=True)
class Palette:
    """Four group colours in :data:`GROUPS` order."""

    name: str
    colors: tuple[str, str, str, str]

    def for_group(self, group: str) -> str:
        try:
            return self.colors[GROUPS.index(group)]
        except ValueError as exc:
            raise KeyError(f"unsupported group: {group}") from exc


# Colours reconstructed from the 16 carousel frames (histogram bars and
# scatter markers).  JPEG compression in the source images was snapped to
# nearby saturated hex values while keeping Grass → Land → Water → Urban order.
PALETTES: Final[tuple[Palette, ...]] = (
    Palette("navy-gold-crimson-gray", ("#3D4A61", "#F4A63A", "#D32F4A", "#A9B1BC")),
    Palette("charcoal-coral-amber-lilac", ("#2F2E3E", "#E14B32", "#F09A32", "#C5B3C4")),
    Palette("linen-terracotta-slate-sage", ("#F3E6D6", "#E0927C", "#5E6074", "#8FBEAB")),
    Palette("espresso-brick-gold-azure", ("#3A1C16", "#8A3A32", "#F0B44A", "#4E7ED6")),
    Palette("sage-tan-coral-mint", ("#D2DCC0", "#D7A47C", "#E39B80", "#8FCDC6")),
    Palette("midnight-lilac-pink-cream", ("#1F1433", "#C9A3E6", "#F09BB8", "#F6E9B0")),
    Palette("rose-butter-lime-sky", ("#F3767A", "#F5D56A", "#96C84E", "#4A9BC8")),
    Palette("magenta-violet-periwinkle-gold", ("#F02E86", "#6E45D6", "#7B88E8", "#F5C43A")),
    Palette("plum-wine-teal-ember", ("#4E1C38", "#9A2A40", "#3A6672", "#E07838")),
    Palette("aqua-steel-peach-sand", ("#5BB8B6", "#7A8494", "#E8A070", "#E0B84A")),
    Palette("charcoal-sand-sienna-mint", ("#2F3638", "#D28A3C", "#B65422", "#A8D4C4")),
    Palette("olive-khaki-clay-rose", ("#B4BE9C", "#D4B08C", "#E0B070", "#C45C5C")),
    Palette("teal-lemon-peach-sand", ("#3E8F86", "#E8E070", "#F3A882", "#F5D6A6")),
    Palette("mauve-rose-blush-steel", ("#A56B78", "#C47A86", "#E89098", "#5A6A82")),
    Palette("salmon-taupe-slate-sage", ("#E0907C", "#C4A090", "#5C6074", "#8FB8A8")),
    Palette("navy-cyan-rose-gold", ("#1A1D5C", "#2BB0C8", "#E04A68", "#F0B030")),
)


@dataclass(frozen=True, slots=True)
class GroupedLMData:
    """GST/LST samples aligned with :data:`GROUPS`."""

    groups: tuple[str, ...]
    gst: Mapping[str, NDArray[np.float64]]
    lst: Mapping[str, NDArray[np.float64]]

    def validate(self) -> None:
        if self.groups != GROUPS:
            raise ValueError("groups must match the canonical four-class order")
        if set(self.gst) != set(self.groups) or set(self.lst) != set(self.groups):
            raise ValueError("gst and lst must contain one array per group")
        for group in self.groups:
            x = np.asarray(self.gst[group], dtype=float)
            y = np.asarray(self.lst[group], dtype=float)
            if x.shape != y.shape or x.ndim != 1 or x.size < 3:
                raise ValueError(f"{group} arrays must be 1-D and of equal length >= 3")
            if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
                raise ValueError(f"{group} values must be finite")


@dataclass(frozen=True, slots=True)
class ChartStyle:
    """Layout tuned to the 2132 × 1962 reference image."""

    figure_size: tuple[float, float] = (8.528, 7.848)
    joint_bounds: tuple[float, float, float, float] = (0.1124, 0.1037, 0.7215, 0.7306)
    marg_x_bounds: tuple[float, float, float, float] = (0.1124, 0.8343, 0.7215, 0.1472)
    marg_y_bounds: tuple[float, float, float, float] = (0.8339, 0.1037, 0.1465, 0.7306)
    x_limits: tuple[float, float] = (2.0, 28.0)
    y_limits: tuple[float, float] = (7.0, 29.0)
    hist_bins: int = 22
    scatter_size: float = 38.0
    scatter_alpha: float = 0.88
    ci_alpha: float = 0.18
    hist_alpha: float = 0.92
    line_width: float = 1.55
    kde_width: float = 1.35
    spine_width: float = 1.45
    tick_length: float = 4.4
    label_font_size: float = 15.0
    tick_font_size: float = 11.0
    stats_font_size: float = 10.0
    stats_line_gap: float = 0.055

    def validate(self) -> None:
        if min(self.figure_size) <= 0:
            raise ValueError("figure_size must be positive")
        if self.x_limits[1] <= self.x_limits[0] or self.y_limits[1] <= self.y_limits[0]:
            raise ValueError("axis limits must be increasing")
        if self.hist_bins < 4:
            raise ValueError("hist_bins must be at least 4")


DEFAULT_STYLE: Final[ChartStyle] = ChartStyle()


def _sample_x(
    n: int,
    mean: float,
    std: float,
    low: float,
    high: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    values = np.empty(n, dtype=float)
    for index in range(n):
        value = rng.normal(mean, std)
        for _ in range(40):
            if low <= value <= high:
                break
            value = rng.normal(mean, std)
        values[index] = float(np.clip(value, low, high))
    return values


def _response_with_target_fit(
    x: NDArray[np.float64],
    slope: float,
    intercept: float,
    r_squared: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Build y so OLS recovers ``slope``, ``intercept``, and ``r_squared``."""

    centered = x - x.mean()
    residual = rng.normal(size=x.size)
    residual -= residual.mean()
    residual -= np.dot(residual, centered) / np.dot(centered, centered) * centered
    ss_regression = slope**2 * np.dot(centered, centered)
    ss_residual = (1.0 - r_squared) / r_squared * ss_regression
    residual *= np.sqrt(ss_residual / np.dot(residual, residual))
    return intercept + slope * x + residual


def _build_default_data(seed: int = 20260818) -> GroupedLMData:
    rng = np.random.default_rng(seed)
    x_specs = {
        "Grass": (8.2, 2.05, 4.2, 13.0),
        "Land": (12.1, 2.35, 6.8, 17.5),
        "Water": (16.3, 2.45, 10.5, 22.0),
        "Urban": (20.4, 2.15, 15.5, 25.5),
    }
    gst: dict[str, NDArray[np.float64]] = {}
    lst: dict[str, NDArray[np.float64]] = {}
    for group in GROUPS:
        mean, std, low, high = x_specs[group]
        slope, intercept, r_squared = TARGET_FITS[group]
        x = _sample_x(25, mean, std, low, high, rng)
        gst[group] = x
        lst[group] = _response_with_target_fit(x, slope, intercept, r_squared, rng)
    return GroupedLMData(groups=GROUPS, gst=gst, lst=lst)


DEFAULT_DATA: Final[GroupedLMData] = _build_default_data()


@dataclass(frozen=True, slots=True)
class OLSFit:
    slope: float
    intercept: float
    r_squared: float
    p_value: float
    n: int
    x_mean: float
    sxx: float
    mse: float

    def mean_band(self, x_grid: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        t_crit = float(t.ppf(0.975, self.n - 2))
        fitted = self.intercept + self.slope * x_grid
        se = np.sqrt(self.mse * (1.0 / self.n + (x_grid - self.x_mean) ** 2 / self.sxx))
        return fitted, fitted - t_crit * se, fitted + t_crit * se


def fit_ols(x: NDArray[np.float64], y: NDArray[np.float64]) -> OLSFit:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = int(x.size)
    x_mean = float(x.mean())
    y_mean = float(y.mean())
    centered = x - x_mean
    sxx = float(np.dot(centered, centered))
    slope = float(np.dot(centered, y - y_mean) / sxx)
    intercept = y_mean - slope * x_mean
    residual = y - (intercept + slope * x)
    sse = float(np.dot(residual, residual))
    sst = float(np.dot(y - y_mean, y - y_mean))
    mse = sse / (n - 2)
    t_stat = slope / np.sqrt(mse / sxx)
    p_value = float(2.0 * t.sf(abs(t_stat), n - 2))
    return OLSFit(
        slope=slope,
        intercept=intercept,
        r_squared=1.0 - sse / sst,
        p_value=p_value,
        n=n,
        x_mean=x_mean,
        sxx=sxx,
        mse=mse,
    )


def format_fit_line(group: str, fit: OLSFit) -> str:
    sign = "+" if fit.intercept >= 0 else "−"
    if fit.p_value < 0.001:
        p_text = "p < 0.001"
    else:
        p_text = f"p = {fit.p_value:.3f}"
    return (
        f"{group}: y = {fit.slope:.2f}x {sign} {abs(fit.intercept):.2f}, "
        f"$R^2$ = {fit.r_squared:.3f}, {p_text}"
    )


def _darken(color: str, factor: float = 0.62) -> tuple[float, float, float]:
    red, green, blue = to_rgb(color)
    return (red * factor, green * factor, blue * factor)


def _style_joint_axis(ax: Axes, style: ChartStyle) -> None:
    ax.set_xlim(*style.x_limits)
    ax.set_ylim(*style.y_limits)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(2.5))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
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
    ax.set_xlabel("GST", fontsize=style.label_font_size, fontweight="bold", labelpad=8)
    ax.set_ylabel("LST", fontsize=style.label_font_size, fontweight="bold", labelpad=6)
    for label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        label.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(style.spine_width)
        spine.set_color("black")
    ax.set_axisbelow(False)
    ax.grid(False)


def _style_marginal_axis(ax: Axes) -> None:
    ax.grid(False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False,
        labelright=False,
        labeltop=False,
        length=0,
    )
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_joint(
    ax: Axes,
    data: GroupedLMData,
    palette: Palette,
    style: ChartStyle,
    fits: Mapping[str, OLSFit],
) -> None:
    x_grid = np.linspace(*style.x_limits, 200)
    for group in GROUPS:
        color = palette.for_group(group)
        fit = fits[group]
        fitted, lower, upper = fit.mean_band(x_grid)
        ax.fill_between(x_grid, lower, upper, color=color, alpha=style.ci_alpha, linewidth=0, zorder=2)
        ax.plot(
            x_grid,
            fitted,
            linestyle=(0, (4.0, 2.6)),
            color=color,
            linewidth=style.line_width,
            solid_capstyle="round",
            zorder=4,
        )
        ax.scatter(
            data.gst[group],
            data.lst[group],
            s=style.scatter_size,
            facecolors=color,
            edgecolors=_darken(color),
            linewidths=0.55,
            alpha=style.scatter_alpha,
            zorder=3,
        )

    for index, group in enumerate(GROUPS):
        ax.text(
            0.028,
            0.965 - index * style.stats_line_gap,
            format_fit_line(group, fits[group]),
            transform=ax.transAxes,
            ha="left",
            va="top",
            color=palette.for_group(group),
            fontsize=style.stats_font_size,
            fontweight="bold",
            zorder=5,
        )


def _kde_curve(
    values: NDArray[np.float64],
    grid: NDArray[np.float64],
    bin_width: float,
) -> NDArray[np.float64]:
    kde = gaussian_kde(values)
    return kde(grid) * values.size * bin_width


def _draw_marginals(
    ax_x: Axes,
    ax_y: Axes,
    data: GroupedLMData,
    palette: Palette,
    style: ChartStyle,
) -> None:
    x_edges = np.linspace(*style.x_limits, style.hist_bins + 1)
    y_edges = np.linspace(*style.y_limits, style.hist_bins + 1)
    colors = [palette.for_group(group) for group in GROUPS]
    x_samples = [np.asarray(data.gst[group], dtype=float) for group in GROUPS]
    y_samples = [np.asarray(data.lst[group], dtype=float) for group in GROUPS]

    ax_x.hist(
        x_samples,
        bins=x_edges,
        stacked=True,
        color=colors,
        edgecolor="white",
        linewidth=0.55,
        alpha=style.hist_alpha,
        histtype="bar",
        zorder=1,
    )
    ax_y.hist(
        y_samples,
        bins=y_edges,
        stacked=True,
        color=colors,
        edgecolor="white",
        linewidth=0.55,
        alpha=style.hist_alpha,
        histtype="bar",
        orientation="horizontal",
        zorder=1,
    )

    x_grid = np.linspace(*style.x_limits, 400)
    y_grid = np.linspace(*style.y_limits, 400)
    x_width = float(np.diff(x_edges).mean())
    y_width = float(np.diff(y_edges).mean())
    for group, x_values, y_values in zip(GROUPS, x_samples, y_samples, strict=True):
        color = palette.for_group(group)
        ax_x.plot(
            x_grid,
            _kde_curve(x_values, x_grid, x_width),
            linestyle=(0, (4.0, 2.4)),
            color=color,
            linewidth=style.kde_width,
            zorder=3,
            clip_on=True,
        )
        ax_y.plot(
            _kde_curve(y_values, y_grid, y_width),
            y_grid,
            linestyle=(0, (4.0, 2.4)),
            color=color,
            linewidth=style.kde_width,
            zorder=3,
            clip_on=True,
        )

    ax_x.set_xlim(*style.x_limits)
    ax_y.set_ylim(*style.y_limits)
    ax_x.set_ylim(bottom=0)
    ax_y.set_xlim(left=0)


def create_figure(
    palette: Palette = PALETTES[0],
    data: GroupedLMData = DEFAULT_DATA,
    style: ChartStyle = DEFAULT_STYLE,
) -> Figure:
    """Create the grouped linear-fit figure without writing it to disk."""

    data.validate()
    style.validate()
    fits = {
        group: fit_ols(np.asarray(data.gst[group]), np.asarray(data.lst[group]))
        for group in data.groups
    }

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
        ax_joint = figure.add_axes(style.joint_bounds)
        ax_marg_x = figure.add_axes(style.marg_x_bounds)
        ax_marg_y = figure.add_axes(style.marg_y_bounds)
        ax_marg_x.set_zorder(1)
        ax_marg_y.set_zorder(1)
        ax_joint.set_zorder(3)
        ax_joint.patch.set_facecolor("white")
        ax_joint.patch.set_edgecolor("none")

        _style_joint_axis(ax_joint, style)
        _draw_joint(ax_joint, data, palette, style, fits)
        _draw_marginals(ax_marg_x, ax_marg_y, data, palette, style)
        _style_marginal_axis(ax_marg_x)
        _style_marginal_axis(ax_marg_y)

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
    data: GroupedLMData = DEFAULT_DATA,
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
    stem = f"grouped_lm_marginal_palette_{index:02d}_{palette.name.replace('-', '_')}"
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
        description="Render the grouped linear-fit scatter plot with marginal histograms."
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
        help="raster DPI; 250 reproduces the 2132×1962 reference (default: 250)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/grouped_lm_marginal"),
        help="destination directory (default: output/grouped_lm_marginal)",
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
            print(f"{index:2d}  {palette.name:32s}  {' '.join(palette.colors)}")
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
