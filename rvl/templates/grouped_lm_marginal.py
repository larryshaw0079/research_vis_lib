"""Grouped scatter with per-series linear fits and kernel-density marginals.

A joint panel carries the point cloud, a dashed ordinary-least-squares line per
series, a confidence band and the fitted equation; stacked histograms with
dashed KDE curves flank it on the top and right margins.

``DEFAULT_DATA`` comes from a Xiaohongshu carousel that repeats the same GST-LST
land-cover comparison 16 times, changing only the four-colour palette.  The post
publishes the annotated fits but not the point table, so the reference cloud is
fabricated by :func:`_synthesise_reference_points` to reproduce them.  That
helper exists only for ``DEFAULT_DATA``; :meth:`GroupedLmMarginalData.from_xy`
plots exactly the observations it is handed.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Final, Sequence

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

from ..contract import DataKind, Extent, Feature, Geometry, TemplateSpec
from ..palettes import Palette
from ..render import nice_step, resolve_limits, run_cli

SPEC: Final[TemplateSpec] = TemplateSpec(
    template_id="grouped-lm-marginal",
    title="Grouped scatter with linear fits and marginal densities",
    summary=(
        "A joint scatter panel with a per-series ordinary least squares fit and "
        "confidence band, annotated with the fitted equation, flanked by kernel "
        "density marginals on both axes."
    ),
    kinds=(DataKind.XY_SAMPLES,),
    geometry=Geometry.COMPOSITE,
    categories=Extent(3, None),
    series=Extent(1, 6),
    builder="GroupedLmMarginalData.from_xy",
    data_contract=(
        "Paired (x, y) observations per series. Each series needs at least three "
        "pairs for a fit and a confidence band to be meaningful."
    ),
    good_for=(
        "showing that two measured quantities co-vary, per group",
        "comparing regression slopes between land covers, sites or treatments",
        "reporting the fitted equation and R-squared alongside the cloud",
    ),
    avoid_when=(
        "x and y are not paired observations of the same units",
        "the relationship is clearly non-linear",
        "any series has fewer than three pairs",
    ),
    affinities=(
        (Feature.MANY_SERIES, -10.0),
        (Feature.MANY_CATEGORIES, 4.0),
    ),
    default_dpi=200,
    reference=(
        "Digitised from a Xiaohongshu carousel; DEFAULT_DATA points are synthetic "
        "samples reproducing the printed fits."
    ),
)


# Colours reconstructed from the 16 carousel frames (histogram bars and
# scatter markers).  JPEG compression in the source images was snapped to
# nearby saturated hex values while keeping the original series order.
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
class LinearFit:
    """One series' ordinary least squares fit.

    The drawn line, the confidence band and the printed equation all read this
    object, so the annotation cannot drift away from the geometry.
    """

    slope: float
    intercept: float
    r_squared: float
    p_value: float
    n: int
    x_mean: float
    sxx: float
    mse: float

    def predict(self, xs: Sequence[float] | NDArray[np.float64]) -> NDArray[np.float64]:
        return self.intercept + self.slope * np.asarray(xs, dtype=float)

    def confidence_band(
        self,
        xs: Sequence[float] | NDArray[np.float64],
        level: float = 0.95,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Lower and upper bounds of the mean-response band at ``level``."""

        grid = np.asarray(xs, dtype=float)
        fitted = self.predict(grid)
        if self.mse <= 0.0 or self.n <= 2:
            return fitted, fitted
        critical = float(t.ppf(0.5 + 0.5 * level, self.n - 2))
        error = np.sqrt(
            self.mse * (1.0 / self.n + (grid - self.x_mean) ** 2 / self.sxx)
        )
        return fitted - critical * error, fitted + critical * error


def fit_ols(
    x: Sequence[float] | NDArray[np.float64],
    y: Sequence[float] | NDArray[np.float64],
) -> LinearFit:
    """Least squares fit of ``y`` on ``x`` with its R-squared and p-value."""

    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    n = int(xs.size)
    x_mean = float(xs.mean())
    y_mean = float(ys.mean())
    centered = xs - x_mean
    sxx = float(np.dot(centered, centered))
    slope = float(np.dot(centered, ys - y_mean) / sxx)
    intercept = y_mean - slope * x_mean
    residual = ys - (intercept + slope * xs)
    sse = float(np.dot(residual, residual))
    sst = float(np.dot(ys - y_mean, ys - y_mean))
    mse = sse / (n - 2)
    if mse > 0.0:
        t_stat = slope / np.sqrt(mse / sxx)
        p_value = float(2.0 * t.sf(abs(t_stat), n - 2))
    else:
        # A perfectly collinear series has no residual scatter left to test.
        p_value = 0.0
    return LinearFit(
        slope=slope,
        intercept=intercept,
        r_squared=1.0 if sst <= 0.0 else 1.0 - sse / sst,
        p_value=p_value,
        n=n,
        x_mean=x_mean,
        sxx=sxx,
        mse=mse,
    )


_DEFAULT_EQUATION_FORMAT: Final[str] = (
    "y = {slope:.2f}x + {intercept:.2f}, R2 = {r_squared:.2f}"
)


@dataclass(frozen=True, slots=True)
class GroupedLmMarginalData:
    """Paired ``(x, y)`` observations per series, ragged across series."""

    series: tuple[str, ...]
    x: tuple[tuple[float, ...], ...]
    y: tuple[tuple[float, ...], ...]
    x_label: str = "x"
    y_label: str = "y"
    show_confidence_band: bool = True
    confidence_level: float = 0.95
    equation_format: str = _DEFAULT_EQUATION_FORMAT
    """Annotation template; see :meth:`equation_text` for the available fields.

    The default hard-codes a ``+`` before the intercept, so pass a template
    built from ``{sign}`` and ``{abs_intercept}`` when an intercept may be
    negative, and widen ``{slope:.2f}`` when slopes are smaller than 0.01.
    """

    @classmethod
    def from_xy(
        cls,
        *,
        series: Sequence[str],
        x: Sequence[Sequence[float]],
        y: Sequence[Sequence[float]],
        x_label: str = "x",
        y_label: str = "y",
        show_confidence_band: bool = True,
        confidence_level: float = 0.95,
        equation_format: str = _DEFAULT_EQUATION_FORMAT,
    ) -> "GroupedLmMarginalData":
        """Build from ``[series][observation]`` rows of paired measurements.

        ``x[i]`` and ``y[i]`` must have the same length; different series may
        carry different numbers of pairs.
        """

        built = cls(
            series=tuple(str(name) for name in series),
            x=tuple(tuple(float(value) for value in row) for row in x),
            y=tuple(tuple(float(value) for value in row) for row in y),
            x_label=x_label,
            y_label=y_label,
            show_confidence_band=bool(show_confidence_band),
            confidence_level=float(confidence_level),
            equation_format=equation_format,
        )
        built.validate()
        return built

    def validate(self) -> None:
        if not self.series:
            raise ValueError("need at least one series")
        if len(set(self.series)) != len(self.series):
            raise ValueError("series labels must be unique")
        if len(self.x) != len(self.series) or len(self.y) != len(self.series):
            raise ValueError(
                f"x has {len(self.x)} rows and y has {len(self.y)} rows but there "
                f"are {len(self.series)} series"
            )
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must sit strictly between 0 and 1")
        for label, x_row, y_row in zip(self.series, self.x, self.y, strict=True):
            if len(x_row) != len(y_row):
                raise ValueError(
                    f"series {label!r} has {len(x_row)} x values but "
                    f"{len(y_row)} y values"
                )
            if len(x_row) < 3:
                raise ValueError(
                    f"series {label!r} has {len(x_row)} pair(s); a fit with a "
                    "confidence band needs at least three"
                )
            xs = np.asarray(x_row, dtype=float)
            ys = np.asarray(y_row, dtype=float)
            if not (np.all(np.isfinite(xs)) and np.all(np.isfinite(ys))):
                raise ValueError(f"series {label!r} contains a non-finite value")
            if np.ptp(xs) <= 0.0:
                raise ValueError(
                    f"series {label!r} has a single x value; no slope exists"
                )
        try:
            self.equation_format.format(**_equation_fields("series", _PROBE_FIT))
        except (IndexError, KeyError, ValueError) as exc:
            raise ValueError(
                f"equation_format {self.equation_format!r} is not usable"
            ) from exc

    def fit(self, index: int) -> LinearFit:
        """Least squares fit of series ``index``."""

        return fit_ols(self.x[index], self.y[index])

    def equation_text(self, index: int) -> str:
        """Render ``equation_format`` for series ``index``.

        Available fields: ``series``, ``slope``, ``intercept``,
        ``abs_intercept``, ``sign``, ``r_squared``, ``p_value``, ``p_text``
        and ``n``.
        """

        return self.equation_format.format(
            **_equation_fields(self.series[index], self.fit(index))
        )

    def x_range(self) -> tuple[float, float]:
        flat = np.concatenate([np.asarray(row, dtype=float) for row in self.x])
        return float(flat.min()), float(flat.max())

    def y_range(self) -> tuple[float, float]:
        flat = np.concatenate([np.asarray(row, dtype=float) for row in self.y])
        return float(flat.min()), float(flat.max())


def _equation_fields(name: str, fit: LinearFit) -> dict[str, Any]:
    return {
        "series": name,
        "slope": fit.slope,
        "intercept": fit.intercept,
        "abs_intercept": abs(fit.intercept),
        "sign": "+" if fit.intercept >= 0 else "\u2212",
        "r_squared": fit.r_squared,
        "p_value": fit.p_value,
        "p_text": (
            "p < 0.001" if fit.p_value < 0.001 else f"p = {fit.p_value:.3f}"
        ),
        "n": fit.n,
    }


# Stand-in used to check a format string before any real fit is available.
_PROBE_FIT: Final[LinearFit] = LinearFit(
    slope=1.0,
    intercept=0.0,
    r_squared=1.0,
    p_value=1.0,
    n=3,
    x_mean=0.0,
    sxx=1.0,
    mse=1.0,
)


_REFERENCE_SERIES: Final[tuple[str, ...]] = ("Grass", "Land", "Water", "Urban")

# ``(mean, sd, low, high)`` bounds for the fabricated GST draws, per series.
_REFERENCE_X_SPECS: Final[tuple[tuple[float, float, float, float], ...]] = (
    (8.2, 2.05, 4.2, 13.0),
    (12.1, 2.35, 6.8, 17.5),
    (16.3, 2.45, 10.5, 22.0),
    (20.4, 2.15, 15.5, 25.5),
)

# ``(slope, intercept, R^2)`` as printed in the reference annotations.
_REFERENCE_FITS: Final[tuple[tuple[float, float, float], ...]] = (
    (1.69, 2.09, 0.845),
    (1.24, 4.17, 0.717),
    (0.91, 3.16, 0.579),
    (0.44, 8.33, 0.250),
)

_REFERENCE_POINTS_PER_SERIES: Final[int] = 25
_REFERENCE_SEED: Final[int] = 20260818

_REFERENCE_EQUATION_FORMAT: Final[str] = (
    "{series}: y = {slope:.2f}x {sign} {abs_intercept:.2f}, "
    "$R^2$ = {r_squared:.3f}, {p_text}"
)


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


def _synthesise_reference_points(
    seed: int = _REFERENCE_SEED,
) -> tuple[tuple[tuple[float, ...], ...], tuple[tuple[float, ...], ...]]:
    """Fabricate the reference point cloud for ``DEFAULT_DATA`` only.

    The source post publishes the annotated fits but not the underlying table,
    so this invents 25 pairs per series whose OLS slope, intercept, R-squared
    and p-value round to the printed values.  No user data ever passes through
    here: :meth:`GroupedLmMarginalData.from_xy` plots real observations only.
    """

    rng = np.random.default_rng(seed)
    x_rows: list[tuple[float, ...]] = []
    y_rows: list[tuple[float, ...]] = []
    for (mean, std, low, high), (slope, intercept, r_squared) in zip(
        _REFERENCE_X_SPECS, _REFERENCE_FITS, strict=True
    ):
        xs = _sample_x(_REFERENCE_POINTS_PER_SERIES, mean, std, low, high, rng)
        ys = _response_with_target_fit(xs, slope, intercept, r_squared, rng)
        x_rows.append(tuple(float(value) for value in xs))
        y_rows.append(tuple(float(value) for value in ys))
    return tuple(x_rows), tuple(y_rows)


_REFERENCE_X, _REFERENCE_Y = _synthesise_reference_points()

DEFAULT_DATA: Final[GroupedLmMarginalData] = GroupedLmMarginalData.from_xy(
    series=_REFERENCE_SERIES,
    x=_REFERENCE_X,
    y=_REFERENCE_Y,
    x_label="GST",
    y_label="LST",
    equation_format=_REFERENCE_EQUATION_FORMAT,
)


@dataclass(frozen=True, slots=True)
class ChartStyle:
    """Layout tuned to the 2132 x 1962 reference image.

    ``x_tick_step``, ``y_tick_step`` and ``y_tick_format`` only apply while the
    pinned limits still contain the data; once a limit auto-fits, the ticks fall
    back to a derived step so a rescaled axis stays readable.
    """

    figure_size: tuple[float, float] = (8.528, 7.848)
    joint_bounds: tuple[float, float, float, float] = (0.1124, 0.1037, 0.7215, 0.7306)
    marg_x_bounds: tuple[float, float, float, float] = (0.1124, 0.8343, 0.7215, 0.1472)
    marg_y_bounds: tuple[float, float, float, float] = (0.8339, 0.1037, 0.1465, 0.7306)
    x_limits: tuple[float, float] | None = (2.0, 28.0)
    y_limits: tuple[float, float] | None = (7.0, 29.0)
    x_tick_step: float | None = 5.0
    y_tick_step: float | None = 2.5
    y_tick_format: str | None = "%.1f"
    hist_bins: int = 22
    fit_grid_points: int = 200
    kde_grid_points: int = 400
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

    def validate(self, *, series: int) -> None:
        if min(self.figure_size) <= 0:
            raise ValueError("figure_size must be positive")
        for name, limits in (("x_limits", self.x_limits), ("y_limits", self.y_limits)):
            if limits is not None and limits[1] <= limits[0]:
                raise ValueError(f"{name} must be increasing")
        if self.hist_bins < 4:
            raise ValueError("hist_bins must be at least 4")
        if self.fit_grid_points < 16 or self.kde_grid_points < 16:
            raise ValueError("grid point counts must be at least 16")
        if series < 1:
            raise ValueError("need at least one series")
        if series * self.stats_line_gap >= 0.95:
            raise ValueError(
                f"{series} annotation lines do not fit above the cloud; "
                "lower stats_line_gap"
            )


DEFAULT_STYLE: Final[ChartStyle] = ChartStyle()


def _tick_step(
    limits: tuple[float, float],
    pinned: tuple[float, float] | None,
    step: float | None,
) -> float:
    """The pinned tick step while the pinned limits hold, else a derived one."""

    if step and pinned is not None and limits == tuple(pinned):
        return float(step)
    return nice_step(limits[1] - limits[0])


def _darken(color: str, factor: float = 0.62) -> tuple[float, float, float]:
    red, green, blue = to_rgb(color)
    return (red * factor, green * factor, blue * factor)


def _style_joint_axis(
    ax: Axes,
    data: GroupedLmMarginalData,
    style: ChartStyle,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
) -> None:
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.xaxis.set_major_locator(
        MultipleLocator(_tick_step(x_limits, style.x_limits, style.x_tick_step))
    )
    ax.yaxis.set_major_locator(
        MultipleLocator(_tick_step(y_limits, style.y_limits, style.y_tick_step))
    )
    y_pinned = style.y_limits is not None and y_limits == tuple(style.y_limits)
    if style.y_tick_format and y_pinned:
        ax.yaxis.set_major_formatter(FormatStrFormatter(style.y_tick_format))
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
    ax.set_xlabel(
        data.x_label, fontsize=style.label_font_size, fontweight="bold", labelpad=8
    )
    ax.set_ylabel(
        data.y_label, fontsize=style.label_font_size, fontweight="bold", labelpad=6
    )
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
    data: GroupedLmMarginalData,
    palette: Palette,
    style: ChartStyle,
    x_limits: tuple[float, float],
) -> None:
    grid = np.linspace(*x_limits, style.fit_grid_points)
    colors = palette.take(len(data.series))
    for index in range(len(data.series)):
        color = colors[index]
        fit = data.fit(index)
        if data.show_confidence_band:
            lower, upper = fit.confidence_band(grid, data.confidence_level)
            ax.fill_between(
                grid,
                lower,
                upper,
                color=color,
                alpha=style.ci_alpha,
                linewidth=0,
                zorder=2,
            )
        ax.plot(
            grid,
            fit.predict(grid),
            linestyle=(0, (4.0, 2.6)),
            color=color,
            linewidth=style.line_width,
            solid_capstyle="round",
            zorder=4,
        )
        ax.scatter(
            np.asarray(data.x[index], dtype=float),
            np.asarray(data.y[index], dtype=float),
            s=style.scatter_size,
            facecolors=color,
            edgecolors=_darken(color),
            linewidths=0.55,
            alpha=style.scatter_alpha,
            zorder=3,
        )

    for index in range(len(data.series)):
        ax.text(
            0.028,
            0.965 - index * style.stats_line_gap,
            data.equation_text(index),
            transform=ax.transAxes,
            ha="left",
            va="top",
            color=colors[index],
            fontsize=style.stats_font_size,
            fontweight="bold",
            zorder=5,
        )


def _kde_curve(
    values: NDArray[np.float64],
    grid: NDArray[np.float64],
    bin_width: float,
) -> NDArray[np.float64] | None:
    """Density curve scaled to the histogram, or ``None`` when it is undefined.

    ``gaussian_kde`` needs at least two distinct values; a series that is one
    repeated reading keeps its histogram bars and simply loses its curve.
    """

    sample = np.asarray(values, dtype=float)
    if np.unique(sample).size < 2:
        return None
    try:
        kde = gaussian_kde(sample)
    except (ValueError, np.linalg.LinAlgError):
        return None
    return kde(grid) * sample.size * bin_width


def _draw_marginals(
    ax_x: Axes,
    ax_y: Axes,
    data: GroupedLmMarginalData,
    palette: Palette,
    style: ChartStyle,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
) -> None:
    x_edges = np.linspace(*x_limits, style.hist_bins + 1)
    y_edges = np.linspace(*y_limits, style.hist_bins + 1)
    colors = list(palette.take(len(data.series)))
    x_samples = [np.asarray(row, dtype=float) for row in data.x]
    y_samples = [np.asarray(row, dtype=float) for row in data.y]

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

    x_grid = np.linspace(*x_limits, style.kde_grid_points)
    y_grid = np.linspace(*y_limits, style.kde_grid_points)
    x_width = float(np.diff(x_edges).mean())
    y_width = float(np.diff(y_edges).mean())
    for index, (x_values, y_values) in enumerate(zip(x_samples, y_samples, strict=True)):
        color = colors[index]
        x_density = _kde_curve(x_values, x_grid, x_width)
        if x_density is not None:
            ax_x.plot(
                x_grid,
                x_density,
                linestyle=(0, (4.0, 2.4)),
                color=color,
                linewidth=style.kde_width,
                zorder=3,
                clip_on=True,
            )
        y_density = _kde_curve(y_values, y_grid, y_width)
        if y_density is not None:
            ax_y.plot(
                y_density,
                y_grid,
                linestyle=(0, (4.0, 2.4)),
                color=color,
                linewidth=style.kde_width,
                zorder=3,
                clip_on=True,
            )

    ax_x.set_xlim(*x_limits)
    ax_y.set_ylim(*y_limits)
    ax_x.set_ylim(bottom=0)
    ax_y.set_xlim(left=0)


def create_figure(
    palette: Palette = PALETTES[0],
    data: GroupedLmMarginalData = DEFAULT_DATA,
    style: ChartStyle = DEFAULT_STYLE,
) -> Figure:
    """Create the grouped linear-fit figure without writing it to disk."""

    data.validate()
    style.validate(series=len(data.series))
    x_limits = resolve_limits(style.x_limits, *data.x_range(), include_zero=False)
    y_limits = resolve_limits(style.y_limits, *data.y_range(), include_zero=False)

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

        _style_joint_axis(ax_joint, data, style, x_limits, y_limits)
        _draw_joint(ax_joint, data, palette, style, x_limits)
        _draw_marginals(
            ax_marg_x, ax_marg_y, data, palette, style, x_limits, y_limits
        )
        _style_marginal_axis(ax_marg_x)
        _style_marginal_axis(ax_marg_y)

    return figure


def main(argv: Sequence[str] | None = None) -> int:
    return run_cli(sys.modules[__name__], argv)


if __name__ == "__main__":
    raise SystemExit(main())
