"""Flower / petal chart: one rounded petal per group around a shared core.

Each petal is a rounded annular sector with radial sides, a circular inner edge
and filleted outer tips.  The large number inside a petal is that group's total
count and the parenthetical number is the count unique to it; the central disc
reports the count shared by every group.  Petal size is uniform, so nothing
here encodes a value as an area.

``DEFAULT_DATA`` holds the 10-group OTU counts digitised from a Xiaohongshu
carousel that repeats the same flower 18 times, changing only the palette.  The
source post does not name the underlying table.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Final, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MatplotlibPath
from numpy.typing import NDArray

from ..contract import DataKind, Extent, Feature, Geometry, TemplateSpec
from ..palettes import Palette
from ..render import resolve_limits, run_cli

SPEC: Final[TemplateSpec] = TemplateSpec(
    template_id="flower-plot",
    title="Flower / petal chart of per-group and shared counts",
    summary=(
        "One rounded petal per group, labelled with the group's total and unique "
        "counts, arranged around a central disc that reports the count shared by "
        "every group."
    ),
    kinds=(DataKind.SET_OVERLAP,),
    geometry=Geometry.CIRCULAR,
    categories=Extent(3, 24),
    series=Extent(3, 24),
    builder="FlowerPlotData.from_counts",
    data_contract=(
        "Per group: a total count and the count unique to that group, plus one "
        "shared-core count common to every group."
    ),
    good_for=(
        "OTU, gene or feature counts across many treatment groups",
        "showing a large shared core alongside group-specific members",
        "many groups at once, where an UpSet plot would need too many columns",
    ),
    avoid_when=(
        "pairwise intersections matter, which petals cannot show",
        "unique counts exceed totals, which is not a valid overlap",
        "fewer than three groups, where a Venn diagram is clearer",
    ),
    requires=(Feature.NON_NEGATIVE,),
    affinities=(
        (Feature.MANY_CATEGORIES, 7.0),
        (Feature.LONG_LABELS, -5.0),
    ),
    default_dpi=200,
    reference="Digitised from a Xiaohongshu carousel; petal labels were read off the image.",
)


# Palettes reconstructed from carousel images 1-18.  Named qualitative sets
# use canonical ColorBrewer / Tableau / matplotlib / Paul Tol / Okabe-Ito
# hex values; the rest are median colours sampled from the petal interiors.
PALETTES: Final[tuple[Palette, ...]] = (
    Palette(
        "gray-peach-orchid",
        (
            "#BDBDBD",
            "#F8CDAB",
            "#F09B88",
            "#EC655D",
            "#A1DFC8",
            "#6FCDD6",
            "#75A7EE",
            "#ADB8F5",
            "#C6A5E0",
            "#F6C3DF",
        ),
    ),
    Palette(
        "tab10",
        (
            "#1F77B4",
            "#FF7F0E",
            "#2CA02C",
            "#D62728",
            "#9467BD",
            "#8C564B",
            "#E377C2",
            "#7F7F7F",
            "#BCBD22",
            "#17BECF",
        ),
    ),
    Palette(
        "set3",
        (
            "#8DD3C7",
            "#FFFFB3",
            "#BEBADA",
            "#FB8072",
            "#80B1D3",
            "#FDB462",
            "#B3DE69",
            "#FCCDE5",
            "#D9D9D9",
            "#BC80BD",
        ),
    ),
    Palette(
        "paired",
        (
            "#A6CEE3",
            "#1F78B4",
            "#B2DF8A",
            "#33A02C",
            "#FB9A99",
            "#E31A1C",
            "#FDBF6F",
            "#FF7F00",
            "#CAB2D6",
            "#6A3D9A",
        ),
    ),
    Palette(
        "pastel1",
        (
            "#FBB4AE",
            "#B3CDE3",
            "#CCEBC5",
            "#DECBE4",
            "#FED9A6",
            "#FFFFCC",
            "#E5D8BD",
            "#FDDAEC",
            "#F2F2F2",
            "#B3B3B3",
        ),
    ),
    Palette(
        "set1",
        (
            "#E41A1C",
            "#377EB8",
            "#4DAF4A",
            "#984EA3",
            "#FF7F00",
            "#FFFF33",
            "#A65628",
            "#F781BF",
            "#999999",
            "#B15928",
        ),
    ),
    Palette(
        "dark2",
        (
            "#1B9E77",
            "#D95F02",
            "#7570B3",
            "#E7298A",
            "#66A61E",
            "#E6AB02",
            "#A6761D",
            "#666666",
            "#B3E2CD",
            "#999999",
        ),
    ),
    Palette(
        "accent",
        (
            "#7FC97F",
            "#BEAED4",
            "#FDC086",
            "#FFFF99",
            "#386CB0",
            "#F0027F",
            "#BF5B17",
            "#666666",
            "#FDCDAC",
            "#444444",
        ),
    ),
    Palette(
        "tableau10",
        (
            "#4E79A7",
            "#F28E2B",
            "#E15759",
            "#76B7B2",
            "#59A14F",
            "#EDC948",
            "#B07AA1",
            "#FF9D9A",
            "#9C755F",
            "#BAB0AC",
        ),
    ),
    Palette(
        "okabe-ito",
        (
            "#E69F00",
            "#56B4E9",
            "#009E73",
            "#F0E442",
            "#0072B2",
            "#D55E00",
            "#CC79A7",
            "#8DD3C7",
            "#737373",
            "#CCCCCC",
        ),
    ),
    Palette(
        "tol-muted",
        (
            "#332288",
            "#88CCEE",
            "#44AA99",
            "#117733",
            "#999933",
            "#DDCC77",
            "#CC6677",
            "#882255",
            "#AA4499",
            "#DDDDDD",
        ),
    ),
    Palette(
        "dust-rainbow",
        (
            "#88CDE5",
            "#B8D4A1",
            "#E7D893",
            "#EFA88E",
            "#D78994",
            "#B5759E",
            "#8D77A5",
            "#7A93B1",
            "#999999",
            "#555555",
        ),
    ),
    Palette(
        "coral-slate-gold",
        (
            "#FCA07C",
            "#20B1AA",
            "#778898",
            "#9470D8",
            "#3DB271",
            "#FB6349",
            "#4882B4",
            "#D8A523",
            "#FD69B2",
            "#CD5E5E",
        ),
    ),
    Palette(
        "pastel-candy",
        (
            "#FCB4BB",
            "#FEDFB9",
            "#FFFEB9",
            "#BAFDC9",
            "#BAE1FE",
            "#D7BFD7",
            "#FEBFCD",
            "#E5E6F7",
            "#EE807F",
            "#21B0A8",
        ),
    ),
    Palette(
        "atlantic-sunset",
        (
            "#264653",
            "#2A9D8F",
            "#E9C46A",
            "#F4A261",
            "#E76F51",
            "#8AB07D",
            "#BABB75",
            "#DDA06E",
            "#C16C5D",
            "#5B7470",
        ),
    ),
    Palette(
        "gold-teal-rust",
        (
            "#FCB706",
            "#045F73",
            "#0E9396",
            "#94D1BD",
            "#E9D7A5",
            "#EC9B01",
            "#CC6708",
            "#BB400A",
            "#AE2114",
            "#9B2329",
        ),
    ),
    Palette(
        "lime-navy",
        (
            "#D9ED93",
            "#B6E48C",
            "#99D98B",
            "#77C793",
            "#52B69A",
            "#349FA2",
            "#198AAD",
            "#1E769F",
            "#206092",
            "#1A4E78",
        ),
    ),
    Palette(
        "neon-rainbow",
        (
            "#FB595E",
            "#FDCA39",
            "#8BC927",
            "#1D82C4",
            "#6A4C92",
            "#EF5BB4",
            "#01F3D3",
            "#FBE442",
            "#00BBF8",
            "#9B5EE6",
        ),
    ),
)


@dataclass(frozen=True, slots=True)
class FlowerPlotData:
    """Per-group total and unique counts plus the count shared by every group.

    ``totals[i]`` and ``uniques[i]`` belong to ``groups[i]``, and the petals are
    laid out clockwise from north in that order.
    """

    groups: tuple[str, ...]
    totals: tuple[float, ...]
    uniques: tuple[float, ...]
    core: float
    core_label: str = "Core"
    unit_label: str = ""
    count_format: str = "{:.0f}"

    @classmethod
    def from_counts(
        cls,
        *,
        groups: Sequence[str],
        totals: Sequence[float],
        uniques: Sequence[float],
        core: float,
        core_label: str = "Core",
        unit_label: str = "",
        count_format: str = "{:.0f}",
    ) -> "FlowerPlotData":
        """Build from parallel group, total and unique sequences.

        ``unit_label`` is optional; when given it is printed under the core count
        to name what is being counted, for example ``"OTUs"``.
        """

        built = cls(
            groups=tuple(str(name) for name in groups),
            totals=tuple(float(value) for value in totals),
            uniques=tuple(float(value) for value in uniques),
            core=float(core),
            core_label=core_label,
            unit_label=unit_label,
            count_format=count_format,
        )
        built.validate()
        return built

    def validate(self) -> None:
        if len(self.groups) < 3:
            raise ValueError("need at least three groups")
        if len(set(self.groups)) != len(self.groups):
            raise ValueError("group labels must be unique")
        if len(self.totals) != len(self.groups):
            raise ValueError(
                f"totals has {len(self.totals)} entries but there are "
                f"{len(self.groups)} groups"
            )
        if len(self.uniques) != len(self.groups):
            raise ValueError(
                f"uniques has {len(self.uniques)} entries but there are "
                f"{len(self.groups)} groups"
            )
        for group, total, unique in zip(
            self.groups, self.totals, self.uniques, strict=True
        ):
            for name, value in (("total", total), ("unique", unique)):
                if not math.isfinite(value) or value < 0.0:
                    raise ValueError(
                        f"{group} has {name} count {value!r}; counts must be finite "
                        "and non-negative"
                    )
            if unique > total:
                raise ValueError(
                    f"{group} has unique count {unique:g} above its total {total:g}"
                )
        if not math.isfinite(self.core) or self.core < 0.0:
            raise ValueError(
                f"core count {self.core!r} must be finite and non-negative"
            )
        try:
            self.count_format.format(1.0)
        except (IndexError, KeyError, ValueError) as exc:
            raise ValueError(f"count_format {self.count_format!r} is not usable") from exc

    def petal_lines(self, index: int) -> tuple[str, str]:
        """The two label lines printed inside petal ``index``."""

        return (
            self.count_format.format(self.totals[index]),
            f"({self.count_format.format(self.uniques[index])})",
        )

    def core_lines(self) -> tuple[str, ...]:
        """Central disc text: the label, the core count and the optional unit."""

        lines = [self.core_label, self.count_format.format(self.core)]
        if self.unit_label:
            lines.append(self.unit_label)
        return tuple(lines)


_REFERENCE_GROUPS: Final[tuple[str, ...]] = (
    "CK",
    "Cu-L",
    "Cu-M",
    "Cu-H",
    "Cr-L",
    "Cr-M",
    "Cr-H",
    "Mix-L",
    "Mix-M",
    "Mix-H",
)

# Totals and uniques digitised from the labelled petals of the reference image.
_REFERENCE_TOTALS: Final[tuple[float, ...]] = (
    1755,
    2167,
    2182,
    2080,
    2308,
    1906,
    2014,
    2190,
    1877,
    1641,
)

_REFERENCE_UNIQUES: Final[tuple[float, ...]] = (
    171,
    167,
    158,
    181,
    223,
    170,
    157,
    168,
    127,
    105,
)

DEFAULT_DATA: Final[FlowerPlotData] = FlowerPlotData.from_counts(
    groups=_REFERENCE_GROUPS,
    totals=_REFERENCE_TOTALS,
    uniques=_REFERENCE_UNIQUES,
    core=936,
)


@dataclass(frozen=True, slots=True)
class ChartStyle:
    """Layout tuned to the 2560 x 2560 reference canvas.

    Angular geometry is derived from the data: each group owns
    ``360 / n_groups`` degrees, of which ``petal_fill`` is covered by the petal.
    With the reference ten groups this reproduces the original 36-degree pitch
    and 31.2-degree petals exactly.
    """

    figure_size: tuple[float, float] = (10.24, 10.24)
    x_limits: tuple[float, float] | None = (-1.0, 1.0)
    y_limits: tuple[float, float] | None = (-1.0, 1.0)
    inner_radius: float = 0.248
    outer_radius: float = 0.648
    label_radius: float = 0.755
    label_clearance: float = 0.09
    """Radial room reserved past ``label_radius`` when auto-fitting the axes."""

    petal_fill: float = 31.2 / 36.0
    inner_corner_radius: float = 0.012
    outer_corner_radius: float = 0.055
    core_font_size: float = 20.0
    core_line_gap: float = 0.058
    value_font_size: float = 13.0
    value_line_gap: float = 0.040
    label_font_size: float = 14.0
    min_font_size: float = 6.0
    luminance_cutoff: float = 0.22

    def validate(self, *, groups: int) -> None:
        if min(self.figure_size) <= 0:
            raise ValueError("figure_size must be positive")
        if groups < 1:
            raise ValueError("groups must be positive")
        if not 0 < self.inner_radius < self.outer_radius:
            raise ValueError("radii must satisfy 0 < inner < outer")
        if self.label_radius <= self.outer_radius:
            raise ValueError("label_radius must sit outside the petals")
        if not 0 < self.petal_fill < 1.0:
            raise ValueError("petal_fill must sit in (0, 1) to leave a gap between petals")
        thickness = self.outer_radius - self.inner_radius
        if not 0 <= self.inner_corner_radius < 0.5 * thickness:
            raise ValueError("inner_corner_radius must be smaller than half the petal thickness")
        if not 0 <= self.outer_corner_radius < 0.5 * thickness:
            raise ValueError("outer_corner_radius must be smaller than half the petal thickness")
        if self.min_font_size <= 0:
            raise ValueError("min_font_size must be positive")


DEFAULT_STYLE: Final[ChartStyle] = ChartStyle()


def petal_geometry(style: ChartStyle, *, groups: int) -> tuple[float, float]:
    """Return ``(angular_pitch, petal_width)`` in degrees for ``groups`` petals."""

    if groups < 1:
        raise ValueError("groups must be positive")
    pitch = 360.0 / groups
    return pitch, style.petal_fill * pitch


def petal_center_degrees(index: int, *, groups: int) -> float:
    """Math angle (CCW from +x) of petal ``index``, clockwise from north."""

    return (90.0 - index * (360.0 / groups)) % 360.0


def _points_per_unit(style: ChartStyle, limits: tuple[float, float]) -> float:
    """Figure points spanned by one data unit along the x-axis."""

    return 72.0 * style.figure_size[0] / (limits[1] - limits[0])


def _fitted_font_size(
    base: float, *, room_points: float, characters: int, minimum: float
) -> float:
    """Shrink ``base`` when ``characters`` would not fit into ``room_points``.

    The reference ten petals leave enough room per character to keep ``base``, so
    only denser flowers than the reference lose type size.
    """

    per_character = room_points / max(characters, 1)
    return float(min(base, max(minimum, 0.9 * per_character)))


def _contrasting_text_color(color: str, cutoff: float) -> str:
    red, green, blue = to_rgb(color)

    def linear(channel: float) -> float:
        if channel <= 0.04045:
            return channel / 12.92
        return ((channel + 0.055) / 1.055) ** 2.4

    luminance = (
        0.2126 * linear(red) + 0.7152 * linear(green) + 0.0722 * linear(blue)
    )
    return "#FFFFFF" if luminance < cutoff else "#000000"


def _label_rotation(theta_degrees: float) -> float:
    """Keep outer labels tangential and right-side up."""

    rotation = (theta_degrees - 90.0) % 360.0
    if 90.0 < rotation < 270.0:
        rotation += 180.0
    return (rotation + 180.0) % 360.0 - 180.0


def _unit(angle: float) -> NDArray[np.float64]:
    return np.array([np.cos(angle), np.sin(angle)], dtype=float)


def _arc_points(
    radius: float, start: float, stop: float, count: int
) -> NDArray[np.float64]:
    angles = np.linspace(start, stop, count)
    return radius * np.column_stack((np.cos(angles), np.sin(angles)))


def _circle_arc(
    center: NDArray[np.float64],
    start: NDArray[np.float64],
    stop: NDArray[np.float64],
    count: int,
) -> NDArray[np.float64]:
    """Short arc from ``start`` to ``stop`` on the circle around ``center``."""

    radius = float(np.linalg.norm(start - center))
    a0 = np.arctan2(start[1] - center[1], start[0] - center[0])
    a1 = np.arctan2(stop[1] - center[1], stop[0] - center[0])
    ccw = (a1 - a0) % (2.0 * np.pi)
    cw = (a0 - a1) % (2.0 * np.pi)
    angles = np.linspace(a0, a0 + ccw, count) if ccw <= cw else np.linspace(a0, a0 - cw, count)
    return center + radius * np.column_stack((np.cos(angles), np.sin(angles)))


def _clamped_corner_radius(
    requested: float,
    *,
    radial_room: float,
    half_width: float,
) -> float:
    return float(max(0.0, min(requested, 0.48 * radial_room, 0.90 * half_width)))


def _polar_fillet(
    theta: float,
    radius: float,
    corner_radius: float,
    *,
    inner: bool,
    toward_increasing: bool,
    count: int,
) -> tuple[NDArray[np.float64], float]:
    """Return a fillet at a polar-sector corner and the adjacent arc angle."""

    radial = _unit(theta)
    side = 1.0 if toward_increasing else -1.0
    if inner:
        center_radius = radius + corner_radius
        alpha = np.arcsin(
            np.clip(corner_radius / max(center_radius, 1e-9), 0.0, 0.99)
        )
        center = center_radius * _unit(theta + side * alpha)
        on_side = (radius + corner_radius) * radial
        on_arc = radius * _unit(theta + side * alpha)
    else:
        center_radius = radius - corner_radius
        alpha = np.arcsin(
            np.clip(corner_radius / max(center_radius, 1e-9), 0.0, 0.99)
        )
        center = center_radius * _unit(theta + side * alpha)
        on_side = (radius - corner_radius) * radial
        on_arc = radius * _unit(theta + side * alpha)
    return _circle_arc(center, on_side, on_arc, count), float(theta + side * alpha)


def rounded_petal_vertices(
    center_degrees: float,
    width_degrees: float,
    inner_radius: float,
    outer_radius: float,
    inner_corner: float,
    outer_corner: float,
    *,
    arc_points: int = 40,
    fillet_points: int = 16,
) -> NDArray[np.float64]:
    """Vertices of a rounded annular sector, walking counter-clockwise."""

    theta1 = np.deg2rad(center_degrees - 0.5 * width_degrees)
    theta2 = np.deg2rad(center_degrees + 0.5 * width_degrees)
    half_angle = 0.5 * (theta2 - theta1)
    radial_room = outer_radius - inner_radius
    rho_out = _clamped_corner_radius(
        outer_corner,
        radial_room=radial_room,
        half_width=outer_radius * np.sin(half_angle),
    )
    rho_in = _clamped_corner_radius(
        inner_corner,
        radial_room=radial_room,
        half_width=inner_radius * np.sin(half_angle),
    )

    inner1, inner_a1 = _polar_fillet(
        theta1,
        inner_radius,
        rho_in,
        inner=True,
        toward_increasing=True,
        count=fillet_points,
    )
    inner2, inner_a2 = _polar_fillet(
        theta2,
        inner_radius,
        rho_in,
        inner=True,
        toward_increasing=False,
        count=fillet_points,
    )
    outer2, outer_a2 = _polar_fillet(
        theta2,
        outer_radius,
        rho_out,
        inner=False,
        toward_increasing=False,
        count=fillet_points,
    )
    outer1, outer_a1 = _polar_fillet(
        theta1,
        outer_radius,
        rho_out,
        inner=False,
        toward_increasing=True,
        count=fillet_points,
    )

    # CCW around the petal: inner θ1 → inner arc → inner θ2 → outer θ2 →
    # outer arc (decreasing θ) → outer θ1.  Reverse fillets that were built
    # from the radial side toward the arc when the walk hits the arc first.
    return np.vstack(
        (
            inner1,
            _arc_points(inner_radius, inner_a1, inner_a2, arc_points),
            inner2[::-1],
            outer2,
            _arc_points(outer_radius, outer_a2, outer_a1, arc_points),
            outer1[::-1],
        )
    )


def _draw_petals(
    ax: Axes,
    data: FlowerPlotData,
    palette: Palette,
    style: ChartStyle,
    *,
    width_degrees: float,
    font_size: float,
) -> None:
    n_groups = len(data.groups)
    value_radius = 0.5 * (style.inner_radius + style.outer_radius)
    colors = palette.take(n_groups)
    for index in range(n_groups):
        center = petal_center_degrees(index, groups=n_groups)
        color = colors[index]
        vertices = rounded_petal_vertices(
            center,
            width_degrees,
            style.inner_radius,
            style.outer_radius,
            style.inner_corner_radius,
            style.outer_corner_radius,
        )
        codes = np.full(len(vertices) + 1, MatplotlibPath.LINETO, dtype=np.uint8)
        codes[0] = MatplotlibPath.MOVETO
        codes[-1] = MatplotlibPath.CLOSEPOLY
        path = MatplotlibPath(np.vstack((vertices, vertices[0])), codes)
        ax.add_patch(
            PathPatch(
                path,
                facecolor=color,
                edgecolor=color,
                linewidth=0.35,
                joinstyle="round",
                zorder=2,
            )
        )
        theta = np.deg2rad(center)
        text_color = _contrasting_text_color(color, style.luminance_cutoff)
        x = value_radius * np.cos(theta)
        y = value_radius * np.sin(theta)
        total_line, unique_line = data.petal_lines(index)
        ax.text(
            x,
            y + 0.5 * style.value_line_gap,
            total_line,
            ha="center",
            va="center",
            fontsize=font_size,
            color=text_color,
            zorder=4,
        )
        ax.text(
            x,
            y - 0.5 * style.value_line_gap,
            unique_line,
            ha="center",
            va="center",
            fontsize=font_size,
            color=text_color,
            zorder=4,
        )


def _draw_labels(
    ax: Axes,
    data: FlowerPlotData,
    style: ChartStyle,
    *,
    font_size: float,
) -> None:
    n_groups = len(data.groups)
    for index, group in enumerate(data.groups):
        center = petal_center_degrees(index, groups=n_groups)
        theta = np.deg2rad(center)
        ax.text(
            style.label_radius * np.cos(theta),
            style.label_radius * np.sin(theta),
            group,
            ha="center",
            va="center",
            rotation=_label_rotation(center),
            rotation_mode="anchor",
            fontsize=font_size,
            color="black",
            clip_on=False,
            zorder=5,
        )


def _draw_core(ax: Axes, data: FlowerPlotData, style: ChartStyle) -> None:
    lines = data.core_lines()
    for index, text in enumerate(lines):
        # Stack the lines symmetrically about the centre of the disc.
        offset = (0.5 * (len(lines) - 1) - index) * style.core_line_gap
        unit_line = bool(data.unit_label) and index == len(lines) - 1
        ax.text(
            0.0,
            offset,
            text,
            ha="center",
            va="center",
            fontsize=style.value_font_size if unit_line else style.core_font_size,
            fontweight="normal" if unit_line else "bold",
            color="#444444" if unit_line else "black",
            zorder=6,
        )


def create_figure(
    palette: Palette = PALETTES[0],
    data: FlowerPlotData = DEFAULT_DATA,
    style: ChartStyle = DEFAULT_STYLE,
) -> Figure:
    """Create the flower plot without writing it to disk."""

    data.validate()
    n_groups = len(data.groups)
    style.validate(groups=n_groups)

    pitch, width = petal_geometry(style, groups=n_groups)
    reach = style.label_radius + style.label_clearance
    x_limits = resolve_limits(style.x_limits, -reach, reach, snap=False)
    y_limits = resolve_limits(style.y_limits, -reach, reach, snap=False)

    points_per_unit = _points_per_unit(style, x_limits)
    value_radius = 0.5 * (style.inner_radius + style.outer_radius)
    value_font_size = _fitted_font_size(
        style.value_font_size,
        room_points=value_radius * math.radians(width) * points_per_unit,
        characters=max(
            len(line)
            for index in range(n_groups)
            for line in data.petal_lines(index)
        ),
        minimum=style.min_font_size,
    )
    label_font_size = _fitted_font_size(
        style.label_font_size,
        room_points=style.label_radius * math.radians(pitch) * points_per_unit,
        characters=max(len(group) for group in data.groups),
        minimum=style.min_font_size,
    )

    with plt.rc_context(
        {
            "font.family": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
            "font.size": style.value_font_size,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    ):
        figure = plt.figure(figsize=style.figure_size, facecolor="white")
        ax = figure.add_axes((0.0, 0.0, 1.0, 1.0))
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        ax.set_aspect("equal", adjustable="box")
        ax.set_axis_off()
        ax.set_facecolor("white")

        _draw_petals(
            ax,
            data,
            palette,
            style,
            width_degrees=width,
            font_size=value_font_size,
        )
        _draw_labels(ax, data, style, font_size=label_font_size)
        _draw_core(ax, data, style)

    return figure


def main(argv: Sequence[str] | None = None) -> int:
    return run_cli(sys.modules[__name__], argv)


if __name__ == "__main__":
    raise SystemExit(main())
