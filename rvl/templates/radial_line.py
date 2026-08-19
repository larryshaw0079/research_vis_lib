"""Radial line profiles over many features, optionally grouped into arcs.

One closed line per series runs around a circle of features on a shared scale,
so a handful of profiles can be compared feature by feature.  Features may be
grouped into labelled arcs; the arcs open a gap in the ring and carry a label
inside it.

``DEFAULT_DATA`` follows Figure 2 of Niemann et al. (2020), Scientific Reports,
https://doi.org/10.1038/s41598-020-73402-8, CC BY 4.0.  The patient-level table
is not public, so the aggregate z-score means were digitised from the published
figure.  The 18 palettes come from a Xiaohongshu carousel that redraws the same
figure in different colours.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Final, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Circle
from numpy.typing import NDArray

from ..contract import DataKind, Extent, Feature, Geometry, TemplateSpec
from ..palettes import Palette
from ..render import nice_step, resolve_limits, run_cli

SPEC: Final[TemplateSpec] = TemplateSpec(
    template_id="radial-line",
    title="Radial line profile over many features",
    summary=(
        "One closed radial line per series across many features arranged around a "
        "circle, with optional labelled arcs grouping the features into blocks."
    ),
    kinds=(DataKind.MATRIX,),
    geometry=Geometry.POLAR,
    categories=Extent(8, 200),
    series=Extent(1, 8),
    builder="RadialLineData.from_matrix",
    data_contract=(
        "A value for every (feature, series) pair, on a shared scale such as a "
        "z-score. Many features and few series. Features may be grouped into "
        "labelled arcs."
    ),
    good_for=(
        "profile or signature comparison across dozens of standardised features",
        "phenotype, cluster or subtype profiles on a common scale",
        "showing where a few series diverge across a long feature list",
    ),
    avoid_when=(
        "features are on different units and were not standardised",
        "there are fewer than about eight features",
        "exact per-feature values must be read off the chart",
    ),
    long_category_labels=False,
    argument_names=(("categories", "features"),),
    affinities=(
        (Feature.MANY_CATEGORIES, 12.0),
        (Feature.BOUNDED_SCALE, 4.0),
        (Feature.LONG_LABELS, -7.0),
        (Feature.WIDE_DYNAMIC_RANGE, -7.0),
    ),
    default_dpi=200,
    reference="Niemann et al. (2020), Scientific Reports, Fig. 2, CC BY 4.0.",
)


# Colours reconstructed from the solid legend marks in carousel images 1-18.
# Most are canonical Okabe-Ito, Matplotlib/Tableau, or ColorBrewer/D3 colours.
PALETTES: Final[tuple[Palette, ...]] = (
    Palette("okabe-ito", ("#0072B2", "#D55E00", "#E69F00", "#CC79A7")),
    Palette("tab10-primary", ("#1F77B4", "#FF7F0E", "#2CA02C", "#D62728")),
    Palette("set1-primary", ("#E41A1C", "#377EB8", "#4DAF4A", "#984EA3")),
    Palette("set2-primary", ("#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3")),
    Palette("paired-blue-green", ("#A6CEE3", "#1F78B4", "#B2DF8A", "#33A02C")),
    Palette("set3-primary", ("#8DD3C7", "#FFFFB3", "#BEBADA", "#FB8072")),
    Palette("pastel1-primary", ("#FBB4AE", "#B3CDE3", "#CCEBC5", "#DECBE4")),
    Palette("pastel2-primary", ("#B3E2CD", "#FDCDAC", "#CBD5E8", "#F4CAE4")),
    Palette("set1-secondary", ("#FF7F00", "#FFFF33", "#A65628", "#F781BF")),
    Palette("tab10-secondary-a", ("#8C564B", "#E377C2", "#7F7F7F", "#BCBD22")),
    Palette("tab10-secondary-b", ("#9467BD", "#8C564B", "#E377C2", "#7F7F7F")),
    Palette("tab20-mixed", ("#17BECF", "#AEC7E8", "#FFBB78", "#98DF8A")),
    Palette("tab20-light", ("#FF9896", "#C5B0D5", "#C49C94", "#F7B6D2")),
    Palette("category20b-transition", ("#C7C7C7", "#DBDB8D", "#9EDAE5", "#393B79")),
    Palette("category20b-blue", ("#5254A3", "#6B6ECF", "#9C9EDE", "#637939")),
    Palette("category20b-green-brown", ("#8CA252", "#B5CF6B", "#CEDB9C", "#8C6D31")),
    Palette("category20b-gold-red", ("#BD9E39", "#E7BA52", "#E7CB94", "#843C39")),
    Palette("category20b-red-purple", ("#AD494A", "#D6616B", "#E7969C", "#7B4173")),
)


@dataclass(frozen=True, slots=True)
class FeatureArc:
    """An inclusive run of feature indices sharing one outer label."""

    label: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class RadialLineData:
    """A ``features x series`` matrix on one shared scale."""

    features: tuple[str, ...]
    series: tuple[str, ...]
    values: tuple[tuple[float, ...], ...]
    arcs: tuple[FeatureArc, ...] | None = None
    value_label: str = "z-score"
    value_limits: tuple[float, float] | None = None

    @classmethod
    def from_matrix(
        cls,
        *,
        features: Sequence[str],
        series: Sequence[str],
        values: Sequence[Sequence[float | None]],
        arcs: Sequence[FeatureArc] | None = None,
        value_label: str = "z-score",
        value_limits: tuple[float, float] | None = None,
    ) -> "RadialLineData":
        """Build from a ``[feature][series]`` matrix of standardised values.

        ``arcs`` is optional; without it the outer grouping is skipped and the
        features run continuously around the circle.  ``None`` entries become
        NaN and break the line there rather than being plotted as zero.
        """

        built = cls(
            features=tuple(str(name) for name in features),
            series=tuple(str(name) for name in series),
            values=tuple(
                tuple(float("nan") if value is None else float(value) for value in row)
                for row in values
            ),
            arcs=None if arcs is None else tuple(arcs),
            value_label=value_label,
            value_limits=None if value_limits is None else (
                float(value_limits[0]),
                float(value_limits[1]),
            ),
        )
        built.validate()
        return built

    @classmethod
    def from_mapping(
        cls,
        *,
        features: Sequence[str],
        series: Sequence[str],
        values: Mapping[str, Mapping[str, float | None]],
        **kwargs: object,
    ) -> "RadialLineData":
        """Build from nested ``values[feature][series]`` mappings."""

        matrix = [
            [values.get(feature, {}).get(name) for name in series]
            for feature in features
        ]
        return cls.from_matrix(
            features=features, series=series, values=matrix, **kwargs  # type: ignore[arg-type]
        )

    def validate(self) -> None:
        if len(self.features) < 8:
            raise ValueError("need at least eight features")
        if not self.series:
            raise ValueError("need at least one series")
        if len(set(self.features)) != len(self.features):
            raise ValueError("feature labels must be unique")
        if len(set(self.series)) != len(self.series):
            raise ValueError("series labels must be unique")
        if len(self.values) != len(self.features):
            raise ValueError(
                f"values has {len(self.values)} rows but there are "
                f"{len(self.features)} features"
            )
        for feature, row in zip(self.features, self.values, strict=True):
            if len(row) != len(self.series):
                raise ValueError(
                    f"values[{feature!r}] has {len(row)} entries but there are "
                    f"{len(self.series)} series"
                )
        if np.count_nonzero(np.isfinite(self.matrix())) == 0:
            raise ValueError("values must contain at least one finite measurement")
        if self.value_limits is not None and self.value_limits[0] >= self.value_limits[1]:
            raise ValueError("value_limits must be increasing")
        self._validate_arcs()

    def _validate_arcs(self) -> None:
        if self.arcs is None:
            return
        last_end = -1
        for arc in self.arcs:
            if not 0 <= arc.start <= arc.end < len(self.features):
                raise ValueError(
                    f"arc {arc.label!r} spans features {arc.start}-{arc.end}, "
                    f"outside 0-{len(self.features) - 1}"
                )
            if arc.start <= last_end:
                raise ValueError(
                    f"arc {arc.label!r} starts at feature {arc.start}, which "
                    "overlaps the previous arc; arcs must be sorted and disjoint"
                )
            last_end = arc.end

    def matrix(self) -> NDArray[np.float64]:
        """Values as a ``(features, series)`` array."""

        return np.asarray(self.values, dtype=float)

    def value_range(self) -> tuple[float, float]:
        finite = self.matrix()
        finite = finite[np.isfinite(finite)]
        return float(finite.min()), float(finite.max())


_REFERENCE_FEATURES: Final[tuple[str, ...]] = (
    "TINSKAL_frequency",
    "TINSKAL_impairment",
    "TINSKAL_loudness",
    "TLQ_01_bothears",
    "TLQ_01_entirehead",
    "TLQ_01_leftear",
    "TLQ_01_rightear",
    "TLQ_02_hissing",
    "TLQ_02_ringing",
    "TLQ_02_rustling",
    "TLQ_02_whistling",
    "SF8_overallhealth*",
    "SF8_physicalcomp*",
    "SF8_physicalfunct*",
    "SF8_rolephysical*",
    "SES_affectivepain",
    "SES_sensoricpain",
    "SF8_bodilyhealth*",
    "SSKAL_painfrequency",
    "SSKAL_painimpairment",
    "SSKAL_painseverity",
    "BI_abdominalsymptoms",
    "BI_fatigue",
    "BI_heartsymptoms",
    "BI_limbpain",
    "BI_overallcomplaints",
    "ADSL_depression",
    "BSF_anger",
    "BSF_anxdepression",
    "BSF_apathy",
    "BSF_elevatedmood*",
    "BSF_fatigue",
    "BSF_mindset*",
    "ISR_additionalitems",
    "ISR_anxiety",
    "ISR_compulsivesyn",
    "ISR_depression",
    "ISR_eatingdisorder",
    "ISR_somatosyn",
    "ISR_totalpsychiatricsyn",
    "PHQK_depression",
    "PHQK_panicsyn",
    "TQ_auditoryperceptdiff",
    "TQ_cognitivedistress",
    "TQ_distress",
    "TQ_emodistress",
    "TQ_intrusiveness",
    "TQ_psychodistress",
    "TQ_sleepdisturbances",
    "TQ_somatocomplaints",
    "SWOP_optimism*",
    "SWOP_pessimism",
    "SWOP_selfefficacy*",
    "PSQ_demand",
    "PSQ_joy*",
    "PSQ_stress",
    "PSQ_tension",
    "PSQ_worries",
    "ACSA_qualityoflife*",
    "SF8_mentalcomp*",
    "SF8_mentalhealth*",
    "SF8_roleemotional*",
    "SF8_socialfunct*",
    "SF8_vitality*",
)

_REFERENCE_SERIES: Final[tuple[str, ...]] = (
    "PT 1: avoidant\ngroup (n=697)",
    "PT 2: psychosomatic\ngroup (n=173)",
    "PT 3: somatic\ngroup (n=187)",
    "PT 4: distress\ngroup (n=171)",
)

# The published figure breaks its features with a manual hyphen; keeping the
# line breaks here reproduces the reference wrapping exactly.
_REFERENCE_ARCS: Final[tuple[FeatureArc, ...]] = (
    FeatureArc("Tinnitus\ncharac−\nteristics", 0, 10),
    FeatureArc("Physical\nquality\nof life", 11, 14),
    FeatureArc("Experiences of pain", 15, 20),
    FeatureArc("Somatic ex−\npressions", 21, 25),
    FeatureArc("Affective\nsymptoms", 26, 41),
    FeatureArc("Tinnitus−\nrelated\ndistress", 42, 49),
    FeatureArc("Internal\nresources", 50, 52),
    FeatureArc("Perceived\nstress", 53, 57),
    FeatureArc("Mental qual−\nity of life", 58, 63),
)

# Within-phenotype means digitised from the published radial line and
# single-phenotype bar charts, as z scores relative to the patient mean.
_REFERENCE_Z_SCORES: Final[tuple[tuple[float, ...], ...]] = (
    (0.02, 0.08, 0.07, -0.10),
    (-0.34, 0.65, 0.39, 0.28),
    (-0.26, 0.46, 0.30, 0.24),
    (0.03, 0.11, -0.03, -0.11),
    (-0.11, 0.22, 0.07, 0.09),
    (0.07, -0.12, -0.12, -0.08),
    (0.03, -0.21, 0.10, 0.07),
    (0.00, -0.14, 0.01, 0.08),
    (-0.11, 0.11, 0.04, 0.13),
    (-0.15, -0.18, 0.07, -0.11),
    (-0.12, 0.03, -0.09, 0.07),
    (-0.48, 1.17, 0.48, 0.19),
    (-0.44, 1.03, 0.83, -0.31),
    (-0.41, 0.99, 0.71, -0.18),
    (-0.46, 1.11, 0.62, 0.05),
    (-0.50, 1.28, 0.70, -0.09),
    (-0.39, 1.21, 0.48, -0.22),
    (-0.42, 0.92, 0.88, -0.30),
    (-0.27, 0.65, 0.65, -0.23),
    (-0.42, 1.04, 0.72, -0.19),
    (-0.41, 1.00, 0.78, -0.25),
    (-0.35, 0.93, 0.43, -0.08),
    (-0.62, 1.25, 0.64, 0.53),
    (-0.41, 1.26, 0.32, -0.05),
    (-0.50, 1.11, 0.84, -0.09),
    (-0.59, 1.39, 0.71, 0.16),
    (-0.64, 1.49, 0.25, 0.76),
    (-0.46, 1.18, 0.08, 0.55),
    (-0.61, 1.49, 0.11, 0.78),
    (-0.46, 1.35, -0.09, 0.54),
    (-0.53, 0.96, 0.37, 0.71),
    (-0.59, 1.26, 0.37, 0.67),
    (-0.43, 0.84, 0.28, 0.54),
    (-0.55, 1.49, 0.22, 0.43),
    (-0.45, 1.32, 0.13, 0.28),
    (-0.39, 1.06, 0.06, 0.41),
    (-0.62, 1.49, 0.17, 0.78),
    (-0.18, 0.43, 0.14, 0.08),
    (-0.39, 1.06, 0.16, 0.28),
    (-0.58, 1.49, 0.20, 0.51),
    (-0.62, 1.48, 0.27, 0.67),
    (-0.28, 1.00, -0.03, 0.10),
    (-0.37, 0.75, 0.43, 0.22),
    (-0.46, 0.97, 0.34, 0.47),
    (-0.57, 1.12, 0.61, 0.46),
    (-0.55, 1.08, 0.45, 0.57),
    (-0.44, 0.81, 0.51, 0.38),
    (-0.53, 1.08, 0.43, 0.55),
    (-0.38, 0.61, 0.58, 0.22),
    (-0.42, 0.89, 0.73, -0.09),
    (-0.40, 1.06, 0.01, 0.49),
    (-0.32, 0.82, 0.13, 0.25),
    (-0.42, 1.08, 0.13, 0.42),
    (-0.28, 0.61, 0.08, 0.37),
    (-0.51, 1.01, 0.23, 0.76),
    (-0.57, 1.25, 0.26, 0.72),
    (-0.55, 1.09, 0.36, 0.69),
    (-0.55, 1.26, 0.22, 0.69),
    (-0.44, 0.91, 0.22, 0.57),
    (-0.59, 1.26, 0.21, 0.84),
    (-0.57, 1.21, 0.25, 0.77),
    (-0.53, 1.19, 0.37, 0.49),
    (-0.53, 1.22, 0.29, 0.54),
    (-0.49, 1.08, 0.39, 0.44),
)

DEFAULT_DATA: Final[RadialLineData] = RadialLineData.from_matrix(
    features=_REFERENCE_FEATURES,
    series=_REFERENCE_SERIES,
    values=_REFERENCE_Z_SCORES,
    arcs=_REFERENCE_ARCS,
    value_label="z-score",
    value_limits=(-1.5, 1.5),
)


@dataclass(frozen=True, slots=True)
class ChartStyle:
    """Geometry and typography tuned to the 1084 x 1080 reference image.

    Angular slots are derived from the data: every feature owns one slot, each
    arc boundary opens ``arc_gap_slots`` and the ring closes with a wider
    ``wrap_gap_slots``.  The reference 64 features in nine arcs therefore land
    on the original 87-slot ring without any count being pinned here.
    """

    figure_size: tuple[float, float] = (10.84, 10.80)
    x_limits: tuple[float, float] = (-8.38, 8.75)
    y_limits: tuple[float, float] = (-8.76, 8.31)
    start_angle_degrees: float = 16.55
    arc_gap_slots: float = 2.0
    wrap_gap_slots: float = 7.0
    score_radius_span: tuple[float, float] = (2.035, 5.035)
    """Radii of the low and high ends of the value scale."""

    value_step: float | None = 0.5
    """Spacing of the ring gridlines; ``None`` derives a 1/2/5 step."""

    category_radius: float = 1.80
    category_label_radius: float = 1.14
    feature_label_radius: float = 5.43
    arc_padding_slots: float = 1.10
    category_tick_inner: float = 0.10
    category_tick_outer: float = 0.17
    grid_color: str = "#BEBEBE"
    grid_width: float = 1.25
    grid_dash: tuple[float, float] = (12.0, 10.0)
    zero_width: float = 1.85
    line_width: float = 1.75
    marker_size: float = 6.7
    feature_font_size: float = 11.5
    category_font_size: float = 11.2
    tick_font_size: float = 12.5
    legend_font_size: float = 11.8
    legend_anchor: tuple[float, float] = (0.775, 0.988)
    show_value_label: bool = False
    """The reference prints no axis caption; enable it to label the scale."""

    def validate(self, *, categories: int, series: int) -> None:
        if min(self.figure_size) <= 0:
            raise ValueError("figure_size must be positive")
        if categories < 1 or series < 1:
            raise ValueError("categories and series must be positive")
        if min(self.arc_gap_slots, self.wrap_gap_slots) < 0:
            raise ValueError("slot gaps must be non-negative")
        inner, outer = self.score_radius_span
        if not 0.0 < inner < outer:
            raise ValueError("score_radius_span must be increasing and positive")
        if self.category_radius >= inner:
            raise ValueError("the category ring must sit inside the value scale")
        if self.feature_label_radius <= outer:
            raise ValueError("feature labels must sit outside the value scale")
        if self.value_step is not None and self.value_step <= 0:
            raise ValueError("value_step must be positive when given")


DEFAULT_STYLE: Final[ChartStyle] = ChartStyle()


@dataclass(frozen=True, slots=True)
class _Ring:
    """Where every feature sits on the ring, and how the blocks break it up."""

    slots: tuple[float, ...]
    total_slots: float
    blocks: tuple[tuple[int, int, str | None], ...]
    closed: bool

    @property
    def slot_step_degrees(self) -> float:
        return 360.0 / self.total_slots


def ring_layout(data: RadialLineData, style: ChartStyle) -> _Ring:
    """Place every feature on the ring, opening a gap at each arc boundary.

    Without arcs the features run continuously and the profile closes on
    itself; with arcs each block is drawn as its own open polyline.
    """

    count = len(data.features)
    if data.arcs is None:
        return _Ring(
            slots=tuple(float(index) for index in range(count)),
            total_slots=float(count),
            blocks=((0, count - 1, None),),
            closed=True,
        )

    owner: list[str | None] = [None] * count
    for arc in data.arcs:
        for index in range(arc.start, arc.end + 1):
            owner[index] = arc.label

    blocks: list[tuple[int, int, str | None]] = []
    start = 0
    for index in range(1, count + 1):
        if index == count or owner[index] is not owner[index - 1]:
            blocks.append((start, index - 1, owner[start]))
            start = index

    slots: list[float] = []
    cursor = 0.0
    for block_index, (first, last, _) in enumerate(blocks):
        if block_index:
            cursor += style.arc_gap_slots
        for offset in range(last - first + 1):
            slots.append(cursor + offset)
        cursor += float(last - first + 1)
    return _Ring(
        slots=tuple(slots),
        total_slots=cursor + style.wrap_gap_slots,
        blocks=tuple(blocks),
        closed=False,
    )


def _slot_angle_degrees(slot: float, ring: _Ring, style: ChartStyle) -> float:
    return style.start_angle_degrees + slot * ring.slot_step_degrees


def _xy(
    angle_degrees: float | NDArray[np.float64],
    radius: float | NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert top-origin clockwise polar coordinates to Cartesian coordinates."""

    angle = np.radians(angle_degrees)
    radial = np.asarray(radius, dtype=float)
    return radial * np.sin(angle), radial * np.cos(angle)


def scale_ticks(limits: tuple[float, float], style: ChartStyle) -> NDArray[np.float64]:
    """Gridline values across ``limits``, anchored on zero."""

    low, high = limits
    step = style.value_step or nice_step(high - low)
    if (high - low) / step > 12.0:
        step = nice_step(high - low)
    first = math.ceil(low / step - 1e-9)
    last = math.floor(high / step + 1e-9)
    return step * np.arange(first, last + 1, dtype=float)


def value_radius(
    values: NDArray[np.float64] | Sequence[float] | float,
    limits: tuple[float, float],
    style: ChartStyle,
) -> NDArray[np.float64]:
    """Map values onto the radial band that carries the scale."""

    low, high = limits
    inner, outer = style.score_radius_span
    span = high - low
    fraction = (np.asarray(values, dtype=float) - low) / (span if span else 1.0)
    return inner + fraction * (outer - inner)


def _tick_text(value: float, step: float) -> str:
    decimals = max(0, -math.floor(math.log10(step) + 1e-9))
    if abs(value) < 0.5 * step:
        return "0"
    sign = "+" if value > 0 else "−"
    return f"{sign}{abs(value):.{decimals}f}"


def _label_scale(ring: _Ring, style: ChartStyle) -> float:
    """Shrink factor keeping feature labels apart as the ring gets denser."""

    width, height = style.figure_size
    x_span = style.x_limits[1] - style.x_limits[0]
    y_span = style.y_limits[1] - style.y_limits[0]
    points_per_unit = min(72.0 * width / x_span, 72.0 * height / y_span)
    pitch = np.radians(ring.slot_step_degrees) * style.feature_label_radius
    allowed = 0.75 * pitch * points_per_unit
    return min(1.0, allowed / style.feature_font_size)


def _draw_arc(
    ax: Axes,
    radius: float,
    start_degrees: float,
    end_degrees: float,
    *,
    color: str,
    linewidth: float,
    zorder: float,
) -> None:
    ax.add_patch(
        Arc(
            (0.0, 0.0),
            2.0 * radius,
            2.0 * radius,
            theta1=90.0 - end_degrees,
            theta2=90.0 - start_degrees,
            color=color,
            linewidth=linewidth,
            capstyle="butt",
            zorder=zorder,
        )
    )


def _block_bounds(
    block: tuple[int, int, str | None], ring: _Ring, style: ChartStyle
) -> tuple[float, float]:
    first, last, _ = block
    padding = style.arc_padding_slots * ring.slot_step_degrees
    return (
        _slot_angle_degrees(ring.slots[first], ring, style) - padding,
        _slot_angle_degrees(ring.slots[last], ring, style) + padding,
    )


def _draw_score_grid(
    ax: Axes,
    ring: _Ring,
    limits: tuple[float, float],
    ticks: NDArray[np.float64],
    style: ChartStyle,
) -> None:
    for radius in value_radius(ticks, limits, style):
        ax.add_patch(
            Circle(
                (0.0, 0.0),
                float(radius),
                facecolor="none",
                edgecolor=style.grid_color,
                linewidth=style.grid_width,
                linestyle=(0.0, style.grid_dash),
                zorder=1,
            )
        )

    low, high = limits
    baseline = float(value_radius(0.0 if low <= 0.0 <= high else low, limits, style))
    if ring.closed:
        ax.add_patch(
            Circle(
                (0.0, 0.0),
                baseline,
                facecolor="none",
                edgecolor="black",
                linewidth=style.zero_width,
                zorder=3,
            )
        )
        return

    for block in ring.blocks:
        start, end = _block_bounds(block, ring, style)
        _draw_arc(
            ax, baseline, start, end, color="black", linewidth=style.zero_width, zorder=3
        )
        if block[2] is None:
            continue
        _draw_arc(
            ax,
            style.category_radius,
            start,
            end,
            color="black",
            linewidth=style.zero_width,
            zorder=3,
        )
        for boundary in (start, end):
            radii = np.array(
                [
                    style.category_radius - style.category_tick_inner,
                    style.category_radius + style.category_tick_outer,
                ]
            )
            x, y = _xy(boundary, radii)
            ax.plot(
                x,
                y,
                color="black",
                linewidth=style.zero_width,
                solid_capstyle="butt",
                zorder=3,
            )


def _draw_scale_labels(
    ax: Axes,
    data: RadialLineData,
    limits: tuple[float, float],
    ticks: NDArray[np.float64],
    style: ChartStyle,
) -> None:
    step = float(ticks[1] - ticks[0]) if len(ticks) > 1 else 1.0
    radii = value_radius(ticks, limits, style)
    for tick, radius in zip(ticks, radii, strict=True):
        ax.text(
            0.0,
            float(radius),
            _tick_text(float(tick), step),
            ha="center",
            va="center",
            fontsize=style.tick_font_size,
            color="black",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.3},
            zorder=8,
        )
    if style.show_value_label and data.value_label:
        ax.text(
            0.0,
            float(radii.max()) + 0.55 * style.tick_font_size / 12.0,
            data.value_label,
            ha="center",
            va="bottom",
            fontsize=style.tick_font_size,
            color="black",
            zorder=8,
        )


def _draw_category_labels(ax: Axes, ring: _Ring, style: ChartStyle) -> None:
    for block in ring.blocks:
        label = block[2]
        if label is None:
            continue
        first, last, _ = block
        midpoint = 0.5 * (
            _slot_angle_degrees(ring.slots[first], ring, style)
            + _slot_angle_degrees(ring.slots[last], ring, style)
        )
        x, y = _xy(midpoint, style.category_label_radius)
        ax.text(
            float(x),
            float(y),
            label,
            ha="center",
            va="center",
            multialignment="center",
            linespacing=0.95,
            fontsize=style.category_font_size,
            color="#262626",
            zorder=7,
        )


def _draw_feature_labels(
    ax: Axes, data: RadialLineData, ring: _Ring, style: ChartStyle
) -> None:
    font_size = style.feature_font_size * _label_scale(ring, style)
    for label, slot in zip(data.features, ring.slots, strict=True):
        angle = _slot_angle_degrees(slot, ring, style) % 360.0
        x, y = _xy(angle, style.feature_label_radius)
        rotation = 90.0 - angle
        if rotation < -90.0:
            rotation += 180.0
        elif rotation > 90.0:
            rotation -= 180.0
        right_half = 0.0 <= angle < 180.0
        ax.text(
            float(x),
            float(y),
            label,
            ha="left" if right_half else "right",
            va="center",
            rotation=rotation,
            rotation_mode="anchor",
            fontsize=font_size,
            color="#262626",
            zorder=7,
        )


def _draw_profiles(
    ax: Axes,
    data: RadialLineData,
    ring: _Ring,
    palette: Palette,
    limits: tuple[float, float],
    style: ChartStyle,
) -> None:
    matrix = data.matrix()
    colors = palette.take(len(data.series))
    scale = _label_scale(ring, style)
    for series_index in range(len(data.series)):
        color = colors[series_index]
        column = matrix[:, series_index]
        for first, last, _ in ring.blocks:
            indices = list(range(first, last + 1))
            if ring.closed and len(indices) > 1:
                indices.append(first)
            radii = value_radius(column[indices], limits, style)
            angles = np.array(
                [_slot_angle_degrees(ring.slots[index], ring, style) for index in indices],
                dtype=float,
            )
            if ring.closed and len(indices) > 1:
                angles[-1] -= 360.0
            x, y = _xy(angles, radii)
            ax.plot(
                x,
                y,
                color=color,
                linewidth=style.line_width * scale,
                marker="o",
                markersize=style.marker_size * scale,
                markerfacecolor=color,
                markeredgecolor=color,
                markeredgewidth=0.0,
                solid_capstyle="round",
                solid_joinstyle="round",
                zorder=4 + series_index * 0.1,
            )


def _draw_legend(
    ax: Axes, data: RadialLineData, palette: Palette, style: ChartStyle
) -> None:
    colors = palette.take(len(data.series))
    handles = [
        Line2D(
            [0],
            [0],
            color=color,
            linewidth=style.line_width,
            marker="o",
            markersize=style.marker_size,
            markerfacecolor=color,
            markeredgewidth=0.0,
        )
        for color in colors
    ]
    ax.legend(
        handles,
        list(data.series),
        loc="upper left",
        bbox_to_anchor=style.legend_anchor,
        borderaxespad=0.0,
        borderpad=0.0,
        frameon=False,
        fontsize=style.legend_font_size,
        handlelength=2.8,
        handletextpad=0.8,
        labelspacing=0.75,
    )


def create_figure(
    palette: Palette = PALETTES[0],
    data: RadialLineData = DEFAULT_DATA,
    style: ChartStyle = DEFAULT_STYLE,
) -> Figure:
    """Create the radial line figure without writing it to disk."""

    data.validate()
    style.validate(categories=len(data.features), series=len(data.series))
    limits = resolve_limits(data.value_limits, *data.value_range())
    ring = ring_layout(data, style)
    ticks = scale_ticks(limits, style)

    with plt.rc_context(
        {
            "font.family": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
            "font.size": style.feature_font_size,
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

        _draw_score_grid(ax, ring, limits, ticks, style)
        _draw_profiles(ax, data, ring, palette, limits, style)
        _draw_category_labels(ax, ring, style)
        _draw_feature_labels(ax, data, ring, style)
        _draw_scale_labels(ax, data, limits, ticks, style)
        _draw_legend(ax, data, palette, style)

    return figure


def main(argv: Sequence[str] | None = None) -> int:
    return run_cli(sys.modules[__name__], argv)


if __name__ == "__main__":
    raise SystemExit(main())
