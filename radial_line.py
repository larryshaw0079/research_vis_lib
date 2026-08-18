"""Reproduce the 64-feature radial phenotype line chart.

The geometry follows Figure 2 from Niemann et al. (2020),
https://doi.org/10.1038/s41598-020-73402-8.  The article and figure are
available under CC BY 4.0.  The aggregate z-score means below were digitised
from the published figure because the patient-level dataset is not public.

The Xiaohongshu reference contains 18 copies of the same chart with different
four-colour palettes.  All 18 palettes are exposed through the command-line
interface and the library API.
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
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Circle
from numpy.typing import NDArray


# Preserve editable text when callers save a returned figure as SVG or PDF.
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["pdf.fonttype"] = 42


PHENOTYPES: Final[tuple[str, ...]] = ("pt1", "pt2", "pt3", "pt4")

PHENOTYPE_LABELS: Final[Mapping[str, str]] = {
    "pt1": "PT 1: avoidant\ngroup (n=697)",
    "pt2": "PT 2: psychosomatic\ngroup (n=173)",
    "pt3": "PT 3: somatic\ngroup (n=187)",
    "pt4": "PT 4: distress\ngroup (n=171)",
}


@dataclass(frozen=True, slots=True)
class Palette:
    """Four phenotype colours in PT1, PT2, PT3, PT4 order."""

    name: str
    colors: tuple[str, str, str, str]

    def for_phenotype(self, phenotype: str) -> str:
        try:
            return self.colors[PHENOTYPES.index(phenotype)]
        except ValueError as exc:
            raise KeyError(f"unsupported phenotype: {phenotype}") from exc


# Colours are reconstructed from the solid legend marks in carousel images 1-18.
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
class FeatureGroup:
    """A semantic feature group and its occupied angular slots."""

    name: str
    display_label: str
    features: tuple[str, ...]
    slots: tuple[int, ...]

    def validate(self) -> None:
        if not self.features:
            raise ValueError(f"feature group {self.name!r} is empty")
        if len(self.features) != len(self.slots):
            raise ValueError(
                f"feature group {self.name!r} has {len(self.features)} labels "
                f"but {len(self.slots)} slots"
            )
        if any(second <= first for first, second in zip(self.slots, self.slots[1:])):
            raise ValueError(f"feature group {self.name!r} slots must increase")


FEATURE_GROUPS: Final[tuple[FeatureGroup, ...]] = (
    FeatureGroup(
        "tinnitus_characteristics",
        "Tinnitus\ncharac−\nteristics",
        (
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
        ),
        tuple(range(0, 11)),
    ),
    FeatureGroup(
        "physical_quality_of_life",
        "Physical\nquality\nof life",
        (
            "SF8_overallhealth*",
            "SF8_physicalcomp*",
            "SF8_physicalfunct*",
            "SF8_rolephysical*",
        ),
        tuple(range(13, 17)),
    ),
    FeatureGroup(
        "experiences_of_pain",
        "Experiences of pain",
        (
            "SES_affectivepain",
            "SES_sensoricpain",
            "SF8_bodilyhealth*",
            "SSKAL_painfrequency",
            "SSKAL_painimpairment",
            "SSKAL_painseverity",
        ),
        tuple(range(19, 25)),
    ),
    FeatureGroup(
        "somatic_expressions",
        "Somatic ex−\npressions",
        (
            "BI_abdominalsymptoms",
            "BI_fatigue",
            "BI_heartsymptoms",
            "BI_limbpain",
            "BI_overallcomplaints",
        ),
        tuple(range(27, 32)),
    ),
    FeatureGroup(
        "affective_symptoms",
        "Affective\nsymptoms",
        (
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
        ),
        tuple(range(34, 50)),
    ),
    FeatureGroup(
        "tinnitus_related_distress",
        "Tinnitus−\nrelated\ndistress",
        (
            "TQ_auditoryperceptdiff",
            "TQ_cognitivedistress",
            "TQ_distress",
            "TQ_emodistress",
            "TQ_intrusiveness",
            "TQ_psychodistress",
            "TQ_sleepdisturbances",
            "TQ_somatocomplaints",
        ),
        tuple(range(52, 60)),
    ),
    FeatureGroup(
        "internal_resources",
        "Internal\nresources",
        (
            "SWOP_optimism*",
            "SWOP_pessimism",
            "SWOP_selfefficacy*",
        ),
        tuple(range(62, 65)),
    ),
    FeatureGroup(
        "perceived_stress",
        "Perceived\nstress",
        (
            "PSQ_demand",
            "PSQ_joy*",
            "PSQ_stress",
            "PSQ_tension",
            "PSQ_worries",
        ),
        tuple(range(67, 72)),
    ),
    FeatureGroup(
        "mental_quality_of_life",
        "Mental qual−\nity of life",
        (
            "ACSA_qualityoflife*",
            "SF8_mentalcomp*",
            "SF8_mentalhealth*",
            "SF8_roleemotional*",
            "SF8_socialfunct*",
            "SF8_vitality*",
        ),
        tuple(range(74, 80)),
    ),
)

FEATURES: Final[tuple[str, ...]] = tuple(
    feature for group in FEATURE_GROUPS for feature in group.features
)
FEATURE_SLOTS: Final[tuple[int, ...]] = tuple(
    slot for group in FEATURE_GROUPS for slot in group.slots
)


@dataclass(frozen=True, slots=True)
class RadialLineData:
    """Z-score means aligned with :data:`FEATURES`."""

    features: tuple[str, ...]
    z_scores: Mapping[str, NDArray[np.float64]]

    def validate(self) -> None:
        if self.features != FEATURES:
            raise ValueError("features must match the canonical 64-feature order")
        if set(self.z_scores) != set(PHENOTYPES):
            raise ValueError(f"z_scores must contain exactly {PHENOTYPES}")
        for phenotype, values in self.z_scores.items():
            values = np.asarray(values, dtype=float)
            if values.shape != (len(self.features),):
                raise ValueError(
                    f"z_scores[{phenotype!r}] has shape {values.shape}; "
                    f"expected ({len(self.features)},)"
                )
            if not np.all(np.isfinite(values)):
                raise ValueError(f"z_scores[{phenotype!r}] must be finite")


# Aggregate within-phenotype means digitised from the published radial line and
# single-phenotype bar charts. Values are z scores relative to the patient mean.
DEFAULT_DATA: Final[RadialLineData] = RadialLineData(
    features=FEATURES,
    z_scores={
        "pt1": np.array(
            [
                0.02, -0.34, -0.26, 0.03, -0.11, 0.07, 0.03, 0.00,
                -0.11, -0.15, -0.12,
                -0.48, -0.44, -0.41, -0.46,
                -0.50, -0.39, -0.42, -0.27, -0.42, -0.41,
                -0.35, -0.62, -0.41, -0.50, -0.59,
                -0.64, -0.46, -0.61, -0.46, -0.53, -0.59, -0.43,
                -0.55, -0.45, -0.39, -0.62, -0.18, -0.39, -0.58,
                -0.62, -0.28,
                -0.37, -0.46, -0.57, -0.55, -0.44, -0.53, -0.38, -0.42,
                -0.40, -0.32, -0.42,
                -0.28, -0.51, -0.57, -0.55, -0.55,
                -0.44, -0.59, -0.57, -0.53, -0.53, -0.49,
            ],
            dtype=float,
        ),
        "pt2": np.array(
            [
                0.08, 0.65, 0.46, 0.11, 0.22, -0.12, -0.21, -0.14,
                0.11, -0.18, 0.03,
                1.17, 1.03, 0.99, 1.11,
                1.28, 1.21, 0.92, 0.65, 1.04, 1.00,
                0.93, 1.25, 1.26, 1.11, 1.39,
                1.49, 1.18, 1.49, 1.35, 0.96, 1.26, 0.84, 1.49,
                1.32, 1.06, 1.49, 0.43, 1.06, 1.49, 1.48, 1.00,
                0.75, 0.97, 1.12, 1.08, 0.81, 1.08, 0.61, 0.89,
                1.06, 0.82, 1.08,
                0.61, 1.01, 1.25, 1.09, 1.26,
                0.91, 1.26, 1.21, 1.19, 1.22, 1.08,
            ],
            dtype=float,
        ),
        "pt3": np.array(
            [
                0.07, 0.39, 0.30, -0.03, 0.07, -0.12, 0.10, 0.01,
                0.04, 0.07, -0.09,
                0.48, 0.83, 0.71, 0.62,
                0.70, 0.48, 0.88, 0.65, 0.72, 0.78,
                0.43, 0.64, 0.32, 0.84, 0.71,
                0.25, 0.08, 0.11, -0.09, 0.37, 0.37, 0.28, 0.22,
                0.13, 0.06, 0.17, 0.14, 0.16, 0.20, 0.27, -0.03,
                0.43, 0.34, 0.61, 0.45, 0.51, 0.43, 0.58, 0.73,
                0.01, 0.13, 0.13,
                0.08, 0.23, 0.26, 0.36, 0.22,
                0.22, 0.21, 0.25, 0.37, 0.29, 0.39,
            ],
            dtype=float,
        ),
        "pt4": np.array(
            [
                -0.10, 0.28, 0.24, -0.11, 0.09, -0.08, 0.07, 0.08,
                0.13, -0.11, 0.07,
                0.19, -0.31, -0.18, 0.05,
                -0.09, -0.22, -0.30, -0.23, -0.19, -0.25,
                -0.08, 0.53, -0.05, -0.09, 0.16,
                0.76, 0.55, 0.78, 0.54, 0.71, 0.67, 0.54, 0.43,
                0.28, 0.41, 0.78, 0.08, 0.28, 0.51, 0.67, 0.10,
                0.22, 0.47, 0.46, 0.57, 0.38, 0.55, 0.22, -0.09,
                0.49, 0.25, 0.42,
                0.37, 0.76, 0.72, 0.69, 0.69,
                0.57, 0.84, 0.77, 0.49, 0.54, 0.44,
            ],
            dtype=float,
        ),
    },
)


@dataclass(frozen=True, slots=True)
class ChartStyle:
    """Geometry and typography tuned to the 1084 × 1080 reference image."""

    figure_size: tuple[float, float] = (10.84, 10.80)
    x_limits: tuple[float, float] = (-8.38, 8.75)
    y_limits: tuple[float, float] = (-8.76, 8.31)
    total_slots: int = 87
    rotation_degrees: float = 16.55
    zero_radius: float = 3.535
    score_radius_scale: float = 1.0
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

    @property
    def slot_step_degrees(self) -> float:
        return 360.0 / self.total_slots

    def validate(self) -> None:
        if self.total_slots <= max(FEATURE_SLOTS):
            raise ValueError("total_slots must exceed every feature slot")
        if self.score_radius_scale <= 0:
            raise ValueError("score_radius_scale must be positive")
        if self.category_radius >= self.zero_radius - 1.5 * self.score_radius_scale:
            raise ValueError("category ring overlaps the -1.5 score ring")


DEFAULT_STYLE: Final[ChartStyle] = ChartStyle()


def _slot_angle_degrees(slot: int, style: ChartStyle) -> float:
    return style.rotation_degrees + slot * style.slot_step_degrees


def _xy(
    angle_degrees: float | NDArray[np.float64],
    radius: float | NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert top-origin clockwise polar coordinates to Cartesian coordinates."""

    angle = np.radians(angle_degrees)
    radial = np.asarray(radius, dtype=float)
    return radial * np.sin(angle), radial * np.cos(angle)


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


def _draw_score_grid(ax: Axes, style: ChartStyle) -> None:
    for tick in np.arange(-1.5, 1.51, 0.5):
        radius = style.zero_radius + tick * style.score_radius_scale
        circle = Circle(
            (0.0, 0.0),
            radius,
            facecolor="none",
            edgecolor=style.grid_color,
            linewidth=style.grid_width,
            linestyle=(0.0, style.grid_dash),
            zorder=1,
        )
        ax.add_patch(circle)

    padding = style.arc_padding_slots * style.slot_step_degrees
    for group in FEATURE_GROUPS:
        start = _slot_angle_degrees(group.slots[0], style) - padding
        end = _slot_angle_degrees(group.slots[-1], style) + padding
        _draw_arc(
            ax,
            style.zero_radius,
            start,
            end,
            color="black",
            linewidth=style.zero_width,
            zorder=3,
        )
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


def _draw_scale_labels(ax: Axes, style: ChartStyle) -> None:
    for tick in np.arange(-1.5, 1.51, 0.5):
        radius = style.zero_radius + tick * style.score_radius_scale
        if tick > 0:
            label = f"+{tick:.1f}"
        elif tick < 0:
            label = f"−{abs(tick):.1f}"
        else:
            label = "0"
        ax.text(
            0.0,
            radius,
            label,
            ha="center",
            va="center",
            fontsize=style.tick_font_size,
            color="black",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.3},
            zorder=8,
        )


def _draw_category_labels(ax: Axes, style: ChartStyle) -> None:
    for group in FEATURE_GROUPS:
        midpoint = (
            _slot_angle_degrees(group.slots[0], style)
            + _slot_angle_degrees(group.slots[-1], style)
        ) / 2.0
        x, y = _xy(midpoint, style.category_label_radius)
        ax.text(
            float(x),
            float(y),
            group.display_label,
            ha="center",
            va="center",
            multialignment="center",
            linespacing=0.95,
            fontsize=style.category_font_size,
            color="#262626",
            zorder=7,
        )


def _draw_feature_labels(ax: Axes, style: ChartStyle) -> None:
    for group in FEATURE_GROUPS:
        for label, slot in zip(group.features, group.slots, strict=True):
            angle = _slot_angle_degrees(slot, style) % 360.0
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
                fontsize=style.feature_font_size,
            color="#262626",
                zorder=7,
            )


def _draw_profiles(
    ax: Axes,
    data: RadialLineData,
    palette: Palette,
    style: ChartStyle,
) -> None:
    feature_index = {feature: index for index, feature in enumerate(data.features)}
    for phenotype_index, phenotype in enumerate(PHENOTYPES):
        color = palette.for_phenotype(phenotype)
        values = np.asarray(data.z_scores[phenotype], dtype=float)
        for group in FEATURE_GROUPS:
            indices = np.array([feature_index[feature] for feature in group.features])
            group_values = values[indices]
            radii = style.zero_radius + group_values * style.score_radius_scale
            angles = np.array(
                [_slot_angle_degrees(slot, style) for slot in group.slots],
                dtype=float,
            )
            x, y = _xy(angles, radii)
            ax.plot(
                x,
                y,
                color=color,
                linewidth=style.line_width,
                marker="o",
                markersize=style.marker_size,
                markerfacecolor=color,
                markeredgecolor=color,
                markeredgewidth=0.0,
                solid_capstyle="round",
                solid_joinstyle="round",
                zorder=4 + phenotype_index * 0.1,
            )


def _draw_legend(ax: Axes, palette: Palette, style: ChartStyle) -> None:
    handles = [
        Line2D(
            [0],
            [0],
            color=palette.for_phenotype(phenotype),
            linewidth=style.line_width,
            marker="o",
            markersize=style.marker_size,
            markerfacecolor=palette.for_phenotype(phenotype),
            markeredgewidth=0.0,
        )
        for phenotype in PHENOTYPES
    ]
    ax.legend(
        handles,
        [PHENOTYPE_LABELS[phenotype] for phenotype in PHENOTYPES],
        loc="upper left",
        bbox_to_anchor=(0.775, 0.988),
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
    style.validate()
    for group in FEATURE_GROUPS:
        group.validate()

    with plt.rc_context(
        {
            "font.family": [
                "Times New Roman",
                "Times",
                "DejaVu Serif",
                "serif",
            ],
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

        _draw_score_grid(ax, style)
        _draw_profiles(ax, data, palette, style)
        _draw_category_labels(ax, style)
        _draw_feature_labels(ax, style)
        _draw_scale_labels(ax, style)
        _draw_legend(ax, palette, style)

    return figure


def palette_from_selector(selector: str) -> tuple[int, Palette]:
    """Resolve a one-based palette number or a palette name."""

    normalized = selector.strip().lower().replace("_", "-")
    if normalized.isdigit():
        index = int(normalized)
        if 1 <= index <= len(PALETTES):
            return index, PALETTES[index - 1]
        raise ValueError(f"palette number must be between 1 and {len(PALETTES)}")

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
    dpi: int = 100,
    data: RadialLineData = DEFAULT_DATA,
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
    stem = f"radial_line_palette_{index:02d}_{palette.name.replace('-', '_')}"
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
        description="Render the 64-feature radial phenotype line chart."
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
        default=100,
        help="raster DPI; 100 reproduces the 1084×1080 reference (default: 100)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/radial_line"),
        help="destination directory (default: output/radial_line)",
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
            print(f"{index:2d}  {palette.name:29s}  {' '.join(palette.colors)}")
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
