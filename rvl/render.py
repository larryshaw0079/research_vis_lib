"""Shared rendering and CLI plumbing for every chart template.

Each template only has to expose ``SPEC``, ``PALETTES``, ``DEFAULT_DATA``,
``DEFAULT_STYLE`` and ``create_figure``; the export loop, filename scheme and
argument parser live here so all templates behave identically.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol, Sequence, runtime_checkable

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .contract import SUPPORTED_FORMATS, TemplateSpec
from .palettes import Palette, selected_palettes


def nice_step(span: float) -> float:
    """A 1/2/5-times-power-of-ten step that divides ``span`` into a few parts."""

    if not math.isfinite(span) or span <= 0.0:
        return 1.0
    raw = span / 5.0
    magnitude = 10.0 ** math.floor(math.log10(raw))
    for multiple in (1.0, 2.0, 2.5, 5.0, 10.0):
        if raw <= multiple * magnitude:
            return multiple * magnitude
    return 10.0 * magnitude


def padded_range(
    low: float,
    high: float,
    *,
    pad: float = 0.06,
    include_zero: bool = True,
    snap: bool = True,
) -> tuple[float, float]:
    """A readable axis range covering ``[low, high]``.

    ``include_zero`` extends the range to the origin when the data does not
    straddle it, which keeps bar lengths honest. ``snap`` rounds the bounds out
    to a 1/2/5 step so tick labels stay tidy.
    """

    if not (math.isfinite(low) and math.isfinite(high)):
        return (0.0, 1.0)
    if low > high:
        low, high = high, low
    if include_zero:
        low = min(low, 0.0)
        high = max(high, 0.0)
    span = high - low
    if span <= 0.0:
        span = abs(high) if high else 1.0
        low -= 0.5 * span
        high += 0.5 * span
        span = high - low
    low -= pad * span if low != 0.0 else 0.0
    high += pad * span
    if snap:
        step = nice_step(high - low)
        low = math.floor(low / step) * step
        high = math.ceil(high / step) * step
    return (float(low), float(high))


def resolve_limits(
    pinned: tuple[float, float] | None,
    low: float,
    high: float,
    *,
    min_fill: float = 0.35,
    **padding: Any,
) -> tuple[float, float]:
    """Honour ``pinned`` limits only while they suit the data.

    Reference figures keep their hand-tuned axis bounds because the reference
    data fills them. Two cases give the pin up:

    - the data would be **clipped**, so bars or points would silently vanish;
    - the data would fill less than ``min_fill`` of the pinned span, so it would
      render as a sliver. Counts of a few dozen on an axis tuned for several
      hundred are technically inside the range and visually useless.

    ``min_fill`` is deliberately low enough that every reference figure keeps its
    own bounds; raise it per call for an axis that needs to be filled more.
    """

    if pinned is not None and pinned[0] <= low and high <= pinned[1]:
        span = float(pinned[1]) - float(pinned[0])
        if span <= 0.0 or (high - low) >= min_fill * span:
            return (float(pinned[0]), float(pinned[1]))
    return padded_range(low, high, **padding)


@runtime_checkable
class TemplateModule(Protocol):
    """The surface every template module in ``rvl.templates`` implements."""

    SPEC: TemplateSpec
    PALETTES: tuple[Palette, ...]
    DEFAULT_DATA: Any
    DEFAULT_STYLE: Any

    def create_figure(self, *args: Any, **kwargs: Any) -> Figure: ...


def normalize_formats(formats: Sequence[str]) -> tuple[str, ...]:
    """Lower-case, strip dots, and reject anything matplotlib cannot write here."""

    normalized = tuple(str(name).lower().lstrip(".") for name in formats)
    if not normalized:
        raise ValueError("at least one output format is required")
    unsupported = [name for name in normalized if name not in SUPPORTED_FORMATS]
    if unsupported:
        raise ValueError(
            f"unsupported output format(s): {', '.join(sorted(set(unsupported)))}; "
            f"choose from: {' '.join(SUPPORTED_FORMATS)}"
        )
    seen: list[str] = []
    for name in normalized:
        if name not in seen:
            seen.append(name)
    return tuple(seen)


def save_figure(
    figure: Figure,
    output_dir: Path,
    stem: str,
    *,
    formats: Sequence[str] = ("png",),
    dpi: int = 200,
    close: bool = True,
) -> list[Path]:
    """Write one figure to every requested format and return the paths."""

    resolved_formats = normalize_formats(formats)
    if dpi <= 0:
        raise ValueError("dpi must be positive")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    try:
        for name in resolved_formats:
            path = output_dir / f"{stem}.{name}"
            figure.savefig(
                path,
                format=name,
                dpi=dpi,
                facecolor="white",
                edgecolor="none",
            )
            paths.append(path)
    finally:
        if close:
            plt.close(figure)
    return paths


def palette_stem(template_id: str, index: int, palette: Palette) -> str:
    base = template_id.replace("-", "_")
    return f"{base}_palette_{index:02d}_{palette.name.replace('-', '_')}"


def render_palette(
    module: ModuleType,
    index: int,
    palette: Palette,
    output_dir: Path,
    *,
    formats: Sequence[str] = ("png",),
    dpi: int | None = None,
    data: Any | None = None,
    style: Any | None = None,
) -> list[Path]:
    """Render one palette of one template and return the written paths."""

    spec: TemplateSpec = module.SPEC
    figure = module.create_figure(
        palette=palette,
        data=module.DEFAULT_DATA if data is None else data,
        style=module.DEFAULT_STYLE if style is None else style,
    )
    return save_figure(
        figure,
        output_dir,
        palette_stem(spec.template_id, index, palette),
        formats=formats,
        dpi=spec.default_dpi if dpi is None else dpi,
    )


def build_argument_parser(spec: TemplateSpec) -> argparse.ArgumentParser:
    """Standard parser shared by every template CLI."""

    parser = argparse.ArgumentParser(
        prog=spec.template_id,
        description=f"Render the {spec.title.lower()}.",
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
        help=f"one or more of: {' '.join(SUPPORTED_FORMATS)} (default: png)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=spec.default_dpi,
        help=f"raster DPI (default: {spec.default_dpi})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output") / spec.template_id.replace("-", "_"),
        help="destination directory",
    )
    parser.add_argument(
        "--list-palettes",
        action="store_true",
        help="list palette names and colours, then exit",
    )
    return parser


def run_cli(module: ModuleType, argv: Sequence[str] | None = None) -> int:
    """Entry point shared by every template's ``main``."""

    spec: TemplateSpec = module.SPEC
    palettes: tuple[Palette, ...] = module.PALETTES
    args = build_argument_parser(spec).parse_args(argv)

    if args.list_palettes:
        width = max(len(palette.name) for palette in palettes)
        for index, palette in enumerate(palettes, start=1):
            print(f"{index:2d}  {palette.name:{width}s}  {' '.join(palette.colors)}")
        return 0

    try:
        written = [
            path
            for index, palette in selected_palettes(palettes, args.palette)
            for path in render_palette(
                module,
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
