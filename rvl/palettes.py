"""Shared palette type for every chart template.

Templates used to hard-code a fixed-length colour tuple per chart, which meant
a five-colour palette could only ever draw five series.  ``Palette`` keeps the
curated colour order but serves any number of series: extra series reuse the
cycle at a shifted lightness so a legend stays readable past the cycle length.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from matplotlib.colors import to_hex, to_rgb


def _shift_lightness(color: str, amount: float) -> str:
    """Move a colour toward white (positive) or black (negative)."""

    red, green, blue = to_rgb(color)
    if amount >= 0.0:
        target = 1.0
        weight = min(amount, 1.0)
    else:
        target = 0.0
        weight = min(-amount, 1.0)
    return to_hex(
        tuple(channel + (target - channel) * weight for channel in (red, green, blue))
    )


# Cycle 0 is the curated order; later cycles alternate lighter and darker so a
# 12-series chart drawn from a 5-colour palette still has distinguishable rows.
_CYCLE_SHIFTS: tuple[float, ...] = (0.0, 0.38, -0.30, 0.60, -0.50)


@dataclass(frozen=True, slots=True)
class Palette:
    """A named, ordered colour cycle."""

    name: str
    colors: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name or self.name != self.name.strip().lower():
            raise ValueError("palette name must be lowercase and trimmed")
        if not self.colors:
            raise ValueError(f"palette {self.name!r} must define at least one colour")
        for color in self.colors:
            to_rgb(color)

    def __len__(self) -> int:
        return len(self.colors)

    def color(self, index: int) -> str:
        """Colour for a zero-based series index, extending past the cycle."""

        if index < 0:
            raise IndexError("palette index must be non-negative")
        size = len(self.colors)
        cycle, position = divmod(index, size)
        base = self.colors[position]
        if cycle == 0:
            return base
        shift = _CYCLE_SHIFTS[cycle % len(_CYCLE_SHIFTS)]
        if shift == 0.0:
            shift = 0.38 * (1 if cycle % 2 else -1)
        return _shift_lightness(base, shift)

    def take(self, count: int) -> tuple[str, ...]:
        """First ``count`` colours, extending past the cycle when needed."""

        if count < 0:
            raise ValueError("count must be non-negative")
        return tuple(self.color(index) for index in range(count))

    def reversed(self) -> "Palette":
        return Palette(name=f"{self.name}-reversed", colors=tuple(reversed(self.colors)))


def palette_from_selector(
    palettes: Sequence[Palette], selector: str | int
) -> tuple[int, Palette]:
    """Resolve a one-based palette number or a palette name.

    Returns the one-based index alongside the palette so callers can build
    reproducible output filenames.
    """

    if not palettes:
        raise ValueError("no palettes available")
    text = str(selector).strip()
    if isinstance(selector, int) or text.isdigit():
        index = int(text)
        if 1 <= index <= len(palettes):
            return index, palettes[index - 1]
        raise ValueError(f"palette number must be between 1 and {len(palettes)}")

    normalized = text.lower().replace("_", "-")
    for index, palette in enumerate(palettes, start=1):
        if palette.name == normalized:
            return index, palette
    choices = ", ".join(palette.name for palette in palettes)
    raise ValueError(f"unknown palette {selector!r}; choose from: {choices}")


def selected_palettes(
    palettes: Sequence[Palette], selector: str | int
) -> tuple[tuple[int, Palette], ...]:
    """Resolve a selector that may be ``all``."""

    if str(selector).strip().lower() == "all":
        return tuple(enumerate(palettes, start=1))
    return (palette_from_selector(palettes, selector),)


def relative_luminance(color: str) -> float:
    red, green, blue = to_rgb(color)
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def readable_text_color(background: str, *, cutoff: float = 0.55) -> str:
    """Black or white, whichever stays legible on ``background``."""

    return "#111111" if relative_luminance(background) > cutoff else "#FFFFFF"


def darken_if_pale(color: str, *, cutoff: float = 0.93, factor: float = 0.55) -> str:
    """Darken a near-white fill so it can also be used as label ink."""

    if relative_luminance(color) <= cutoff:
        return color
    red, green, blue = to_rgb(color)
    return to_hex(tuple(channel * factor for channel in (red, green, blue)))
