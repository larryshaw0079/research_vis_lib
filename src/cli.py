"""Command-line interface for the chart renderers."""

from __future__ import annotations

import sys
from collections.abc import Sequence

from . import grouped_lm_marginal, grouped_ring_bar, radar_bubble, radial_line

COMMANDS = {
    "grouped-lm-marginal": grouped_lm_marginal.main,
    "grouped-ring-bar": grouped_ring_bar.main,
    "radar-bubble": radar_bubble.main,
    "radial-line": radial_line.main,
}

HELP = """\
usage: main.py <chart> [options]

Reproducible scientific visualization examples.

charts:
  grouped-lm-marginal  grouped linear-fit scatter with marginal histograms
  grouped-ring-bar     grouped annular bar chart of forecasting MSE
  radar-bubble         31-region annular radar-bubble chart
  radial-line          64-feature radial phenotype line chart

Run `main.py <chart> --help` for chart-specific options.
"""


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print(HELP, end="")
        return 0

    command = argv[0]
    handler = COMMANDS.get(command)
    if handler is None:
        print(HELP, end="", file=sys.stderr)
        print(f"error: unknown chart {command!r}", file=sys.stderr)
        return 2
    return handler(argv[1:])
