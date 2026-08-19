"""Command-line interface for the chart renderers."""

from __future__ import annotations

import sys
from collections.abc import Sequence

from . import (
    bar_line,
    cartoon_stacked_bar,
    concordance_upset,
    flower_plot,
    grouped_gradient_hist,
    grouped_lm_marginal,
    grouped_ring_bar,
    pie_3d,
    pie_ring,
    radar_bubble,
    radial_line,
    smooth_radar,
)

COMMANDS = {
    "bar-line": bar_line.main,
    "cartoon-stacked-bar": cartoon_stacked_bar.main,
    "concordance-upset": concordance_upset.main,
    "flower-plot": flower_plot.main,
    "grouped-gradient-hist": grouped_gradient_hist.main,
    "grouped-lm-marginal": grouped_lm_marginal.main,
    "grouped-ring-bar": grouped_ring_bar.main,
    "pie-3d": pie_3d.main,
    "pie-ring": pie_ring.main,
    "radar-bubble": radar_bubble.main,
    "radial-line": radial_line.main,
    "smooth-radar": smooth_radar.main,
}

HELP = """\
usage: main.py <chart> [options]

Reproducible scientific visualization examples.

charts:
  bar-line               stacked N2O bar-plus-line chart
  cartoon-stacked-bar    cartoon stacked-bar policy-mix chart
  concordance-upset      patient-level concordance UpSet variant
  flower-plot            10-group OTU flower / petal chart
  grouped-gradient-hist  grouped residual histogram with gradient bars
  grouped-lm-marginal    grouped linear-fit scatter with marginal histograms
  grouped-ring-bar       grouped annular bar chart of forecasting MSE
  pie-3d                 15-factor 3D contribution pie chart
  pie-ring               pie plus annular stacked energy chart
  radar-bubble           31-region annular radar-bubble chart
  radial-line            64-feature radial phenotype line chart
  smooth-radar           31-task smooth-curve pathology radar chart

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
