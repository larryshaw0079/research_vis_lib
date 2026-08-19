# research-vis-lib

Reproducible matplotlib chart examples. Each module in `src` exposes palettes, `DEFAULT_DATA`, and `create_figure`.

## Gallery

Palette 1 of every chart. CLI names are in the headers.

| `bar-line` | `cartoon-stacked-bar` | `concordance-upset` |
|:---:|:---:|:---:|
| <img src="docs/gallery/bar_line.png" width="280" alt="bar-line"> | <img src="docs/gallery/cartoon_stacked_bar.png" width="280" alt="cartoon-stacked-bar"> | <img src="docs/gallery/concordance_upset.png" width="280" alt="concordance-upset"> |
| stacked N2O bar + line | cartoon policy-mix bars | concordance UpSet |

| `flower-plot` | `grouped-gradient-hist` | `grouped-lm-marginal` |
|:---:|:---:|:---:|
| <img src="docs/gallery/flower_plot.png" width="280" alt="flower-plot"> | <img src="docs/gallery/grouped_gradient_hist.png" width="280" alt="grouped-gradient-hist"> | <img src="docs/gallery/grouped_lm_marginal.png" width="280" alt="grouped-lm-marginal"> |
| OTU flower / petal | gradient residual histogram | scatter + marginal histograms |

| `grouped-ring-bar` | `pie-3d` | `pie-ring` |
|:---:|:---:|:---:|
| <img src="docs/gallery/grouped_ring_bar.png" width="280" alt="grouped-ring-bar"> | <img src="docs/gallery/pie_3d.png" width="280" alt="pie-3d"> | <img src="docs/gallery/pie_ring.png" width="280" alt="pie-ring"> |
| annular grouped bars | 3D contribution pie | pie + annular stacked sectors |

| `radar-bubble` | `radial-line` | `smooth-radar` |
|:---:|:---:|:---:|
| <img src="docs/gallery/radar_bubble.png" width="280" alt="radar-bubble"> | <img src="docs/gallery/radial_line.png" width="280" alt="radial-line"> | <img src="docs/gallery/smooth_radar.png" width="280" alt="smooth-radar"> |
| annular radar-bubble | radial phenotype lines | smooth pathology radar |

## Usage

Python 3.14+. Shared flags: `--palette` (number, name, or `all`), `--formats`, `--dpi`, `--output-dir`, `--list-palettes`.

```bash
uv run python main.py --help
uv run python main.py radial-line --palette 1
uv run python main.py radial-line --palette all --formats png svg pdf --dpi 300
```

```python
from src.radial_line import PALETTES, create_figure

figure = create_figure(palette=PALETTES[0])
figure.savefig("radial_line.svg")
```

Replace `DEFAULT_DATA` when using another table. Sources for published figures are in the module docstrings.
