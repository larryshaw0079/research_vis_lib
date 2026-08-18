# Research visualization reproductions

The chart renderers live in the `src` package. `main.py` at
the project root is the command-line entry point.

```bash
uv run python main.py --help
uv run python -m src --help
```

## 64-feature radial phenotype line chart

`src.radial_line` reproduces the Xiaohongshu carousel's segmented
radial line chart: 64 questionnaire features, nine semantic groups, four
tinnitus phenotypes, seven z-score rings, and all 18 colour palettes.

Render palette 1 at the 1084 × 1080 reference size:

```bash
uv run python main.py radial-line --palette 1
```

Render all 18 palette variants:

```bash
uv run python main.py radial-line --palette all
```

Export editable vector files and a 300-DPI PNG:

```bash
uv run python main.py radial-line \
  --palette all \
  --formats png svg pdf \
  --dpi 300 \
  --output-dir output/radial_line
```

List palette names and hexadecimal colours:

```bash
uv run python main.py radial-line --list-palettes
```

Use it as a library:

```python
from pathlib import Path

from src.radial_line import PALETTES, create_figure

figure = create_figure(palette=PALETTES[0])
figure.savefig(Path("radial_line.svg"))
```

The chart structure and aggregate phenotype means come from Figure 2 of
[Niemann et al. (2020)](https://doi.org/10.1038/s41598-020-73402-8), which is
licensed CC BY 4.0. The patient-level data are not public; `DEFAULT_DATA`
contains values digitized from the published line and single-phenotype bar
charts. Replace those four arrays with your own 64-feature z-score means when
using the renderer with another dataset.

## Annular radar-bubble chart

`src.radar_bubble` reproduces the 31-region annular radar-bubble
chart from the provided Xiaohongshu post. The renderer includes the chart
geometry, labels, dual data encodings, legends, smooth closed profiles, and
all 18 color palettes shown in the carousel.

Render the first palette at the reference image size (1196 × 1080 pixels):

```bash
uv run python main.py radar-bubble --palette 1
```

Render all 18 palettes:

```bash
uv run python main.py radar-bubble --palette all
```

Export publication-ready PNG, SVG, and PDF files:

```bash
uv run python main.py radar-bubble \
  --palette all \
  --formats png svg pdf \
  --dpi 300 \
  --output-dir output
```

List the palette names and hexadecimal colors:

```bash
uv run python main.py radar-bubble --list-palettes
```

`--palette` accepts either a number (`1` through `18`) or one of those names.

## Customize the data

The source post exposes the rendered images but not its underlying table.
`DEFAULT_DATA` in `src.radar_bubble` therefore contains values
digitized from the visible curve radii and bubble sizes. Replace its three
`grain_yield` arrays and three `planting_area` arrays with your real data;
every array must have one value for each label in `REGIONS`.

For use as a library:

```python
from pathlib import Path

from src.radar_bubble import PALETTES, create_figure

figure = create_figure(palette=PALETTES[3])
figure.savefig(Path("my_figure.svg"))
```

## Visual mapping

- Smoothed filled profiles: grain yield in 2020, 2010, and 2000.
- Bubble area: planting area for the same three years.
- Bubble radii: fixed at 0.90, 0.70, and 0.50 to keep the three years legible.
- Central white mask: turns the radar chart into an annular composition.

The implementation uses only NumPy and Matplotlib and runs headlessly.

## Grouped annular bar chart

`src.grouped_ring_bar` reproduces the Xiaohongshu carousel's
grouped polar bar chart: eight forecasting datasets, five models, inverted
within-dataset MSE bars, curved sector labels, a central legend, and all 18
colour palettes.

Render palette 1 at the 2601 × 2601 reference size:

```bash
uv run python main.py grouped-ring-bar --palette 1
```

Render all 18 palette variants:

```bash
uv run python main.py grouped-ring-bar --palette all
```

Export editable vector files and a 300-DPI PNG:

```bash
uv run python main.py grouped-ring-bar \
  --palette all \
  --formats png svg pdf \
  --dpi 300 \
  --output-dir output/grouped_ring_bar
```

List palette names and hexadecimal colours:

```bash
uv run python main.py grouped-ring-bar --list-palettes
```

Use it as a library:

```python
from pathlib import Path

from src.grouped_ring_bar import PALETTES, create_figure

figure = create_figure(palette=PALETTES[0])
figure.savefig(Path("grouped_ring_bar.svg"))
```

The comparison is the horizon-720 MSE panel associated with
[Ma et al. (2025)](https://openreview.net/forum?id=sbvLts2HqR). `DEFAULT_DATA`
contains the values printed on the carousel bars, which match that table for
most dataset–model pairs; a few labelled entries differ from the camera-ready
numbers and were kept as shown in the figure. Replace the nested `mse` mapping
when using the renderer with another table.

Within each dataset the shortest bar is the worst (largest) MSE and the longest
bar is the best (smallest) MSE, so the five models are comparable inside a
sector even when absolute scores differ across datasets. "MoFo (Ours)" is
drawn last in every group and labelled in bold red in the legend.

## Grouped linear-fit scatter with marginal histograms

`src.grouped_lm_marginal` reproduces the Xiaohongshu carousel's
GST–LST land-cover scatter: four groups (Grass, Land, Water, Urban), dashed
OLS fits with 95% confidence bands, stacked histograms plus dashed KDE
curves on the top and right margins, and all 16 colour palettes.

Render palette 1 at the 2132 × 1962 reference size:

```bash
uv run python main.py grouped-lm-marginal --palette 1
```

Render all 16 palette variants:

```bash
uv run python main.py grouped-lm-marginal --palette all
```

Export editable vector files and a 300-DPI PNG:

```bash
uv run python main.py grouped-lm-marginal \
  --palette all \
  --formats png svg pdf \
  --dpi 300 \
  --output-dir output/grouped_lm_marginal
```

List palette names and hexadecimal colours:

```bash
uv run python main.py grouped-lm-marginal --list-palettes
```

Use it as a library:

```python
from pathlib import Path

from src.grouped_lm_marginal import PALETTES, create_figure

figure = create_figure(palette=PALETTES[0])
figure.savefig(Path("grouped_lm_marginal.svg"))
```

The source post publishes the chart but not the underlying table.
`DEFAULT_DATA` is a synthetic 25-point sample per group whose OLS slope,
intercept, $R^2$, and $p$-value round to the printed annotations (Grass
$R^2=0.845$, Urban $p=0.011$). Replace the `gst` and `lst` arrays when
using the renderer with another dataset.
