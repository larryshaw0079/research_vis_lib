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

## OTU flower / petal chart

`src.flower_plot` reproduces the Xiaohongshu carousel's 10-petal OTU
flower plot: a shared core count, per-group totals and unique counts,
clockwise labels from CK through Mix-H, and all 18 colour palettes.

Render palette 1 at the 2560 × 2560 reference size:

```bash
uv run python main.py flower-plot --palette 1
```

Render all 18 palette variants:

```bash
uv run python main.py flower-plot --palette all
```

Export editable vector files and a 300-DPI PNG:

```bash
uv run python main.py flower-plot \
  --palette all \
  --formats png svg pdf \
  --dpi 300 \
  --output-dir output/flower_plot
```

List palette names and hexadecimal colours:

```bash
uv run python main.py flower-plot --list-palettes
```

Use it as a library:

```python
from pathlib import Path

from src.flower_plot import PALETTES, create_figure

figure = create_figure(palette=PALETTES[0])
figure.savefig(Path("flower_plot.svg"))
```

The source post publishes the chart but not the underlying table.
`DEFAULT_DATA` contains the core, total, and unique counts printed on the
petals. Replace those mappings when using the renderer with another
dataset. Petal text switches between black and white from the fill
luminance so the labels stay readable on both pastel and dark palettes.

## Smooth-curve pathology radar chart

`src.smooth_radar` reproduces the Xiaohongshu carousel's 31-task AUROC
radar: five models (EAGLE, CHIEF, GigaPath, CTransPath, Virchow2),
periodic spline profiles, two-line category boxes, and all 18 colour
palettes.

Render palette 1 at the 2560 × 2243 reference size:

```bash
uv run python main.py smooth-radar --palette 1
```

Render all 18 palette variants:

```bash
uv run python main.py smooth-radar --palette all
```

Export editable vector files and a 300-DPI PNG:

```bash
uv run python main.py smooth-radar \
  --palette all \
  --formats png svg pdf \
  --dpi 300 \
  --output-dir output/smooth_radar
```

List palette names and hexadecimal colours:

```bash
uv run python main.py smooth-radar --list-palettes
```

Use it as a library:

```python
from pathlib import Path

from src.smooth_radar import PALETTES, create_figure

figure = create_figure(palette=PALETTES[0])
figure.savefig(Path("smooth_radar.svg"))
```

Task names and categories follow Figure 1c of
[Neidlinger et al. (2026)](https://doi.org/10.1038/s41467-026-74918-9),
licensed CC BY 4.0. The source post prints two-decimal scores that do not
match that paper's fold-mean table; `DEFAULT_DATA` stores the labelled
radii. Replace the five AUROC arrays when using the renderer with another
dataset.

## Grouped gradient histogram

`src.grouped_gradient_hist` reproduces the Xiaohongshu carousel's
three-group residual histogram: 14 interval bins, vertical colour-to-white
bar gradients, dashed normal-fit overlays, and all 18 colour palettes.

Render palette 1 at the 2121 × 1762 reference size:

```bash
uv run python main.py grouped-gradient-hist --palette 1
```

Render all 18 palette variants:

```bash
uv run python main.py grouped-gradient-hist --palette all
```

Export editable vector files and a 300-DPI PNG:

```bash
uv run python main.py grouped-gradient-hist \
  --palette all \
  --formats png svg pdf \
  --dpi 300 \
  --output-dir output/grouped_gradient_hist
```

List palette names and hexadecimal colours:

```bash
uv run python main.py grouped-gradient-hist --list-palettes
```

Use it as a library:

```python
from pathlib import Path

from src.grouped_gradient_hist import PALETTES, create_figure

figure = create_figure(palette=PALETTES[0])
figure.savefig(Path("grouped_gradient_hist.svg"))
```

The source post publishes the chart but not the residual table.
`DEFAULT_DATA` reconstructs one residual per visible bar count. The
legend keeps the printed moments (Zhang $\mu=-4.9$, Kioumarsi
$\sigma=25.1$, Xue $\mu=-3.3$). Replace the three residual arrays when
using the renderer with another dataset.

## 15-factor 3D pie chart

`src.pie_3d` reproduces the Xiaohongshu carousel's 3D contribution pie:
15 remote-sensing factors from GSR through ASPECT, exploded elliptical
wedges with visible side walls, a left-hand legend, and all 18 colour
palettes.

Render palette 1 at the 2019 × 1350 reference size:

```bash
uv run python main.py pie-3d --palette 1
```

Render all 18 palette variants:

```bash
uv run python main.py pie-3d --palette all
```

Export editable vector files and a 300-DPI PNG:

```bash
uv run python main.py pie-3d \
  --palette all \
  --formats png svg pdf \
  --dpi 300 \
  --output-dir output/pie_3d
```

List palette names and hexadecimal colours:

```bash
uv run python main.py pie-3d --list-palettes
```

Use it as a library:

```python
from pathlib import Path

from src.pie_3d import PALETTES, create_figure

figure = create_figure(palette=PALETTES[0])
figure.savefig(Path("pie_3d.svg"))
```

The source post publishes the chart but not the underlying table.
`DEFAULT_DATA` contains the percentages printed on the wedges. Replace
that mapping when using the renderer with another dataset.

## Patient-level concordance UpSet variant

`src.concordance_upset` reproduces the Xiaohongshu carousel's UpSet-style
chart of tissue–plasma mutation concordance: eight combination columns,
three concordance groups, a check/cross membership matrix, and all 18
colour palettes.

Render palette 1 at the 2431 × 1603 reference size:

```bash
uv run python main.py concordance-upset --palette 1
```

Render all 18 palette variants:

```bash
uv run python main.py concordance-upset --palette all
```

Export editable vector files and a 300-DPI PNG:

```bash
uv run python main.py concordance-upset \
  --palette all \
  --formats png svg pdf \
  --dpi 300 \
  --output-dir output/concordance_upset
```

List palette names and hexadecimal colours:

```bash
uv run python main.py concordance-upset --list-palettes
```

Use it as a library:

```python
from pathlib import Path

from src.concordance_upset import PALETTES, create_figure

figure = create_figure(palette=PALETTES[0])
figure.savefig(Path("concordance_upset.svg"))
```

The combination counts follow Figure 4b of
[Zhang et al. (2026)](https://doi.org/10.1038/s41698-026-01549-0),
licensed CC BY 4.0: 420 tissue-only, 7 plasma-only, and 76 tissue+plasma
patients in the first group; 8 double-negative and 42 shared-only patients
in the complete-concordant group; and 99, 209, and 250 partially
concordant patients. The carousel labels the first group "Disconcordant"
(the paper uses "discordant"). Replace `DEFAULT_DATA` when using the
renderer with another table.

## Stacked N2O bar-plus-line chart

`src.bar_line` reproduces the Xiaohongshu carousel's two-panel N2O figure:
horizontal LF / HF cumulative-emission bars with replicate points, SEM
caps, and a `**` significance bracket, plus the matching flux time series
with error bars, two dashed period guides, and all 18 colour palettes.

Render palette 1 at the 1260 × 1080 reference size:

```bash
uv run python main.py bar-line --palette 1
```

Render all 18 palette variants:

```bash
uv run python main.py bar-line --palette all
```

Export editable vector files and a 300-DPI PNG:

```bash
uv run python main.py bar-line \
  --palette all \
  --formats png svg pdf \
  --dpi 300 \
  --output-dir output/bar_line
```

List palette names and hexadecimal colours:

```bash
uv run python main.py bar-line --list-palettes
```

Use it as a library:

```python
from pathlib import Path

from src.bar_line import PALETTES, create_figure

figure = create_figure(palette=PALETTES[0])
figure.savefig(Path("bar_line.svg"))
```

The source post publishes the chart but not the flux table, and notes that
the top bars were a simple sum of the time-series points. `DEFAULT_DATA`
keeps that shortcut (LF $303$, HF $467$) so the rendered bars match the
carousel. Replace the flux arrays or the cumulative mapping when using the
renderer with another dataset.
