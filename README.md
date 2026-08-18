# Research visualization reproductions

## 64-feature radial phenotype line chart

`radial_line.py` reproduces the Xiaohongshu carousel's segmented radial line
chart: 64 questionnaire features, nine semantic groups, four tinnitus
phenotypes, seven z-score rings, and all 18 colour palettes.

Render palette 1 at the 1084 × 1080 reference size:

```bash
uv run python radial_line.py --palette 1
```

Render all 18 palette variants:

```bash
uv run python radial_line.py --palette all
```

Export editable vector files and a 300-DPI PNG:

```bash
uv run python radial_line.py \
  --palette all \
  --formats png svg pdf \
  --dpi 300 \
  --output-dir output/radial_line
```

List palette names and hexadecimal colours:

```bash
uv run python radial_line.py --list-palettes
```

Use it as a library:

```python
from pathlib import Path

from radial_line import PALETTES, create_figure

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

`radar_bubble.py` reproduces the 31-region annular radar-bubble chart from the
provided Xiaohongshu post. The renderer includes the chart geometry, labels,
dual data encodings, legends, smooth closed profiles, and all 18 color
palettes shown in the carousel.

## Run it

Render the first palette at the reference image size (1196 × 1080 pixels):

```bash
uv run python main.py --palette 1
```

Render all 18 palettes:

```bash
uv run python main.py --palette all
```

Export publication-ready PNG, SVG, and PDF files:

```bash
uv run python main.py \
  --palette all \
  --formats png svg pdf \
  --dpi 300 \
  --output-dir output
```

List the palette names and hexadecimal colors:

```bash
uv run python main.py --list-palettes
```

`--palette` accepts either a number (`1` through `18`) or one of those names.

## Customize the data

The source post exposes the rendered images but not its underlying table.
`DEFAULT_DATA` in `radar_bubble.py` therefore contains values digitized from
the visible curve radii and bubble sizes. Replace its three `grain_yield`
arrays and three `planting_area` arrays with your real data; every array must
have one value for each label in `REGIONS`.

For use as a library:

```python
from pathlib import Path

from radar_bubble import PALETTES, create_figure

figure = create_figure(palette=PALETTES[3])
figure.savefig(Path("my_figure.svg"))
```

## Visual mapping

- Smoothed filled profiles: grain yield in 2020, 2010, and 2000.
- Bubble area: planting area for the same three years.
- Bubble radii: fixed at 0.90, 0.70, and 0.50 to keep the three years legible.
- Central white mask: turns the radar chart into an annular composition.

The implementation uses only NumPy and Matplotlib and runs headlessly.
