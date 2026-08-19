---
name: research-vis-lib
description: >-
  Reads experimental data (.xlsx, .xls, .csv, .tsv, .json, .jsonl, .md), picks the
  best-fitting chart from a library of publication-grade matplotlib templates, and
  generates standalone Python that plots that data faithfully. Use when the user
  has a results file, benchmark table, measurement export or spreadsheet and wants
  a figure, asks which chart suits their data, asks for plotting code for a paper
  figure, or wants to add a new chart template to the library.
---

# Research chart selection and code generation

Turn an experimental data file into a verified figure script in four steps. Every
step has a command; run them rather than reimplementing the logic.

Run all commands from the repository root with the local environment:

```bash
export MPLCONFIGDIR="$PWD/.mplcache"
.venv/bin/python -m rvl --help
```

## Workflow

Copy this checklist and keep it updated:

```
- [ ] Step 1: profile the data file
- [ ] Step 2: choose a template and justify it
- [ ] Step 3: generate the script
- [ ] Step 4: verify fidelity, then show the figure
```

### Step 1: profile the data

```bash
.venv/bin/python -m rvl profile path/to/results.xlsx
```

This prints, per sheet or table: column roles, value range, whether values are
non-negative, whether they sum to 100, dynamic range, detected uncertainties,
emphasised (bold) cells, and every **reading** the data admits. A reading is a
concrete way to interpret the table, such as "matrix: 8 categories x 5 series" or
"set_membership: 8 combinations".

Read the output before going further. If it reports no numeric column or no
reading, the file is not plottable as-is — say so and ask what the user wants
plotted instead of guessing.

### Step 2: choose a template

```bash
.venv/bin/python -m rvl recommend path/to/results.xlsx
```

The recommender applies hard filters (data kind, supported category and series
counts) and then scores the survivors, printing a reason and any warnings per
candidate. Treat the ranking as a shortlist, not a verdict.

Make the final call yourself using [references/selection-rubric.md](references/selection-rubric.md),
then **tell the user which template you picked and why in one or two sentences**,
including the runner-up if it was close. To see what a template expects:

```bash
.venv/bin/python -m rvl templates -v
```

Override the top pick with `--template <id>` whenever the rubric or the user's
stated intent disagrees with the score. The score cannot know that the user is
writing a paper whose other figures are all bar charts.

### Step 3: generate the script

```bash
.venv/bin/python -m rvl generate path/to/results.xlsx \
    --template grouped-ring-bar \
    --palette 1 \
    -o figures/my_figure.py
```

The generated script inlines the measurements as literals and builds its data
through the template's `from_*` builder, which validates shape and rejects
inconsistent input. It does not re-read the data file, so it runs anywhere and a
reviewer can see exactly what will be drawn.

Then read the generated script. You are responsible for the axis titles, units
and value formats: the generator carries over column headers, which are rarely
publication-ready. Edit labels, `value_format`, `lower_is_better` and `highlight`
in the script — never the numbers.

### Step 4: verify fidelity

```bash
.venv/bin/python -m rvl verify figures/my_figure.py --source path/to/results.xlsx
```

This runs four checks: the script builds data through a real builder, every
measurement-looking literal appears in the source file, no template reference
values leaked in, and the script executes and writes a non-empty figure.

**Do not show the user a figure until verify passes.** If it fails, fix the script
and re-run; do not suppress the check. Then read the rendered PNG yourself to
confirm it is legible — no overlapping labels, no clipped bars, no illegible
text — and embed it in your reply so the user can see it.

## Fidelity rules

These are the point of the skill. Breaking one produces a figure that misrepresents
an experiment.

1. **Never plot `DEFAULT_DATA`.** Every template ships reference data digitised
   from a published figure or a social-media carousel. It exists to demonstrate
   layout. It is somebody else's experiment.
2. **Never invent, interpolate or extrapolate a value.** If a cell is missing,
   pass `None`; the builders keep it as NaN and the templates skip it. Do not
   substitute zero — zero is a real measurement.
3. **Never renormalise silently.** If values already sum to 100, say so and keep
   them; if you convert raw magnitudes to shares, state it in your reply.
4. **Preserve row order** unless the user asks for sorting, and say so if you sort.
5. **Check the metric direction.** For error metrics (MSE, MAE, loss, latency) set
   `lower_is_better=True`, otherwise a longer bar will mean a worse result.
6. **Watch length-based encodings.** Some templates rescale within a category so
   bar length is not proportional to value. `grouped-ring-bar` defaults to
   `length_scale="absolute"` for this reason; do not switch to
   `"within-category"` for a user's data without telling them the axis is no
   longer readable.
7. **Report what was used.** Name the file, sheet and columns you plotted, and
   mention any rows you dropped and why.

## Choosing among close candidates

Full rules are in [references/selection-rubric.md](references/selection-rubric.md).
The short version:

- **Ordered category axis** (dates, doses, epochs): prefer a template with
  `ordered_categories`. Circular and polar layouts hide where a sequence starts.
- **Values sum to a whole**: prefer a composition template over a matrix one.
- **Wide dynamic range** (more than ~100x): warn the user; radial length
  encodings compress small values into invisibility.
- **Long category labels**: avoid templates flagged `long_category_labels=False`.
- **Precision matters more than impact**: prefer cartesian over polar.
- **Few categories and few series**: the exotic templates look empty; say so.

## Adding a chart template

The library currently ships 12 templates and is designed to grow. A new template
is discovered automatically — there is no central list to edit. See
[references/adding-templates.md](references/adding-templates.md) for the full
procedure and [references/template-contract.md](references/template-contract.md)
for the module contract.

In brief: drop a module into `rvl/templates/` that exposes `SPEC`, `PALETTES`, a
data class with one `from_*` builder, `DEFAULT_DATA`, `ChartStyle`,
`DEFAULT_STYLE`, `create_figure` and a two-line `main`. Then run:

```bash
.venv/bin/python -m unittest discover -s tests -q
.venv/bin/python -m rvl templates
```

`tests/test_contract.py` enforces the contract across every template, so a new
one is validated by the existing suite.

## Reference files

| File | Open when |
|------|-----------|
| [references/selection-rubric.md](references/selection-rubric.md) | Deciding between close candidates, or the recommender returns nothing |
| [references/template-contract.md](references/template-contract.md) | Writing or fixing a template module |
| [references/adding-templates.md](references/adding-templates.md) | Adding a 13th template |
| [references/data-formats.md](references/data-formats.md) | The data file is messy, multi-sheet, or fails to parse |

## Commands

| Command | Purpose |
|---------|---------|
| `python -m rvl profile <data>` | Column roles, value statistics and possible readings |
| `python -m rvl recommend <data>` | Ranked templates with reasons and warnings |
| `python -m rvl generate <data> [--template ID] -o out.py` | Emit a standalone figure script |
| `python -m rvl verify <script> --source <data>` | Fidelity and render checks |
| `python -m rvl templates [-v] [--json]` | List templates and their data contracts |
| `python -m rvl render <template> --palette all` | Render a template's reference figure across palettes |

## Worked example

```bash
export MPLCONFIGDIR="$PWD/.mplcache"

# 1. What is in the file?
.venv/bin/python -m rvl profile examples/forecasting_mse.xlsx

# 2. What could draw it?
.venv/bin/python -m rvl recommend examples/forecasting_mse.xlsx

# 3. Emit code for the chosen template.
.venv/bin/python -m rvl generate examples/forecasting_mse.xlsx \
    --template grouped-ring-bar --table MSE -o figures/mse_ring.py

# 4. Prove it draws the source data, then look at it.
.venv/bin/python -m rvl verify figures/mse_ring.py \
    --source examples/forecasting_mse.xlsx
```

`examples/` holds one dataset per data kind, in every supported format, for
trying the pipeline without user data. Regenerate them with
`.venv/bin/python examples/build_examples.py`.
