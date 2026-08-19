# Adding a chart template

Templates are discovered by importing every module in `rvl/templates/` and reading
its `SPEC`. There is no registry list, no CLI table and no `__init__` import to
update: a new module is picked up by `python -m rvl templates`, the recommender, the code
generator and the test suite as soon as it exists.

## Procedure

```
- [ ] 1. Copy the closest existing template as a starting point
- [ ] 2. Write SPEC
- [ ] 3. Write the data class and its from_* builder
- [ ] 4. Write ChartStyle and the drawing code
- [ ] 5. Wire codegen if the builder needs new roles
- [ ] 6. Run the contract tests
- [ ] 7. Render the reference figure and add it to the gallery
```

### 1. Start from the closest template

`rvl/templates/grouped_ring_bar.py` is the reference implementation and the one to
imitate. It shows derived geometry, auto-relaxing style validation, contrast-aware
labels and a `from_matrix` builder. Read
[template-contract.md](template-contract.md) before writing code.

### 2. Write `SPEC`

`SPEC` is what makes the template discoverable and rankable. The fields that
matter most for automatic selection:

| Field | Effect |
|-------|--------|
| `kinds` | Hard filter. Only readings of these kinds are offered. |
| `categories`, `series` | Hard filter on counts, via `Extent(minimum, maximum)`. |
| `affinities` | Signed weights over `Feature`; positive attracts, negative repels. |
| `requires` | Features the data must have, else the template is never offered. |
| `ordered_categories` | Set True only if the layout preserves reading order. |
| `long_category_labels` | Set False if labels must be short to fit. |
| `good_for`, `avoid_when` | Shown by `templates -v`; written for a human reader. |

Choose `Extent` bounds honestly. The maximum is the point past which the figure
stops being readable, not the point past which the code crashes. The recommender
peaks its shape score in the middle of the band, so a wildly generous maximum will
make the template win datasets it draws badly.

If the data needs a kind that does not exist yet, add it to `DataKind` in
`rvl/contract.py`, teach `rvl/profiling.py` to detect it, and add an emitter to
`_EMITTERS` in `rvl/codegen.py`. All three are small, keyed dispatch tables.

### 3. Write the data class and builder

One frozen dataclass holding exactly what gets drawn, and one `@classmethod`
`from_*` builder named in `SPEC.builder`. The builder takes plain sequences and
mappings, converts `None` to NaN, and calls `validate()` before returning.

`validate()` checks internal consistency — ragged rows, non-finite values, negative
counts where negatives are impossible, labels that do not match their axis. It must
not check that the labels equal the reference figure's labels. That mistake is what
made the original library unusable for real data.

### 4. Write `ChartStyle` and the drawing code

Derive every count-dependent quantity from the data. Angular pitch is
`360 / len(categories)`, not a hard-coded 45 degrees. Bar widths follow from the
series count. Keep the reference pixel dimensions in the docstring.

For axis limits, keep the reference values as defaults but declare the type as
`tuple[float, float] | None` and pass them through `rvl.render.resolve_limits`.
That honours a hand-tuned range while the data genuinely fills it, and auto-fits
anything that would be clipped or reduced to a sliver.

### 5. Wire codegen when the builder needs new roles

The generator addresses builders by role: `categories`, `series`, `values`,
`value_label`, and kind-specific ones such as `sets`, `memberships`, `totals`,
`parts`. If the builder names a parameter differently, declare the mapping rather
than special-casing the generator:

```python
argument_names=(("categories", "spokes"),)
```

Set `transpose_values=True` if the builder wants `values[series][category]`.

### 6. Run the contract tests

```bash
export MPLCONFIGDIR="$PWD/.mplcache"
.venv/bin/python -m unittest discover -s tests -q
```

`tests/test_contract.py` runs against every discovered template and checks the
module surface, that `SPEC` shape limits are self-consistent, that `DEFAULT_DATA`
renders, that the builder is callable and produces a working figure from synthetic
data at both ends of the supported range, and that palettes are well formed. A new
template needs no new test file to be covered.

Add a focused `tests/test_<template>.py` only for behaviour unique to the template,
such as a length-scale mode or an optional annotation.

### 7. Render the reference figure

```bash
.venv/bin/python -m rvl render <template-id> --palette 1 --output-dir docs/gallery
```

Rename the output to `docs/gallery/<template_id>.png` and add a row to the README
gallery table. The gallery is the human-facing index; `templates -v` is the
machine-facing one.

## Common mistakes

- **Reading a module-level label constant inside a drawing helper.** This is the
  original bug. Counts and labels come from the data object only.
- **A fixed-length palette tuple.** Use `palette.take(n)`; `Palette.color` extends
  past the cycle by shifting lightness.
- **Hard-coded axis titles.** They belong in data fields so a user's units can
  replace them.
- **`plt.rcParams.update(...)` inside `create_figure`.** That leaks style into every
  later figure in the process. Use `plt.rc_context`.
- **Validating against the reference shape.** `validate()` guards consistency, not
  identity.
