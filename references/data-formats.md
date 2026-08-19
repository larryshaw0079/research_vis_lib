# Data formats and messy files

`rvl.ingest` reads a file into one or more `Table` objects. Every value keeps the
coordinate it came from — `MSE!C4` for a spreadsheet, `$.factors[2].contribution`
for JSON — which is what lets `python -m rvl verify` prove a plotted number exists in
the source.

## Supported inputs

| Suffix | Reader | Notes |
|--------|--------|-------|
| `.xlsx`, `.xlsm` | openpyxl | One table per worksheet. Formulas are read as their cached values. |
| `.xls` | xlrd | Legacy only. Install with `uv pip install xlrd`, or re-save as `.xlsx`. |
| `.csv`, `.tsv`, `.txt`, `.tab` | stdlib `csv` | Delimiter is sniffed; `.tsv` and `.tab` force tab. |
| `.md`, `.markdown` | built-in parser | One table per pipe table; the nearest heading above becomes the table name. |
| `.json` | stdlib `json` | Every list-of-objects becomes a table. A file with no record array falls back to a flattened key/value table. |
| `.jsonl`, `.ndjson` | stdlib `json` | One object per line. |

`read_tables` returns every table in the file; `read_table` picks the largest.
Pass `--table <name>` to choose a specific sheet or markdown table.

## Cell parsing

`parse_measurement` understands more than plain floats, because research exports
rarely contain plain floats:

| Cell | Parsed as |
|------|-----------|
| `0.447` | value 0.447 |
| `1,234` | value 1234 |
| `12.5%` | value 12.5, `is_percent=True` |
| `0.45 ± 0.02`, `0.45 +/- 0.02` | value 0.45, error 0.02 |
| `0.45 (0.02)` | value 0.45, error 0.02 |
| `**0.94**`, `__0.94__` | value 0.94, `emphasised=True` |
| `0.42**` | value 0.42, significance markers stripped |
| `NA`, `n/a`, `--`, `—`, empty | missing |
| `yes`, `✓`, `true`, `1` | boolean True |
| `no`, `×`, `false`, `0` | boolean False |

Bold emphasis matters: benchmark tables mark the winning method in bold, and the
profiler surfaces that as the `HAS_EMPHASIS` feature so a template that supports
`highlight` can use it.

## Column roles

Each column is classified by inspecting its values:

- **boolean** — every value parses as a flag and there are at most two distinct
  values. Checked before numeric, so a 0/1 membership column is not mistaken for a
  measurement.
- **temporal** — at least 80% look like dates or contain a month name. Drives the
  `ORDERED_CATEGORIES` feature.
- **numeric** — at least 80% parse as measurements.
- **categorical** — everything else.
- **empty** — no values present. Dropped, since these are usually spacer columns.

Columns whose names mention `sd`, `std`, `sem`, `se`, `err`, `ci` or `sigma` are
treated as uncertainties rather than as series, and paired to the value column
whose name they contain.

Columns whose names mention `mse`, `mae`, `rmse`, `loss`, `latency`, `runtime`,
`cost` or `rank` — or a sheet with such a name — set `lower_is_better`.

## Layout detection

The profiler does not require a particular layout. It tries several readings and
reports all of them:

- **Wide**: first label column supplies categories, numeric column headers supply
  series. The usual benchmark-table shape.
- **Long / tidy**: two label columns and one value column are pivoted into a
  matrix. Also offered as `nested_parts` and `stacked_parts` when non-negative.
- **Paired**: two numeric columns become x and y, optionally split by a group
  column with 2-8 levels.
- **Repeated group**: one label column repeating over many rows with a single
  numeric column becomes per-group distributions.
- **Membership**: two or more boolean columns plus a count column.

A file can yield several readings, and `recommend` scores templates against each.

## Fixing files that will not parse

**"found no usable table"** — the sheet is probably title rows and merged cells
above the real header. Delete the preamble, or point at a cleaner sheet with
`--table`.

**Headers not detected** — the reader treats the first non-empty row as a header
only when most of its filled cells are non-numeric. A table whose header row is
years (`2020 2021 2022`) is read as data. Prefix them (`y2020`) or add a text
column label.

**A units row under the header** — it becomes a data row full of text, which turns
the numeric columns categorical. Delete it; put units in the axis label instead.

**Merged category cells** — a merged label spanning several rows reads as one label
and then blanks. Fill the label down before reading.

**Multi-level headers** — two header rows are not supported. Flatten to one row,
combining the levels with an underscore.

**Everything came back categorical** — usually a stray footnote row, a `<0.001`
cell, or a thousands separator the locale writes as `.`. Check the profile output:
it names each column's inferred role.

## Multi-sheet workbooks

`profile` prints every sheet. `generate` uses the largest table and prints a note
naming the others. Pass `--table` to be explicit; a workbook with a `notes` or
`metadata` sheet will otherwise report that sheet as unplottable, which is correct
but noisy.
