"""Emit a standalone Python script that plots a real dataset with a template.

The generated script inlines the measurements as literals rather than re-reading
the source file, for three reasons: it runs anywhere without the data file, a
reviewer can read exactly what will be drawn, and :mod:`rvl.verify` can compare
the literals against the source to prove nothing drifted.

The mapping from a reading of the data to builder arguments lives in
:mod:`rvl.adapters`; this module only renders those values as source.
"""

from __future__ import annotations

import datetime as dt
import math
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

from .adapters import AdapterError, builder_kwargs
from .contract import is_presentation_argument
from .profiling import Interpretation
from .recommend import Candidate

_INDENT: Final[str] = "    "
_MAX_LINE: Final[int] = 88


class CodegenError(RuntimeError):
    """The chosen template cannot be driven by the chosen reading of the data."""


@dataclass(frozen=True, slots=True)
class GeneratedScript:
    """A rendered script plus what it is expected to draw."""

    template_id: str
    code: str
    figure_stem: str
    plotted_values: tuple[float, ...]
    references: tuple[str, ...]

    def write(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.code, encoding="utf-8")
        return destination


def _literal(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        # NaN is how a missing measurement travels; keep it readable as None so the
        # builder's own None handling turns it back into NaN.
        return "None" if math.isnan(value) else repr(value)
    if isinstance(value, (int, str)):
        return repr(value)
    return repr(value)


def _is_row(value: Any) -> bool:
    return isinstance(value, (tuple, list)) and not any(
        isinstance(item, (tuple, list)) for item in value
    )


def _render_flat(values: Sequence[Any], indent: str) -> list[str]:
    items = [_literal(item) for item in values]
    single = "(" + ", ".join(items) + ("," if len(items) == 1 else "") + ")"
    if len(indent) + len(single) <= _MAX_LINE:
        return [single]
    lines = ["("]
    chunk: list[str] = []
    for item in items:
        chunk.append(item)
        if len(indent) + 4 + len(", ".join(chunk)) > _MAX_LINE - 2:
            lines.append(f"{_INDENT}{', '.join(chunk)},")
            chunk = []
    if chunk:
        lines.append(f"{_INDENT}{', '.join(chunk)},")
    lines.append(")")
    return lines


def _render_nested(
    rows: Sequence[Sequence[Any]], labels: Sequence[Any], indent: str
) -> list[str]:
    lines = ["("]
    for index, row in enumerate(rows):
        label = str(labels[index]) if index < len(labels) else ""
        comment = f"  # {label}" if label else ""
        inner = _render_flat(row, indent + _INDENT)
        if len(inner) == 1:
            candidate = f"{_INDENT}{inner[0]},{comment}"
            if len(indent) + len(candidate) <= _MAX_LINE:
                lines.append(candidate)
                continue
        lines.append(f"{_INDENT}({comment}" if comment else f"{_INDENT}(")
        for part in inner[1:-1]:
            lines.append(f"{_INDENT}{part}")
        lines.append(f"{_INDENT}),")
    lines.append(")")
    return lines


def _render_value(value: Any, labels: Sequence[Any], indent: str) -> str:
    if isinstance(value, (tuple, list)):
        if _is_row(value):
            return "\n".join(_render_flat(value, indent))
        return "\n".join(_render_nested(value, labels, indent))
    return _literal(value)


def _keyword_block(arguments: Mapping[str, Any], labels: Sequence[Any]) -> str:
    lines: list[str] = []
    for name, value in arguments.items():
        rendered = _render_value(value, labels, _INDENT)
        if "\n" not in rendered:
            lines.append(f"{_INDENT}{name}={rendered},")
            continue
        parts = rendered.splitlines()
        lines.append(f"{_INDENT}{name}={parts[0]}")
        lines.extend(f"{_INDENT}{part}" for part in parts[1:-1])
        lines.append(f"{_INDENT}{parts[-1]},")
    return "\n".join(lines)


def _provenance(references: Sequence[str], limit: int = 10) -> str:
    if not references:
        return "Source cells were not recorded for this reading."
    shown = ", ".join(references[:limit])
    if len(references) <= limit:
        return f"Source cells: {shown}."
    return f"Source cells: {shown}, and {len(references) - limit} more."


def _row_labels(reading: Interpretation, arguments: Mapping[str, Any]) -> tuple[str, ...]:
    """Labels to comment each row of a nested literal with."""

    widths = {
        len(value)
        for value in arguments.values()
        if isinstance(value, (tuple, list)) and not _is_row(value)
    }
    if widths == {reading.n_series}:
        return reading.series
    return reading.categories


def generate(
    candidate: Candidate,
    *,
    palette: str | int = 1,
    dpi: int | None = None,
    formats: Sequence[str] = ("png",),
    output_dir: str = "output",
    figure_stem: str | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> GeneratedScript:
    """Build a standalone script that draws ``candidate`` from inlined literals."""

    spec = candidate.spec
    reading = candidate.interpretation
    try:
        arguments = dict(builder_kwargs(spec, reading))
    except AdapterError as exc:
        raise CodegenError(str(exc)) from exc

    for name, value in (overrides or {}).items():
        arguments[name] = value

    stem = figure_stem or f"{spec.template_id.replace('-', '_')}_{_slug(reading.table)}"
    resolved_dpi = spec.default_dpi if dpi is None else dpi
    labels = _row_labels(reading, arguments)
    plotted = _numeric_leaves(
        {
            name: value
            for name, value in arguments.items()
            if not is_presentation_argument(name)
        }
    )

    references = _reference_list(reading)
    provenance = "\n".join(
        textwrap.wrap(_provenance(references), 74, subsequent_indent="")
    )
    header = f'''"""Draw {spec.title.lower()} from {reading.table!r}.

Generated by research-vis-lib on {dt.date.today().isoformat()}.

Source:  {reading.source or "not recorded"}
Table:   {reading.table}
Reading: {reading.kind.value}, {reading.n_categories} categories x {reading.n_series} series

{provenance}

Every number below is copied from that source. Change the labels, units and
formats freely; re-run the generator rather than editing the values by hand, so
the figure and the source stay in step.
"""

from pathlib import Path

from {spec.module_name} import {spec.data_class}, PALETTES, create_figure
from rvl.palettes import palette_from_selector
from rvl.render import save_figure

DATA = {spec.data_class}.{spec.builder_name}(
{_keyword_block(arguments, labels)}
)


def main() -> None:
    index, palette = palette_from_selector(PALETTES, {palette!r})
    figure = create_figure(palette=palette, data=DATA)
    for path in save_figure(
        figure,
        Path({output_dir!r}),
        {stem!r},
        formats={tuple(formats)!r},
        dpi={resolved_dpi},
    ):
        print(path)


if __name__ == "__main__":
    main()
'''

    return GeneratedScript(
        template_id=spec.template_id,
        code=header,
        figure_stem=stem,
        plotted_values=plotted,
        references=references,
    )


def _numeric_leaves(node: Any) -> tuple[float, ...]:
    """Every finite number inside nested containers, in order."""

    if isinstance(node, bool):
        return ()
    if isinstance(node, (int, float)):
        return () if math.isnan(float(node)) else (float(node),)
    if isinstance(node, str):
        return ()
    if isinstance(node, Mapping):
        return tuple(
            value for item in node.values() for value in _numeric_leaves(item)
        )
    if isinstance(node, (tuple, list)):
        return tuple(value for item in node for value in _numeric_leaves(item))
    return ()


def _reference_list(reading: Interpretation) -> tuple[str, ...]:
    """Source cells behind a reading, including sample-based ones."""

    collected = list(reading.all_references())
    for key in ("sample_references", "x_references", "y_references", "count_references"):
        payload = reading.extras.get(key)
        if payload is None:
            continue
        if isinstance(payload, (tuple, list)):
            for item in payload:
                if isinstance(item, (tuple, list)):
                    collected.extend(str(value) for value in item if value)
                elif item:
                    collected.append(str(item))
    return tuple(dict.fromkeys(item for item in collected if item))


def _slug(text: str) -> str:
    cleaned = "".join(
        character if character.isalnum() else "_" for character in text.lower()
    )
    return "_".join(part for part in cleaned.split("_") if part) or "table"
