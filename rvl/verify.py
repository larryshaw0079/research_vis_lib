"""Check that a generated script draws the source data and nothing else.

Three failure modes matter and each has a check:

1. **Invented numbers.** Every numeric literal in the script must appear in the
   source file. A value that does not is either a typo or a hallucination.
2. **Leftover demo data.** Templates ship reference data digitised from published
   figures. A script that still carries those numbers, or that imports
   ``DEFAULT_DATA``, would silently plot somebody else's experiment.
3. **Code that does not run.** The script is executed and the figure it claims to
   write must exist afterwards.
"""

from __future__ import annotations

import ast
import importlib
import math
import subprocess
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Final, Iterable, Sequence

import numpy as np

from .contract import is_presentation_argument
from .ingest import ColumnKind, Table, read_tables
from .registry import load_registry

_TOLERANCE: Final[float] = 1e-9
_STALE_MATCH_THRESHOLD: Final[int] = 3


class Level(StrEnum):
    OK = "ok"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class Finding:
    level: Level
    check: str
    message: str

    def render(self) -> str:
        marker = {Level.OK: "ok  ", Level.WARNING: "warn", Level.ERROR: "FAIL"}[self.level]
        return f"[{marker}] {self.check}: {self.message}"


@dataclass(frozen=True, slots=True)
class VerificationReport:
    script: Path
    findings: tuple[Finding, ...]
    rendered: tuple[Path, ...] = ()

    @property
    def ok(self) -> bool:
        return not any(finding.level is Level.ERROR for finding in self.findings)

    @property
    def errors(self) -> tuple[Finding, ...]:
        return tuple(item for item in self.findings if item.level is Level.ERROR)

    @property
    def warnings(self) -> tuple[Finding, ...]:
        return tuple(item for item in self.findings if item.level is Level.WARNING)

    def render(self) -> str:
        lines = [f"verifying {self.script}"]
        lines.extend(finding.render() for finding in self.findings)
        if self.rendered:
            lines.append("figures: " + ", ".join(str(path) for path in self.rendered))
        lines.append("RESULT: " + ("pass" if self.ok else "fail"))
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class ScriptFacts:
    """What static analysis found in a generated script."""

    numbers: tuple[float, ...]
    """Numbers passed to data-bearing builder arguments. These must be traceable."""

    strings: tuple[str, ...]
    """Strings passed to data-bearing builder arguments, i.e. axis labels."""

    builder_calls: tuple[str, ...]
    imported_names: tuple[str, ...]
    template_modules: tuple[str, ...]
    data_keywords: tuple[str, ...] = ()


def _numeric_value(node: ast.AST) -> float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        if isinstance(node.value, bool):
            return None
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _numeric_value(node.operand)
        return None if inner is None else -inner
    return None


def _literals_in(node: ast.AST) -> tuple[list[float], list[str]]:
    """Numeric and string literals under ``node``.

    The traversal is explicit rather than :func:`ast.walk` because a negative
    literal is a unary minus wrapping a positive constant. Walking would yield both
    ``-145.5`` and ``145.5``, and the phantom positive would be reported as a value
    that does not appear in the source.
    """

    numbers: list[float] = []
    strings: list[str] = []
    pending: list[ast.AST] = [node]
    while pending:
        current = pending.pop()
        value = _numeric_value(current)
        if value is not None:
            numbers.append(value)
            continue
        if isinstance(current, ast.Constant) and isinstance(current.value, str):
            strings.append(current.value)
            continue
        pending.extend(ast.iter_child_nodes(current))
    return numbers, strings


def inspect_script(path: Path) -> ScriptFacts:
    """Collect the data literals, builder calls and imports of a script.

    Only literals passed to a builder's data-bearing keyword arguments are
    collected as measurements. A DPI, a palette index and an axis title are all
    numbers or strings in the file, but none of them come from the data.
    """

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    builder_names = {spec.builder_name for spec in load_registry()}

    numbers: list[float] = []
    strings: list[str] = []
    builders: list[str] = []
    imported: list[str] = []
    modules: list[str] = []
    keywords: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            owner = node.func.value
            if isinstance(owner, ast.Name):
                builders.append(f"{owner.id}.{node.func.attr}")
            if node.func.attr in builder_names:
                for keyword in node.keywords:
                    if keyword.arg is None or is_presentation_argument(keyword.arg):
                        continue
                    keywords.append(keyword.arg)
                    found_numbers, found_strings = _literals_in(keyword.value)
                    numbers.extend(found_numbers)
                    strings.extend(found_strings)
                for positional in node.args:
                    found_numbers, found_strings = _literals_in(positional)
                    numbers.extend(found_numbers)
                    strings.extend(found_strings)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                modules.append(node.module)
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)

    return ScriptFacts(
        numbers=tuple(numbers),
        strings=tuple(strings),
        builder_calls=tuple(dict.fromkeys(builders)),
        imported_names=tuple(dict.fromkeys(imported)),
        template_modules=tuple(
            name for name in dict.fromkeys(modules) if name.startswith("rvl.templates.")
        ),
        data_keywords=tuple(dict.fromkeys(keywords)),
    )


def _source_numbers(tables: Sequence[Table]) -> np.ndarray:
    pool: list[float] = []
    for table in tables:
        for column in table.columns:
            for measurement in column.measurements():
                if measurement is None:
                    continue
                pool.append(measurement.value)
                if measurement.error is not None:
                    pool.append(measurement.error)
    return np.sort(np.asarray(pool, dtype=float)) if pool else np.asarray([], dtype=float)


def _aggregate_numbers(tables: Sequence[Table]) -> np.ndarray:
    """Totals a template may legitimately derive from the source.

    Some builders take an aggregate rather than a cell: the centre composition of a
    concentric chart is the column total of its rings. That is derived, not
    invented, so it is checked against these sums instead of being rejected.
    """

    pool: list[float] = []
    for table in tables:
        numeric = [column for column in table.columns if column.kind == ColumnKind.NUMERIC]
        if not numeric:
            continue
        matrix = np.column_stack([column.numeric() for column in numeric])
        with np.errstate(invalid="ignore"):
            pool.extend(float(value) for value in np.nansum(matrix, axis=0))
            pool.extend(float(value) for value in np.nansum(matrix, axis=1))
            pool.append(float(np.nansum(matrix)))
        for label_column in table.columns:
            if label_column.kind not in {ColumnKind.CATEGORICAL, ColumnKind.TEMPORAL}:
                continue
            labels = label_column.labels()
            for group in dict.fromkeys(label for label in labels if label):
                rows = [index for index, label in enumerate(labels) if label == group]
                for column in numeric:
                    values = column.numeric()[rows]
                    values = values[np.isfinite(values)]
                    if values.size:
                        pool.append(float(values.sum()))
                        pool.append(float(values.mean()))
    finite = [value for value in pool if math.isfinite(value)]
    return np.sort(np.asarray(finite, dtype=float)) if finite else np.asarray([], dtype=float)


def _source_strings(tables: Sequence[Table]) -> set[str]:
    found: set[str] = set()
    for table in tables:
        found.add(table.name)
        for column in table.columns:
            found.add(column.name)
            found.update(label for label in column.labels() if label)
    return {item.strip() for item in found if item and item.strip()}


def _matches(haystack: np.ndarray, needle: float) -> bool:
    if haystack.size == 0 or not math.isfinite(needle):
        return False
    position = int(np.searchsorted(haystack, needle))
    tolerance = max(_TOLERANCE, _TOLERANCE * abs(needle))
    for index in (position - 1, position, position + 1):
        if 0 <= index < haystack.size and abs(haystack[index] - needle) <= tolerance:
            return True
    # Fall back to a rounded comparison so a source cell of 0.4470000001 still
    # matches a script literal of 0.447.
    rounded = np.round(haystack, 10)
    return bool(np.any(np.abs(rounded - round(needle, 10)) <= tolerance))


def check_provenance(facts: ScriptFacts, tables: Sequence[Table]) -> list[Finding]:
    """Every value passed to a data argument must exist in the source."""

    haystack = _source_numbers(tables)
    if haystack.size == 0:
        return [
            Finding(
                Level.ERROR,
                "provenance",
                "the source file yielded no numeric cells to check against",
            )
        ]

    unmatched = [value for value in facts.numbers if not _matches(haystack, value)]
    derived: list[float] = []
    missing: list[float] = []
    if unmatched:
        aggregates = _aggregate_numbers(tables)
        for value in unmatched:
            (derived if _matches(aggregates, value) else missing).append(value)
    checked = len(facts.numbers)

    findings: list[Finding] = []
    if not checked:
        findings.append(
            Finding(
                Level.WARNING,
                "provenance",
                "no measurements were passed to the builder's data arguments; "
                f"confirm the data was inlined (data arguments seen: "
                f"{', '.join(facts.data_keywords) or 'none'})",
            )
        )
    elif missing:
        preview = ", ".join(f"{value:g}" for value in sorted(set(missing))[:10])
        findings.append(
            Finding(
                Level.ERROR,
                "provenance",
                f"{len(set(missing))} of {checked} plotted values do not appear in "
                f"the source: {preview}",
            )
        )
    elif derived:
        findings.append(
            Finding(
                Level.OK,
                "provenance",
                f"{checked - len(derived)} of {checked} plotted values trace back to a "
                f"source cell; the other {len(derived)} are totals derived from them",
            )
        )
    else:
        findings.append(
            Finding(
                Level.OK,
                "provenance",
                f"all {checked} plotted values trace back to a source cell",
            )
        )

    source_text = _source_strings(tables)
    unknown = [
        text for text in facts.strings if text.strip() and text.strip() not in source_text
    ]
    if unknown:
        findings.append(
            Finding(
                Level.WARNING,
                "labels",
                f"{len(set(unknown))} label(s) passed as data are not column names or "
                f"cell values from the source: {', '.join(sorted(set(unknown))[:6])}",
            )
        )
    return findings


def check_no_demo_data(facts: ScriptFacts, tables: Sequence[Table]) -> list[Finding]:
    """The script must not carry a template's reference data."""

    findings: list[Finding] = []
    if "DEFAULT_DATA" in facts.imported_names:
        findings.append(
            Finding(
                Level.ERROR,
                "demo-data",
                "the script imports DEFAULT_DATA, so it would plot the template's "
                "reference figure instead of the experiment",
            )
        )

    registry = load_registry()
    haystack = _source_numbers(tables)
    script_numbers = {round(value, 10) for value in facts.numbers}
    if not script_numbers:
        return findings

    for module_name in facts.template_modules or tuple(
        spec.module_name for spec in registry
    ):
        try:
            module = importlib.import_module(module_name)
        except Exception:  # pragma: no cover - a broken template is caught elsewhere
            continue
        demo = _collect_numbers(getattr(module, "DEFAULT_DATA", None))
        if not demo:
            continue
        shared = {
            value
            for value in script_numbers & {round(item, 10) for item in demo}
            if not _matches(haystack, value)
        }
        if len(shared) >= _STALE_MATCH_THRESHOLD:
            preview = ", ".join(f"{value:g}" for value in sorted(shared)[:6])
            findings.append(
                Finding(
                    Level.ERROR,
                    "demo-data",
                    f"{len(shared)} values match {module_name}.DEFAULT_DATA but not "
                    f"the source file: {preview}",
                )
            )

    if not findings:
        findings.append(
            Finding(
                Level.OK,
                "demo-data",
                "no template reference values leaked into the script",
            )
        )
    return findings


def _collect_numbers(node: Any, depth: int = 0) -> list[float]:
    """Every float reachable inside a template's data object."""

    if depth > 6 or node is None:
        return []
    if isinstance(node, bool):
        return []
    if isinstance(node, (int, float)):
        return [] if math.isnan(float(node)) else [float(node)]
    if isinstance(node, np.ndarray):
        return [float(value) for value in node.flatten() if math.isfinite(value)]
    if isinstance(node, str):
        return []
    if isinstance(node, dict):
        found: list[float] = []
        for value in node.values():
            found.extend(_collect_numbers(value, depth + 1))
        return found
    if isinstance(node, (list, tuple, set, frozenset)):
        found = []
        for value in node:
            found.extend(_collect_numbers(value, depth + 1))
        return found
    fields = getattr(node, "__dataclass_fields__", None)
    if fields:
        found = []
        for name in fields:
            found.extend(_collect_numbers(getattr(node, name, None), depth + 1))
        return found
    return []


def check_structure(facts: ScriptFacts) -> list[Finding]:
    registry = load_registry()
    builders = {f"{spec.data_class}.{spec.builder_name}" for spec in registry}
    used = [call for call in facts.builder_calls if call in builders]
    if not used:
        return [
            Finding(
                Level.ERROR,
                "structure",
                "the script never calls a template's from_* builder; data must be "
                "constructed through the builder so it gets validated",
            )
        ]
    if len(used) > 1:
        return [
            Finding(
                Level.WARNING,
                "structure",
                f"the script calls several builders ({', '.join(used)}); one figure "
                "per script keeps verification meaningful",
            )
        ]
    return [Finding(Level.OK, "structure", f"builds its data through {used[0]}")]


def run_script(script: Path, *, timeout: int = 300) -> tuple[list[Finding], list[Path]]:
    """Execute the script and collect the figures it reports writing."""

    repo_root = Path(__file__).resolve().parent.parent
    environment = {
        "MPLCONFIGDIR": str(repo_root / ".mplcache"),
        "PYTHONPATH": str(repo_root),
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "HOME": str(Path.home()),
    }
    try:
        completed = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=repo_root,
            env=environment,
        )
    except subprocess.TimeoutExpired:
        return (
            [Finding(Level.ERROR, "render", f"the script did not finish in {timeout}s")],
            [],
        )

    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout).strip().splitlines()
        detail = tail[-1] if tail else "no output"
        return (
            [
                Finding(
                    Level.ERROR,
                    "render",
                    f"the script exited with code {completed.returncode}: {detail}",
                )
            ],
            [],
        )

    written: list[Path] = []
    for line in completed.stdout.splitlines():
        candidate = repo_root / line.strip() if not line.startswith("/") else Path(line.strip())
        if candidate.suffix.lower() in {".png", ".svg", ".pdf"} and candidate.exists():
            written.append(candidate)

    if not written:
        return (
            [
                Finding(
                    Level.ERROR,
                    "render",
                    "the script ran but no figure file was found; it should print the "
                    "paths it writes",
                )
            ],
            [],
        )
    empty = [path for path in written if path.stat().st_size == 0]
    if empty:
        return (
            [
                Finding(
                    Level.ERROR,
                    "render",
                    f"wrote empty figure file(s): {', '.join(str(item) for item in empty)}",
                )
            ],
            written,
        )
    sizes = ", ".join(f"{path.name} ({path.stat().st_size // 1024} KB)" for path in written)
    return [Finding(Level.OK, "render", f"rendered {sizes}")], written


def verify(
    script: str | Path,
    sources: Iterable[str | Path],
    *,
    run: bool = True,
) -> VerificationReport:
    """Run every fidelity check on a generated script."""

    script_path = Path(script)
    if not script_path.exists():
        raise FileNotFoundError(f"no such script: {script_path}")

    tables: list[Table] = []
    for source in sources:
        tables.extend(read_tables(source))

    facts = inspect_script(script_path)
    findings: list[Finding] = []
    findings.extend(check_structure(facts))
    findings.extend(check_provenance(facts, tables))
    findings.extend(check_no_demo_data(facts, tables))

    rendered: list[Path] = []
    if run:
        render_findings, rendered = run_script(script_path)
        findings.extend(render_findings)

    return VerificationReport(
        script=script_path,
        findings=tuple(findings),
        rendered=tuple(rendered),
    )
