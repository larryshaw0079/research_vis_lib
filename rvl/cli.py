"""Command-line interface: inspect data, pick a template, emit code, verify it.

Typical run:

    python -m rvl recommend results.xlsx
    python -m rvl generate results.xlsx --template grouped-ring-bar -o figure.py
    python -m rvl verify figure.py --source results.xlsx
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from .contract import SUPPORTED_FORMATS


def _add_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "data",
        nargs="+",
        type=Path,
        help="one or more data files (.xlsx .xls .csv .tsv .json .jsonl .md)",
    )
    parser.add_argument(
        "--table",
        default=None,
        help="name of the sheet or table to use when a file holds several",
    )


def _resolve_tables(paths: Sequence[Path], name: str | None):
    from .ingest import read_many, read_tables

    tables = list(read_many(paths)) if len(paths) > 1 else list(read_tables(paths[0]))
    if name is not None:
        matched = [table for table in tables if table.name == name]
        if not matched:
            available = ", ".join(table.name for table in tables)
            raise SystemExit(f"no table named {name!r}; available: {available}")
        return matched
    return tables


def _command_templates(args: argparse.Namespace) -> int:
    from .registry import load_registry

    registry = load_registry()
    if args.json:
        payload = [
            {
                **{
                    key: value
                    for key, value in asdict(spec).items()
                    if key not in {"affinities", "requires", "argument_names"}
                },
                "kinds": [kind.value for kind in spec.kinds],
                "geometry": spec.geometry.value,
                "categories": spec.categories.describe(),
                "series": spec.series.describe(),
            }
            for spec in registry
        ]
        print(json.dumps(payload, indent=2, default=str))
        return 0

    print(f"{len(registry)} chart templates\n")
    for spec in registry:
        print(f"{spec.template_id}  ({spec.palette_count} palettes)")
        print(f"  {spec.title}")
        print(f"  data:     {', '.join(kind.value for kind in spec.kinds)}")
        print(f"  shape:    {spec.describe_shape()}")
        print(f"  builder:  {spec.builder}")
        if args.verbose:
            print(f"  contract: {spec.data_contract}")
            for item in spec.good_for:
                print(f"  good for: {item}")
            for item in spec.avoid_when:
                print(f"  avoid:    {item}")
            if spec.reference:
                print(f"  source:   {spec.reference}")
        print()
    return 0


def _command_profile(args: argparse.Namespace) -> int:
    from .profiling import profile_table

    tables = _resolve_tables(args.data, args.table)
    for index, table in enumerate(tables):
        if index:
            print()
        print(profile_table(table).summary())
    return 0


def _command_recommend(args: argparse.Namespace) -> int:
    from .profiling import profile_table
    from .recommend import recommend

    tables = _resolve_tables(args.data, args.table)
    for index, table in enumerate(tables):
        if index:
            print("\n" + "=" * 72 + "\n")
        result = recommend(profile_table(table))
        print(result.describe(limit=args.limit))
    return 0


def _command_generate(args: argparse.Namespace) -> int:
    from .codegen import generate
    from .profiling import profile_table
    from .recommend import recommend

    tables = _resolve_tables(args.data, args.table)
    if len(tables) > 1 and args.table is None:
        names = ", ".join(table.name for table in tables)
        print(
            f"note: {len(tables)} tables found ({names}); using the largest. "
            "Pass --table to choose another.",
            file=sys.stderr,
        )
        tables = [max(tables, key=lambda item: item.n_rows * item.n_columns)]

    profile = profile_table(tables[0])
    result = recommend(profile)
    if args.template:
        candidate = result.for_template(args.template)
        if candidate is None:
            available = ", ".join(item.template_id for item in result.candidates)
            raise SystemExit(
                f"template {args.template!r} cannot draw this data. "
                f"Applicable templates: {available or 'none'}"
            )
    else:
        candidate = result.best
        if candidate is None:
            raise SystemExit(
                "no template fits this data:\n" + profile.summary()
            )

    script = generate(
        candidate,
        palette=args.palette,
        dpi=args.dpi,
        formats=tuple(args.formats),
        output_dir=str(args.figure_dir),
    )
    if args.output:
        path = script.write(args.output)
        print(f"wrote {path}")
        print(f"template: {script.template_id}")
        print(f"reading:  {candidate.interpretation.describe()}")
        print(f"values:   {len(script.plotted_values)} numbers from {len(script.references)} cells")
    else:
        print(script.code, end="")
    return 0


def _command_verify(args: argparse.Namespace) -> int:
    from .verify import verify

    report = verify(args.script, args.source, run=not args.no_run)
    print(report.render())
    return 0 if report.ok else 1


def _command_render(args: argparse.Namespace) -> int:
    from .registry import load_registry

    registry = load_registry()
    try:
        module = registry.module(args.template)
    except KeyError as exc:
        raise SystemExit(str(exc)) from exc
    forwarded = list(args.template_args)
    return int(module.main(forwarded))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="research-vis-lib",
        description=(
            "Pick a scientific chart template for an experimental dataset and "
            "generate the Python that draws it."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    templates = subparsers.add_parser(
        "templates", help="list the available chart templates"
    )
    templates.add_argument("--json", action="store_true", help="machine-readable output")
    templates.add_argument(
        "-v", "--verbose", action="store_true", help="include guidance and provenance"
    )
    templates.set_defaults(handler=_command_templates)

    profile = subparsers.add_parser(
        "profile", help="describe a data file and how it can be read"
    )
    _add_data_arguments(profile)
    profile.set_defaults(handler=_command_profile)

    recommend_parser = subparsers.add_parser(
        "recommend", help="rank templates against a data file"
    )
    _add_data_arguments(recommend_parser)
    recommend_parser.add_argument(
        "--limit", type=int, default=5, help="how many candidates to show (default: 5)"
    )
    recommend_parser.set_defaults(handler=_command_recommend)

    generate_parser = subparsers.add_parser(
        "generate", help="emit a standalone script that plots a data file"
    )
    _add_data_arguments(generate_parser)
    generate_parser.add_argument(
        "--template",
        default=None,
        help="template id; defaults to the top recommendation",
    )
    generate_parser.add_argument(
        "-o", "--output", type=Path, default=None, help="write the script here"
    )
    generate_parser.add_argument(
        "--palette", default="1", help="palette number or name (default: 1)"
    )
    generate_parser.add_argument("--dpi", type=int, default=None, help="raster DPI")
    generate_parser.add_argument(
        "--formats",
        nargs="+",
        default=["png"],
        metavar="FORMAT",
        help=f"one or more of: {' '.join(SUPPORTED_FORMATS)}",
    )
    generate_parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("output"),
        help="where the generated script writes its figure (default: output)",
    )
    generate_parser.set_defaults(handler=_command_generate)

    verify_parser = subparsers.add_parser(
        "verify", help="check a generated script against its source data"
    )
    verify_parser.add_argument("script", type=Path)
    verify_parser.add_argument(
        "--source",
        type=Path,
        nargs="+",
        required=True,
        help="the data file(s) the script was generated from",
    )
    verify_parser.add_argument(
        "--no-run", action="store_true", help="skip executing the script"
    )
    verify_parser.set_defaults(handler=_command_verify)

    render_parser = subparsers.add_parser(
        "render", help="render a template's reference figure and palettes"
    )
    render_parser.add_argument("template")
    render_parser.add_argument(
        "template_args",
        nargs=argparse.REMAINDER,
        help="flags forwarded to the template CLI, e.g. --palette all",
    )
    render_parser.set_defaults(handler=_command_render)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    try:
        return int(args.handler(args))
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
