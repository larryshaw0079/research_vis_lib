"""research-vis-lib: pick a scientific chart template for a dataset and emit code.

Public surface:

- :mod:`rvl.ingest` reads xlsx/xls/csv/tsv/json/jsonl/markdown into a ``Table``
  that remembers where every number came from.
- :mod:`rvl.profiling` summarises a ``Table`` into a ``DataProfile``.
- :mod:`rvl.recommend` ranks templates against a profile.
- :mod:`rvl.codegen` writes a standalone script for the chosen template.
- :mod:`rvl.verify` renders that script and checks it against the source data.
- :mod:`rvl.registry` discovers the templates in :mod:`rvl.templates`.
"""

from __future__ import annotations

__all__ = [
    "__version__",
    "load_registry",
]

__version__ = "0.2.0"


def __getattr__(name: str) -> object:
    if name == "load_registry":
        from .registry import load_registry

        return load_registry
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
