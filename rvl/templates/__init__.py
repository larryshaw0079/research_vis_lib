"""Chart templates.

Every module here exposes ``SPEC``, ``PALETTES``, a data class with a ``from_*``
builder, ``DEFAULT_DATA``, ``ChartStyle``, ``DEFAULT_STYLE``, ``create_figure``
and ``main``. See ``references/template-contract.md``.

Modules are discovered automatically by :mod:`rvl.registry`, so this file
deliberately imports nothing: adding a template must not require editing it.
"""
