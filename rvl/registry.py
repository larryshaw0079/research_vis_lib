"""Auto-discovering registry of chart templates.

Adding a template means dropping a module into ``rvl/templates/`` that exposes a
``SPEC``. Nothing else in the codebase needs editing: the registry imports every
module in that package, reads its ``SPEC``, and the recommender, code generator
and CLI pick it up from there.
"""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass, replace
from functools import lru_cache
from types import ModuleType
from typing import Iterator

from . import templates as templates_package
from .contract import DataKind, TemplateSpec

_REQUIRED_ATTRIBUTES: tuple[str, ...] = (
    "SPEC",
    "PALETTES",
    "DEFAULT_DATA",
    "DEFAULT_STYLE",
    "create_figure",
    "main",
)


class TemplateError(RuntimeError):
    """A template module does not satisfy the contract."""


@dataclass(frozen=True, slots=True)
class Registry:
    """Every discovered template, keyed by ``template_id``."""

    specs: tuple[TemplateSpec, ...]

    def __len__(self) -> int:
        return len(self.specs)

    def __iter__(self) -> Iterator[TemplateSpec]:
        return iter(self.specs)

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(spec.template_id for spec in self.specs)

    def get(self, template_id: str) -> TemplateSpec:
        wanted = template_id.strip().lower().replace("_", "-")
        for spec in self.specs:
            if spec.template_id == wanted:
                return spec
        raise KeyError(
            f"unknown template {template_id!r}; available: {', '.join(self.ids)}"
        )

    def module(self, template_id: str) -> ModuleType:
        """Import and return the module that draws ``template_id``."""

        return importlib.import_module(self.get(template_id).module_name)

    def by_kind(self, kind: DataKind) -> tuple[TemplateSpec, ...]:
        return tuple(spec for spec in self.specs if kind in spec.kinds)

    def matching(
        self, *, kind: DataKind, categories: int, series: int
    ) -> tuple[TemplateSpec, ...]:
        return tuple(
            spec
            for spec in self.by_kind(kind)
            if spec.accepts_shape(categories=categories, series=series)
        )


def _validate_module(module: ModuleType, spec: TemplateSpec) -> None:
    missing = [name for name in _REQUIRED_ATTRIBUTES if not hasattr(module, name)]
    if missing:
        raise TemplateError(
            f"{module.__name__} is missing required attribute(s): {', '.join(missing)}"
        )
    data_class = getattr(module, spec.data_class, None)
    if data_class is None:
        raise TemplateError(
            f"{module.__name__} declares builder {spec.builder!r} but has no "
            f"class {spec.data_class!r}"
        )
    if not callable(getattr(data_class, spec.builder_name, None)):
        raise TemplateError(
            f"{spec.data_class} in {module.__name__} has no callable "
            f"{spec.builder_name!r} named by SPEC.builder"
        )
    if not module.PALETTES:
        raise TemplateError(f"{module.__name__} defines no palettes")


@lru_cache(maxsize=1)
def load_registry() -> Registry:
    """Import every template module and collect the validated specs."""

    discovered: list[TemplateSpec] = []
    seen: dict[str, str] = {}
    for info in sorted(
        pkgutil.iter_modules(templates_package.__path__), key=lambda item: item.name
    ):
        if info.name.startswith("_"):
            continue
        module_name = f"{templates_package.__name__}.{info.name}"
        module = importlib.import_module(module_name)
        spec = getattr(module, "SPEC", None)
        if spec is None:
            raise TemplateError(f"{module_name} does not define SPEC")
        if not isinstance(spec, TemplateSpec):
            raise TemplateError(f"{module_name}.SPEC is not a TemplateSpec")
        _validate_module(module, spec)
        if spec.template_id in seen:
            raise TemplateError(
                f"template id {spec.template_id!r} is declared by both "
                f"{seen[spec.template_id]} and {module_name}"
            )
        seen[spec.template_id] = module_name
        discovered.append(
            replace(
                spec,
                module=module_name,
                palette_count=len(module.PALETTES),
            )
        )
    if not discovered:
        raise TemplateError("no chart templates found in rvl/templates")
    return Registry(specs=tuple(discovered))


def get_module(template_id: str) -> ModuleType:
    return load_registry().module(template_id)


def get_spec(template_id: str) -> TemplateSpec:
    return load_registry().get(template_id)
