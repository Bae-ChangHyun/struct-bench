from __future__ import annotations

import importlib
import inspect
import pkgutil

from pydantic import BaseModel

_schema_registry: dict[str, type[BaseModel]] = {}


def _discover_schemas():
    package = importlib.import_module("app.schemas")
    for importer, modname, ispkg in pkgutil.iter_modules(
        package.__path__, package.__name__ + "."
    ):
        module = importlib.import_module(modname)
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, BaseModel) and obj is not BaseModel:
                _schema_registry[name] = obj


_discover_schemas()


def get_schema(name: str) -> type[BaseModel]:
    if name not in _schema_registry:
        available = ", ".join(sorted(_schema_registry.keys()))
        raise KeyError(f"Schema '{name}' not found. Available: {available}")
    return _schema_registry[name]


def list_schemas() -> list[str]:
    return sorted(_schema_registry.keys())
