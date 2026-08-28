from __future__ import annotations

import importlib
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

from .types import ColumnBatch


def _to_text(value: Any) -> str:
    return "" if value is None else str(value)


class PromptTransform(ABC):
    @abstractmethod
    def __call__(self, batch: ColumnBatch) -> list[str]: ...


class CompletionTransform(ABC):
    @abstractmethod
    def __call__(self, batch: ColumnBatch) -> list[str]: ...


class ChosenTransform(ABC):
    @abstractmethod
    def __call__(self, batch: ColumnBatch) -> list[str]: ...


class RejectedTransform(ABC):
    @abstractmethod
    def __call__(self, batch: ColumnBatch) -> list[str]: ...


class DefaultPromptTransform(PromptTransform):
    def __call__(self, batch: ColumnBatch) -> list[str]:
        return [_to_text(value) for value in batch["prompt"]]


class DefaultCompletionTransform(CompletionTransform):
    def __call__(self, batch: ColumnBatch) -> list[str]:
        return [_to_text(value) for value in batch["completion"]]


class DefaultChosenTransform(ChosenTransform):
    def __call__(self, batch: ColumnBatch) -> list[str]:
        return [_to_text(value) for value in batch["chosen"]]


class DefaultRejectedTransform(RejectedTransform):
    def __call__(self, batch: ColumnBatch) -> list[str]:
        return [_to_text(value) for value in batch["rejected"]]


def load_transform(dotted_path: str) -> Any:
    """Instantiate a zero-argument class from a dotted import path."""
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls()


_PLACEHOLDER = re.compile(r"\{(\w+)\}")


def template_columns(template: str) -> list[str]:
    """Return unique placeholders in first-occurrence order."""
    seen: list[str] = []
    for name in _PLACEHOLDER.findall(template):
        if name not in seen:
            seen.append(name)
    return seen


class TemplatePromptTransform(PromptTransform):
    """Small, deterministic ``{column}`` template renderer.

    Subclasses may set either ``template`` or ``template_file``. A template
    file is resolved relative to the subclass module's directory, so packaged
    resources can live next to the transform implementation.

    This intentionally is *not* ``str.format`` or Jinja: only ``{word}``
    placeholders are substituted.
    """

    template: ClassVar[str | None] = None
    template_file: ClassVar[str | None] = None

    def __init__(self) -> None:
        if self.template is not None and self.template_file is not None:
            raise ValueError("Set only one of template or template_file")
        if self.template is not None:
            text = self.template
        elif self.template_file is not None:
            module = importlib.import_module(self.__class__.__module__)
            module_file = getattr(module, "__file__", None)
            if module_file is None:
                raise ValueError("Cannot resolve template_file for a module without __file__")
            text = (Path(module_file).resolve().parent / self.template_file).read_text()
        else:
            raise ValueError("TemplatePromptTransform requires template or template_file")

        if not text.strip():
            raise ValueError("Prompt template must not be empty or whitespace-only")

        columns = template_columns(text)
        if not columns:
            raise ValueError("Prompt template must contain at least one {column} placeholder")

        self._template = text
        self._columns = columns

    def __call__(self, batch: ColumnBatch) -> list[str]:
        row_count = len(batch[self._columns[0]])
        for column in self._columns[1:]:
            if len(batch[column]) != row_count:
                raise ValueError(f"Template column {column!r} has a different row count")

        rows: list[str] = []
        for index in range(row_count):
            rows.append(
                _PLACEHOLDER.sub(
                    lambda match: _to_text(batch[match.group(1)][index]),
                    self._template,
                )
            )
        return rows
