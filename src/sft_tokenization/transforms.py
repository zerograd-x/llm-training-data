from __future__ import annotations

import importlib
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

from .types import ColumnBatch


def _to_text(value: Any) -> str:
    return "" if value is None else str(value)


class PromptRenderer(ABC):
    @abstractmethod
    def __call__(self, batch: ColumnBatch) -> list[str]: ...


class CompletionRenderer(ABC):
    @abstractmethod
    def __call__(self, batch: ColumnBatch) -> list[str]: ...


class ChosenResponseRenderer(ABC):
    @abstractmethod
    def __call__(self, batch: ColumnBatch) -> list[str]: ...


class RejectedResponseRenderer(ABC):
    @abstractmethod
    def __call__(self, batch: ColumnBatch) -> list[str]: ...


class DefaultPromptRenderer(PromptRenderer):
    def __call__(self, batch: ColumnBatch) -> list[str]:
        return [_to_text(value) for value in batch["prompt"]]


class DefaultCompletionRenderer(CompletionRenderer):
    def __call__(self, batch: ColumnBatch) -> list[str]:
        return [_to_text(value) for value in batch["completion"]]


class DefaultChosenResponseRenderer(ChosenResponseRenderer):
    def __call__(self, batch: ColumnBatch) -> list[str]:
        return [_to_text(value) for value in batch["chosen"]]


class DefaultRejectedResponseRenderer(RejectedResponseRenderer):
    def __call__(self, batch: ColumnBatch) -> list[str]:
        return [_to_text(value) for value in batch["rejected"]]


def load_renderer(dotted_path: str) -> Any:
    """Instantiate a zero-argument renderer class from a dotted import path."""
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls()


_PLACEHOLDER = re.compile(r"\{(\w+)\}")


def extract_template_fields(template: str) -> list[str]:
    """Return unique template field names in first-occurrence order."""
    seen: list[str] = []
    for name in _PLACEHOLDER.findall(template):
        if name not in seen:
            seen.append(name)
    return seen


class TemplatePromptRenderer(PromptRenderer):
    """Small, deterministic ``{column}`` prompt renderer.

    Subclasses may set either ``template`` or ``template_file``. A template
    file is resolved relative to the subclass module's directory, so packaged
    resources can live next to the renderer implementation.

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
            raise ValueError("TemplatePromptRenderer requires template or template_file")

        if not text.strip():
            raise ValueError("Prompt template must not be empty or whitespace-only")

        fields = extract_template_fields(text)
        if not fields:
            raise ValueError("Prompt template must contain at least one {column} placeholder")

        self._template = text
        self._fields = fields

    def __call__(self, batch: ColumnBatch) -> list[str]:
        row_count = len(batch[self._fields[0]])
        for field in self._fields[1:]:
            if len(batch[field]) != row_count:
                raise ValueError(f"Template field {field!r} has a different row count")

        rows: list[str] = []
        for index in range(row_count):
            rows.append(
                _PLACEHOLDER.sub(
                    lambda match: _to_text(batch[match.group(1)][index]),
                    self._template,
                )
            )
        return rows


# Backward-compatible aliases. Prefer the renderer vocabulary in new code.
PromptTransform = PromptRenderer
CompletionTransform = CompletionRenderer
ChosenTransform = ChosenResponseRenderer
RejectedTransform = RejectedResponseRenderer
DefaultPromptTransform = DefaultPromptRenderer
DefaultCompletionTransform = DefaultCompletionRenderer
DefaultChosenTransform = DefaultChosenResponseRenderer
DefaultRejectedTransform = DefaultRejectedResponseRenderer
TemplatePromptTransform = TemplatePromptRenderer
load_transform = load_renderer
template_columns = extract_template_fields
