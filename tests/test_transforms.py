from __future__ import annotations

import pytest

from sft_tokenization.transforms import (
    DefaultChosenTransform,
    DefaultCompletionTransform,
    DefaultPromptTransform,
    DefaultRejectedTransform,
    TemplatePromptTransform,
    load_transform,
    template_columns,
)


def test_default_transforms_stringify_values():
    assert DefaultPromptTransform()({"prompt": ["x", None, 7]}) == ["x", "", "7"]
    assert DefaultCompletionTransform()({"completion": [False]}) == ["False"]
    assert DefaultChosenTransform()({"chosen": [1]}) == ["1"]
    assert DefaultRejectedTransform()({"rejected": [None]}) == [""]


def test_load_transform_from_dotted_path():
    obj = load_transform("sft_tokenization.transforms.DefaultPromptTransform")
    assert isinstance(obj, DefaultPromptTransform)


def test_template_columns_preserve_first_occurrence_order():
    assert template_columns("{a} {b} {a} {c}") == ["a", "b", "c"]


def test_template_prompt_transform():
    class ExampleTemplate(TemplatePromptTransform):
        template = "Question: {question}\nContext: {context}"

    transform = ExampleTemplate()
    assert transform({"question": ["Q", None], "context": ["C", 3]}) == [
        "Question: Q\nContext: C",
        "Question: \nContext: 3",
    ]


def test_template_prompt_transform_rejects_misaligned_columns():
    class ExampleTemplate(TemplatePromptTransform):
        template = "{a} {b}"

    with pytest.raises(ValueError):
        ExampleTemplate()({"a": [1, 2], "b": [1]})
