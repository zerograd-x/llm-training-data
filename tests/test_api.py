from __future__ import annotations

from pathlib import Path

import pytest

from llm_training_data import (
    ChatPromptFormatter,
    DefaultCompletionRenderer,
    DefaultPromptRenderer,
    SFTDataConfig,
    ShortestFirstSequencePacker,
    TemplatePromptRenderer,
    build_packed_sft_collator,
    build_sft_collator,
    build_training_example,
    extract_template_fields,
    read_system_prompt_metadata,
    resolve_system_prompt,
    write_system_prompt_metadata,
)


def test_default_renderers():
    batch = {"prompt": ["a", None], "completion": ["x", 3]}
    assert DefaultPromptRenderer()(batch) == ["a", ""]
    assert DefaultCompletionRenderer()(batch) == ["x", "3"]


def test_template_prompt_renderer():
    class Renderer(TemplatePromptRenderer):
        template = "{title}: {body}"

    assert extract_template_fields("{title}-{body}-{title}") == ["title", "body"]
    assert Renderer()({"title": ["A"], "body": ["B"]}) == ["A: B"]


def test_chat_prompt_formatter_system_fallback(tokenizer_cls):
    tokenizer = tokenizer_cls(supports_system=False)
    formatter = ChatPromptFormatter(tokenizer, "system")
    rendered = formatter("hello")
    assert "system\n\nhello" in rendered
    assert formatter.supports_system_role is False


def test_build_training_example_completion_priority():
    input_ids, labels = build_training_example([1, 2, 3, 4], [8, 9], 4, 2)
    assert input_ids == [4, 8, 9, 2]
    assert labels == [-100, 8, 9, 2]


def test_sft_collator(tokenizer_cls):
    tokenizer = tokenizer_cls()
    collator = build_sft_collator(tokenizer, max_sequence_length=16, pad_to_multiple_of=4)
    batch = collator({"prompt": ["hello"], "completion": ["world"]})
    assert set(batch) == {"input_ids", "attention_mask", "labels"}
    assert batch["input_ids"].shape == batch["labels"].shape
    assert (batch["labels"] == -100).any()
    assert (batch["labels"] != -100).any()


def test_packed_sft_collator(tokenizer_cls):
    tokenizer = tokenizer_cls()
    collator = build_packed_sft_collator(
        tokenizer,
        max_sequence_length=8,
        packing_factor=2,
    )
    batch = collator({"prompt": ["a", "b"], "completion": ["x", "y"]})
    assert batch["input_ids"].shape == (1, 16)
    assert batch["labels"].shape == (1, 16)
    assert batch["position_ids"].shape == (1, 16)


def test_shortest_first_packer_pending():
    packer = ShortestFirstSequencePacker(token_budget=5)
    packed = packer.pack([([1, 2], [1, 2]), ([3, 4, 5, 6], [3, 4, 5, 6])], pad_id=0)
    assert packed.packed_example_count == 1
    assert packed.padding_length == 3
    assert len(packer.pending) == 1
    assert packer.pending_tokens == 4


def test_config_defaults_use_clean_namespace():
    config = SFTDataConfig()
    assert config.prompt_renderer.startswith("llm_training_data.")
    assert config.completion_renderer.startswith("llm_training_data.")


def test_system_prompt_metadata(tmp_path: Path):
    write_system_prompt_metadata(tmp_path, "hello")
    assert read_system_prompt_metadata(tmp_path) == "hello"
    assert resolve_system_prompt(None, tmp_path) == "hello"
    assert resolve_system_prompt("override", tmp_path) == "override"


def test_invalid_budgets(tokenizer_cls):
    tokenizer = tokenizer_cls()
    with pytest.raises(ValueError, match="max_sequence_length"):
        build_sft_collator(tokenizer, max_sequence_length=0)
    with pytest.raises(ValueError, match="packing_factor"):
        build_packed_sft_collator(tokenizer, max_sequence_length=8, packing_factor=0)
