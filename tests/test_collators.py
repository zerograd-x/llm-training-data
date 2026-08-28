from __future__ import annotations

import pytest
import torch

from sft_tokenization.collators import create_packed_sft_collate_fn, create_sft_collate_fn
from sft_tokenization.transforms import CompletionTransform, PromptTransform


class QuestionPrompt(PromptTransform):
    def __call__(self, batch):
        return [str(x) for x in batch["question"]]


class AnswerCompletion(CompletionTransform):
    def __call__(self, batch):
        return [str(x) for x in batch["answer"]]


def test_sft_collator_masks_prompt_and_padding(tokenizer_cls):
    tokenizer = tokenizer_cls()
    collate = create_sft_collate_fn(
        tokenizer,
        max_seq_length=8,
        prompt_transform=QuestionPrompt(),
        completion_transform=AnswerCompletion(),
        pad_to_multiple_of=4,
    )
    result = collate({"question": ["one two", "three"], "answer": ["yes", "no"]})

    assert result["input_ids"].shape == (2, 8)
    assert result["attention_mask"].tolist() == [
        [1, 1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0],
    ]
    assert result["labels"][0, :3].tolist() == [-100, -100, -100]
    assert result["labels"][1, -1].item() == -100
    assert tokenizer.calls[0][1] is True
    assert tokenizer.calls[1][1] is False


def test_sft_collator_rejects_empty_batch(tokenizer_cls):
    tokenizer = tokenizer_cls()
    collate = create_sft_collate_fn(tokenizer, 8)
    with pytest.raises(ValueError):
        collate({"prompt": [], "completion": []})


def test_packed_collator_resets_positions_and_has_batch_size_one(tokenizer_cls):
    tokenizer = tokenizer_cls()
    collate = create_packed_sft_collate_fn(
        tokenizer,
        max_seq_length=4,
        max_packed_rows=2,
        prompt_transform=QuestionPrompt(),
        completion_transform=AnswerCompletion(),
    )
    result = collate({"question": ["a", "b"], "answer": ["x", "y"]})

    assert result["input_ids"].shape == (1, 8)
    assert result["labels"].shape == (1, 8)
    assert result["position_ids"].shape == (1, 8)
    assert "attention_mask" not in result
    assert result["position_ids"].tolist()[0] == [0, 1, 2, 3, 0, 1, 2, 3]


def test_optional_zero_token_type_ids(tokenizer_cls):
    tokenizer = tokenizer_cls()
    collate = create_sft_collate_fn(tokenizer, 8, needs_mm_token_type_ids=True)
    result = collate({"prompt": ["a"], "completion": ["b"]})
    assert torch.equal(result["mm_token_type_ids"], torch.zeros_like(result["input_ids"]))
