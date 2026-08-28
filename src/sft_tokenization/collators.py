from __future__ import annotations

from collections.abc import Callable

import torch

from .chat import ChatTemplateWrapper
from .examples import build_example_ids, round_up
from .packing import GreedySequencePacker
from .transforms import (
    CompletionTransform,
    DefaultCompletionTransform,
    DefaultPromptTransform,
    PromptTransform,
)
from .types import ColumnBatch, TokenizerLike


def _resolve_pad_id(tokenizer: TokenizerLike) -> int:
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("Tokenizer must define pad_token_id or eos_token_id")
    return pad_id


def _validate_alignment(prompts: list[str], completions: list[str]) -> None:
    if len(prompts) != len(completions):
        raise ValueError(
            f"prompt_transform returned {len(prompts)} rows but completion_transform "
            f"returned {len(completions)} rows"
        )


def _tokenize_examples(
    batch: ColumnBatch,
    *,
    tokenizer: TokenizerLike,
    prompt_transform: PromptTransform,
    completion_transform: CompletionTransform,
    prompt_wrapper: ChatTemplateWrapper,
    max_seq_length: int,
    eos_id: int | None,
) -> list[tuple[list[int], list[int]]]:
    prompts = prompt_transform(batch)
    completions = completion_transform(batch)
    _validate_alignment(prompts, completions)

    wrapped_prompts = [prompt_wrapper(prompt) for prompt in prompts]
    prompt_ids = tokenizer(wrapped_prompts, add_special_tokens=True)["input_ids"]
    completion_ids = tokenizer(completions, add_special_tokens=False)["input_ids"]

    if len(prompt_ids) != len(completion_ids):
        raise ValueError("Tokenizer returned mismatched prompt/completion batch lengths")

    return [
        build_example_ids(list(p_ids), list(c_ids), max_seq_length, eos_id)
        for p_ids, c_ids in zip(prompt_ids, completion_ids)
    ]


def create_sft_collate_fn(
    tokenizer: TokenizerLike,
    max_seq_length: int,
    prompt_transform: PromptTransform | None = None,
    completion_transform: CompletionTransform | None = None,
    system_prompt: str | None = None,
    enable_thinking: bool | None = None,
    pad_to_multiple_of: int = 64,
    needs_mm_token_type_ids: bool = False,
) -> Callable[[ColumnBatch], dict[str, torch.Tensor]]:
    """Create a dynamic-padding SFT collator with completion-only loss."""
    if max_seq_length <= 0:
        raise ValueError("max_seq_length must be > 0")

    prompt_transform = prompt_transform or DefaultPromptTransform()
    completion_transform = completion_transform or DefaultCompletionTransform()
    prompt_wrapper = ChatTemplateWrapper(tokenizer, system_prompt, enable_thinking)
    pad_id = _resolve_pad_id(tokenizer)
    eos_id = tokenizer.eos_token_id

    def collate_fn(batch: ColumnBatch) -> dict[str, torch.Tensor]:
        rows = _tokenize_examples(
            batch,
            tokenizer=tokenizer,
            prompt_transform=prompt_transform,
            completion_transform=completion_transform,
            prompt_wrapper=prompt_wrapper,
            max_seq_length=max_seq_length,
            eos_id=eos_id,
        )
        if not rows:
            raise ValueError("SFT collator received an empty batch")

        longest = max(len(ids) for ids, _ in rows)
        target_len = min(round_up(longest, pad_to_multiple_of), max_seq_length)

        input_rows: list[list[int]] = []
        label_rows: list[list[int]] = []
        attention_rows: list[list[int]] = []
        for ids, labels in rows:
            pad_len = target_len - len(ids)
            input_rows.append(ids + [pad_id] * pad_len)
            label_rows.append(labels + [-100] * pad_len)
            attention_rows.append([1] * len(ids) + [0] * pad_len)

        input_ids = torch.tensor(input_rows, dtype=torch.long)
        result = {
            "input_ids": input_ids,
            "attention_mask": torch.tensor(attention_rows, dtype=torch.long),
            "labels": torch.tensor(label_rows, dtype=torch.long),
        }
        if needs_mm_token_type_ids:
            result["mm_token_type_ids"] = torch.zeros_like(input_ids)
        return result

    return collate_fn


def create_packed_sft_collate_fn(
    tokenizer: TokenizerLike,
    max_seq_length: int,
    max_packed_rows: int,
    prompt_transform: PromptTransform | None = None,
    completion_transform: CompletionTransform | None = None,
    system_prompt: str | None = None,
    enable_thinking: bool | None = None,
    needs_mm_token_type_ids: bool = False,
) -> Callable[[ColumnBatch], dict[str, torch.Tensor]]:
    """Create a stateful shortest-first packed-SFT collator.

    The output has physical batch size 1 and uses ``position_ids`` resets to
    encode segment boundaries. No ``attention_mask`` is returned.

    Correctness therefore requires an attention backend that explicitly
    supports packed-sequence isolation from reset ``position_ids``. This is not
    a universal property of all FlashAttention-enabled or hybrid models.
    """
    if max_seq_length <= 0:
        raise ValueError("max_seq_length must be > 0")
    if max_packed_rows <= 0:
        raise ValueError("max_packed_rows must be > 0")

    prompt_transform = prompt_transform or DefaultPromptTransform()
    completion_transform = completion_transform or DefaultCompletionTransform()
    prompt_wrapper = ChatTemplateWrapper(tokenizer, system_prompt, enable_thinking)
    pad_id = _resolve_pad_id(tokenizer)
    eos_id = tokenizer.eos_token_id

    total_budget = max_seq_length * max_packed_rows
    packer = GreedySequencePacker(total_budget)

    def collate_fn(batch: ColumnBatch) -> dict[str, torch.Tensor]:
        new_examples = _tokenize_examples(
            batch,
            tokenizer=tokenizer,
            prompt_transform=prompt_transform,
            completion_transform=completion_transform,
            prompt_wrapper=prompt_wrapper,
            max_seq_length=max_seq_length,
            eos_id=eos_id,
        )
        layout = packer.pack(new_examples, pad_id=pad_id)

        input_ids = torch.tensor([layout.input_ids], dtype=torch.long)
        result = {
            "input_ids": input_ids,
            "labels": torch.tensor([layout.labels], dtype=torch.long),
            "position_ids": torch.tensor([layout.position_ids], dtype=torch.long),
        }
        if needs_mm_token_type_ids:
            result["mm_token_type_ids"] = torch.zeros_like(input_ids)
        return result

    collate_fn.packer = packer  # type: ignore[attr-defined]
    return collate_fn
