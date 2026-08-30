from __future__ import annotations

from collections.abc import Callable

import torch

from .chat import ChatPromptFormatter
from .examples import build_training_example, round_up
from .packing import ShortestFirstSequencePacker
from .renderers import (
    CompletionRenderer,
    DefaultCompletionRenderer,
    DefaultPromptRenderer,
    PromptRenderer,
)
from .types import ColumnBatch, TokenizerLike


def _resolve_pad_id(tokenizer: TokenizerLike) -> int:
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("Tokenizer must define pad_token_id or eos_token_id")
    return pad_id


def _tokenize_examples(
    batch: ColumnBatch,
    *,
    tokenizer: TokenizerLike,
    prompt_renderer: PromptRenderer,
    completion_renderer: CompletionRenderer,
    prompt_formatter: ChatPromptFormatter,
    max_sequence_length: int,
    eos_token_id: int | None,
) -> list[tuple[list[int], list[int]]]:
    prompts = prompt_renderer(batch)
    completions = completion_renderer(batch)
    if len(prompts) != len(completions):
        raise ValueError(
            f"prompt_renderer returned {len(prompts)} rows but completion_renderer "
            f"returned {len(completions)} rows"
        )

    formatted_prompts = [prompt_formatter(prompt) for prompt in prompts]
    prompt_ids = tokenizer(formatted_prompts, add_special_tokens=True)["input_ids"]
    completion_ids = tokenizer(completions, add_special_tokens=False)["input_ids"]
    if len(prompt_ids) != len(completion_ids):
        raise ValueError("Tokenizer returned mismatched prompt/completion batch lengths")

    return [
        build_training_example(
            list(prompt_token_ids),
            list(response_token_ids),
            max_sequence_length,
            eos_token_id,
        )
        for prompt_token_ids, response_token_ids in zip(prompt_ids, completion_ids)
    ]


def build_sft_collator(
    tokenizer: TokenizerLike,
    max_sequence_length: int,
    prompt_renderer: PromptRenderer | None = None,
    completion_renderer: CompletionRenderer | None = None,
    system_prompt: str | None = None,
    enable_thinking: bool | None = None,
    pad_to_multiple_of: int = 64,
    include_mm_token_type_ids: bool = False,
) -> Callable[[ColumnBatch], dict[str, torch.Tensor]]:
    if max_sequence_length <= 0:
        raise ValueError("max_sequence_length must be > 0")

    prompt_renderer = prompt_renderer or DefaultPromptRenderer()
    completion_renderer = completion_renderer or DefaultCompletionRenderer()
    prompt_formatter = ChatPromptFormatter(tokenizer, system_prompt, enable_thinking)
    pad_id = _resolve_pad_id(tokenizer)
    eos_token_id = tokenizer.eos_token_id

    def collate(batch: ColumnBatch) -> dict[str, torch.Tensor]:
        rows = _tokenize_examples(
            batch,
            tokenizer=tokenizer,
            prompt_renderer=prompt_renderer,
            completion_renderer=completion_renderer,
            prompt_formatter=prompt_formatter,
            max_sequence_length=max_sequence_length,
            eos_token_id=eos_token_id,
        )
        if not rows:
            raise ValueError("SFT collator received an empty batch")

        longest = max(len(ids) for ids, _ in rows)
        target_length = min(round_up(longest, pad_to_multiple_of), max_sequence_length)
        input_rows: list[list[int]] = []
        label_rows: list[list[int]] = []
        attention_rows: list[list[int]] = []
        for ids, labels in rows:
            padding_length = target_length - len(ids)
            input_rows.append(ids + [pad_id] * padding_length)
            label_rows.append(labels + [-100] * padding_length)
            attention_rows.append([1] * len(ids) + [0] * padding_length)

        input_ids = torch.tensor(input_rows, dtype=torch.long)
        result = {
            "input_ids": input_ids,
            "attention_mask": torch.tensor(attention_rows, dtype=torch.long),
            "labels": torch.tensor(label_rows, dtype=torch.long),
        }
        if include_mm_token_type_ids:
            result["mm_token_type_ids"] = torch.zeros_like(input_ids)
        return result

    return collate


def build_packed_sft_collator(
    tokenizer: TokenizerLike,
    max_sequence_length: int,
    packing_factor: int,
    prompt_renderer: PromptRenderer | None = None,
    completion_renderer: CompletionRenderer | None = None,
    system_prompt: str | None = None,
    enable_thinking: bool | None = None,
    include_mm_token_type_ids: bool = False,
) -> Callable[[ColumnBatch], dict[str, torch.Tensor]]:
    if max_sequence_length <= 0:
        raise ValueError("max_sequence_length must be > 0")
    if packing_factor <= 0:
        raise ValueError("packing_factor must be > 0")

    prompt_renderer = prompt_renderer or DefaultPromptRenderer()
    completion_renderer = completion_renderer or DefaultCompletionRenderer()
    prompt_formatter = ChatPromptFormatter(tokenizer, system_prompt, enable_thinking)
    pad_id = _resolve_pad_id(tokenizer)
    eos_token_id = tokenizer.eos_token_id
    packer = ShortestFirstSequencePacker(max_sequence_length * packing_factor)

    def collate(batch: ColumnBatch) -> dict[str, torch.Tensor]:
        examples = _tokenize_examples(
            batch,
            tokenizer=tokenizer,
            prompt_renderer=prompt_renderer,
            completion_renderer=completion_renderer,
            prompt_formatter=prompt_formatter,
            max_sequence_length=max_sequence_length,
            eos_token_id=eos_token_id,
        )
        packed = packer.pack(examples, pad_id=pad_id)
        input_ids = torch.tensor([packed.input_ids], dtype=torch.long)
        result = {
            "input_ids": input_ids,
            "labels": torch.tensor([packed.labels], dtype=torch.long),
            "position_ids": torch.tensor([packed.position_ids], dtype=torch.long),
        }
        if include_mm_token_type_ids:
            result["mm_token_type_ids"] = torch.zeros_like(input_ids)
        return result

    collate.packer = packer  # type: ignore[attr-defined]
    return collate
