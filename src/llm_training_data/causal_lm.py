from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch

from .config import CausalLMDataConfig
from .types import ColumnBatch, TokenizerLike


SequenceExample = dict[str, list[int]]


def _resolve_pad_id(tokenizer: TokenizerLike) -> int:
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("Tokenizer must define pad_token_id or eos_token_id")
    return pad_id


def _validate_documents(documents: Sequence[str]) -> list[str]:
    docs = list(documents)
    if any(not isinstance(document, str) for document in docs):
        raise TypeError("documents must contain strings")
    if any(not document for document in docs):
        raise ValueError("documents must not contain empty strings")
    return docs


def _tokenize_documents(
    documents: Sequence[str],
    *,
    tokenizer: TokenizerLike,
    config: CausalLMDataConfig,
) -> list[list[int]]:
    docs = _validate_documents(documents)
    encoded = tokenizer(
        docs,
        add_special_tokens=config.add_special_tokens,
    )["input_ids"]
    if len(encoded) != len(docs):
        raise ValueError("Tokenizer returned a different number of rows than documents")

    eos_token_id = tokenizer.eos_token_id
    if config.append_eos and eos_token_id is None:
        raise ValueError("append_eos=True requires tokenizer.eos_token_id")

    tokenized: list[list[int]] = []
    for token_ids in encoded:
        ids = list(token_ids)
        if not ids:
            raise ValueError("Tokenizer produced an empty document")
        if config.append_eos:
            ids.append(eos_token_id)
        tokenized.append(ids)
    return tokenized


def _as_example(token_ids: list[int]) -> SequenceExample:
    return {
        "input_ids": list(token_ids),
        "labels": list(token_ids),
    }


def build_causal_lm_sequences(
    documents: Sequence[str],
    *,
    tokenizer: TokenizerLike,
    max_sequence_length: int,
    config: CausalLMDataConfig | None = None,
) -> list[SequenceExample]:
    """Tokenize documents into model-ready causal-LM sequences.

    The returned rows are ordinary map-style dataset rows. labels equal
    input_ids for every real token; the trainer owns the causal next-token
    shift. Padding is added only by build_causal_lm_collator.

    When pack_across_documents is enabled, adjacent documents may share one
    sequence. append_eos marks the boundary, but standard causal attention and
    next-token loss are not isolated across that boundary.
    """

    if max_sequence_length <= 0:
        raise ValueError("max_sequence_length must be > 0")
    config = config or CausalLMDataConfig()
    tokenized = _tokenize_documents(
        documents,
        tokenizer=tokenizer,
        config=config,
    )
    if not tokenized:
        return []

    if not config.allow_document_split:
        too_long = [len(ids) for ids in tokenized if len(ids) > max_sequence_length]
        if too_long:
            raise ValueError(
                "A document exceeds max_sequence_length while "
                "allow_document_split=False"
            )

    sequences: list[SequenceExample] = []

    if config.pack_across_documents and config.allow_document_split:
        stream: list[int] = []
        for ids in tokenized:
            stream.extend(ids)
        for start in range(0, len(stream), max_sequence_length):
            chunk = stream[start : start + max_sequence_length]
            if len(chunk) < max_sequence_length and config.drop_remainder:
                break
            sequences.append(_as_example(chunk))
        return sequences

    if config.pack_across_documents:
        current: list[int] = []
        for ids in tokenized:
            if current and len(current) + len(ids) > max_sequence_length:
                if not (config.drop_remainder and len(current) < max_sequence_length):
                    sequences.append(_as_example(current))
                current = []
            current.extend(ids)
        if current and not (config.drop_remainder and len(current) < max_sequence_length):
            sequences.append(_as_example(current))
        return sequences

    for ids in tokenized:
        if config.allow_document_split:
            chunks = [
                ids[start : start + max_sequence_length]
                for start in range(0, len(ids), max_sequence_length)
            ]
        else:
            chunks = [ids]
        for chunk in chunks:
            if len(chunk) < max_sequence_length and config.drop_remainder:
                continue
            sequences.append(_as_example(chunk))

    return sequences


def build_causal_lm_collator(
    tokenizer: TokenizerLike,
    *,
    max_sequence_length: int,
) -> Callable[[ColumnBatch], dict[str, torch.Tensor]]:
    """Pad pre-tokenized causal-LM rows to one fixed sequence length."""

    if max_sequence_length <= 0:
        raise ValueError("max_sequence_length must be > 0")
    pad_id = _resolve_pad_id(tokenizer)

    def collate(batch: ColumnBatch) -> dict[str, torch.Tensor]:
        input_rows = batch.get("input_ids")
        label_rows = batch.get("labels")
        if input_rows is None or label_rows is None:
            raise KeyError("Causal-LM batch must contain input_ids and labels")
        if len(input_rows) != len(label_rows):
            raise ValueError("input_ids and labels batch lengths must match")
        if not input_rows:
            raise ValueError("Causal-LM collator received an empty batch")

        padded_inputs: list[list[int]] = []
        padded_labels: list[list[int]] = []
        attention_rows: list[list[int]] = []

        for input_ids, labels in zip(input_rows, label_rows):
            ids = list(input_ids)
            targets = list(labels)
            if len(ids) != len(targets):
                raise ValueError("Each causal-LM row must have aligned input_ids and labels")
            if not ids:
                raise ValueError("Causal-LM rows must not be empty")
            if len(ids) > max_sequence_length:
                raise ValueError(
                    f"Causal-LM row length {len(ids)} exceeds "
                    f"max_sequence_length={max_sequence_length}"
                )
            padding = max_sequence_length - len(ids)
            padded_inputs.append(ids + [pad_id] * padding)
            padded_labels.append(targets + [-100] * padding)
            attention_rows.append([1] * len(ids) + [0] * padding)

        return {
            "input_ids": torch.tensor(padded_inputs, dtype=torch.long),
            "attention_mask": torch.tensor(attention_rows, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }

    return collate
