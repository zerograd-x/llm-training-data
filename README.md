# sft-tokenization

A small, generic library for turning column-oriented examples into supervised
causal-LM batches. It focuses on tokenization boundaries and sequence packing;
it does **not** include a trainer, data platform, model registry, or
application-specific prompts.

## What it provides

- Pluggable prompt/completion/chosen/rejected transforms.
- Consistent Hugging Face chat-template wrapping.
- Separate prompt/completion tokenization so the loss boundary is explicit.
- Completion-only labels (`prompt=-100`, `completion=token_id`).
- Completion-priority truncation and left-truncated prompts.
- Dynamic padding for normal SFT batches.
- Stateful shortest-first sequence packing with reset `position_ids`.
- Exact system-prompt checkpoint sidecar helpers.

## Design boundary

The core package is independent of Ray, Spark, DeepSpeed, vLLM, and any
application-specific dataset schema. Inputs are ordinary column-oriented
mappings such as:

```python
{
    "prompt": ["Explain gravity.", "What is 2+2?"],
    "completion": ["Gravity is ...", "4"],
}
```

Hugging Face tokenizers are supported structurally through a small protocol;
`transformers` is an optional dependency rather than a runtime requirement for
the package itself.

## Install

```bash
pip install -e .
```

For Hugging Face examples:

```bash
pip install -e '.[hf]'
```

For development:

```bash
pip install -e '.[dev]'
pytest
```

## Basic SFT

```python
from sft_tokenization import create_sft_collate_fn

collate = create_sft_collate_fn(tokenizer, max_seq_length=2048)
batch = collate({
    "prompt": ["Question: 2 + 2 ="],
    "completion": [" 4"],
})
```

The returned dictionary contains `input_ids`, `attention_mask`, and `labels`.
Prompt and padding positions are masked with `-100` in `labels`.

## Packed SFT

```python
from sft_tokenization import create_packed_sft_collate_fn

collate = create_packed_sft_collate_fn(
    tokenizer,
    max_seq_length=2048,
    max_packed_rows=8,
)
packed = collate(batch)
```

Packed output has physical batch size 1 and contains `input_ids`, `labels`, and
reset `position_ids`. It intentionally does not return a normal
`attention_mask`.

### Important backend requirement

Reset `position_ids` only isolate packed examples when the model's attention
backend explicitly interprets those resets as independent sequences. This is
not universal across all FlashAttention-enabled or hybrid architectures. Test
backend compatibility before enabling packed training for a new model family.

The pure packing state is exposed separately as `GreedySequencePacker`; it also
retains segment lengths and cumulative boundaries so future backend adapters
can pass explicit sequence metadata without changing the packing algorithm.

## Intentional hardening vs. the source specification

This public implementation preserves the core token/label/packing semantics but
fails early on several malformed inputs instead of silently continuing:

- non-positive sequence budgets;
- missing both `pad_token_id` and `eos_token_id`;
- empty normal SFT batches;
- mismatched prompt/completion row counts;
- template columns with inconsistent lengths.

These checks are deliberately generic and are not tied to any training
platform.
