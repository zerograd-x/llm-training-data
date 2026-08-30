# llm-training-data

A focused library for turning structured examples into model-ready LLM training batches.
The current implementation covers supervised causal-LM preparation: rendering,
chat formatting, tokenization, label construction, padding, and sequence packing.
It does not include a trainer, model registry, or application-specific prompts.

The distribution name and Python import namespace are intentionally aligned:

```python
import llm_training_data
```

## Public API

The API uses data-preparation terminology rather than implementation-oriented names:

- `PromptRenderer`, `CompletionRenderer`
- `TemplatePromptRenderer`
- `ChatPromptFormatter`
- `SFTDataConfig`
- `build_sft_collator()`
- `build_packed_sft_collator()`
- `ShortestFirstSequencePacker`
- `PackedSequence`

## Install

```bash
pip install -e .
```

For Hugging Face integration:

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
from llm_training_data import build_sft_collator

collator = build_sft_collator(
    tokenizer,
    max_sequence_length=2048,
)

batch = collator({
    "prompt": ["Question: 2 + 2 ="],
    "completion": [" 4"],
})
```

The returned dictionary contains `input_ids`, `attention_mask`, and `labels`.
Prompt and padding positions are masked with `-100` in `labels`.

## Custom rendering

```python
from llm_training_data import TemplatePromptRenderer, build_sft_collator

class ProductPromptRenderer(TemplatePromptRenderer):
    template = "Title: {title}\nDescription: {description}"

collator = build_sft_collator(
    tokenizer,
    max_sequence_length=2048,
    prompt_renderer=ProductPromptRenderer(),
)
```

Renderers operate on ordinary column-oriented mappings, so the core package is
independent of any specific dataset platform.

## Packed SFT

```python
from llm_training_data import build_packed_sft_collator

collator = build_packed_sft_collator(
    tokenizer,
    max_sequence_length=2048,
    packing_factor=8,
)
```

The physical output batch size is 1. The token budget for one packed row is
`max_sequence_length * packing_factor`.

Packed output contains `input_ids`, `labels`, and reset `position_ids`; it does
not include a normal `attention_mask`.

### Backend requirement

Reset `position_ids` only isolate logical examples when the model attention
backend explicitly interprets those resets as independent sequences. This is not
universal across FlashAttention-enabled or hybrid architectures. Validate a new
model/backend combination before enabling packed training.

The GPU integration test compares a packed segment against the same segment run
standalone under FlashAttention-2. It is intentionally separate from normal CI
because it requires CUDA and `flash-attn`.

## Design boundary

The library is independent of Ray, Spark, DeepSpeed, vLLM, and application-specific
schemas. Hugging Face tokenizers are supported through a small protocol, while
`transformers` remains an optional dependency.

The package fails early on malformed inputs such as non-positive sequence budgets,
missing pad/EOS token IDs, empty normal SFT batches, renderer row-count mismatches,
and inconsistent template fields.
