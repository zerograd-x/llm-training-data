# llm-training-data

A focused library for turning structured examples into model-ready LLM training data.
It owns semantic data preparation, deterministic pretraining blends, rendering,
tokenization, label construction, padding, and sequence packing. It does not
include a trainer, model registry, distributed runtime, or application-specific
storage/orchestration.

The distribution name and Python import namespace are intentionally aligned:

```python
import llm_training_data
```

## Public API

The API uses data-preparation terminology rather than implementation-oriented names:

- `PreparedExample`, `BlendSpec`, `DataPlan`, `blend_prepared_examples()`
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

## Pretraining semantic corpus and blend

Pretraining data stays semantic until the rendering/tokenization stage. A prepared
example carries task identity, split/holdout identity, a group id for lineage, and
semantic slots for generation or multiple-choice objectives:

```python
from llm_training_data import BlendSpec, PreparedExample, blend_prepared_examples

rows = [
    PreparedExample(
        task_name="text_to_target",
        split="train",
        group_id="entity-123",
        input_text="Example input",
        target_text="example-target",
    ),
]

result = blend_prepared_examples(
    rows,
    BlendSpec(
        task_row_counts={"text_to_target": 1},
        eval_rows_per_cell=5000,
        probe_rows_per_task=5000,
        seed=42,
    ),
)

print(result.plan.rows_per_cell)
```

`task_row_counts` is both the train quota and the task allowlist. Sampling uses a
stable semantic `sample_id` plus a seeded deterministic hash rank, so the same
prepared corpus and seed produce the same blend. `train_probe` rows are derived
only from rows already selected into train, making `probe ⊆ train` a construction
invariant rather than a convention.

`DataPlan` records available/requested/selected counts by task and split, supply
warnings, the sampling seed, and fingerprint version. Persist it next to a dataset
with `save_data_plan(...)` as `data-plan.json`.

The implementation is intentionally platform independent. External data systems
can materialize the prepared corpus and implement the same blend contract without
becoming dependencies of the core package.

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
backend explicitly interprets those resets as independent sequences. Validate a
new model/backend combination before enabling packed training.

The GPU integration test compares a packed segment against the same segment run
standalone under a compatible packed-attention backend. It is intentionally
separate from normal CI because it requires GPU-specific dependencies.

## Design boundary

The package remains independent of distributed runtimes, storage systems, and
application-specific schemas. Hugging Face tokenizers are supported through a
small protocol, while `transformers` remains an optional dependency.

The package fails early on malformed inputs such as invalid semantic examples,
zero/short train supply, non-positive sequence budgets, missing pad/EOS token IDs,
empty normal SFT batches, renderer row-count mismatches, and inconsistent template
fields.
