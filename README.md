# llm-training-data

A small, generic library for turning structured examples into model-ready LLM
training batches. The current implementation focuses on supervised causal-LM
preparation: rendering, chat formatting, tokenization, label construction,
padding, and sequence packing. It does **not** include a trainer, data platform,
model registry, or application-specific prompts.

The install/distribution name is `llm-training-data`. The Python import namespace
remains `sft_tokenization` for backward compatibility.

## Public API vocabulary

New code should use names that describe the data operation rather than the
implementation mechanism:

- `PromptRenderer` / `CompletionRenderer` instead of `*Transform`;
- `ChatPromptFormatter` instead of `ChatTemplateWrapper`;
- `build_sft_collator` instead of `create_sft_collate_fn`;
- `build_packed_sft_collator` instead of `create_packed_sft_collate_fn`;
- `ShortestFirstSequencePacker` instead of `GreedySequencePacker`;
- `PackedSequence` instead of `PackedSequenceLayout`;
- `SFTDataConfig` instead of `TokenizationConfig`.

Legacy names remain available as compatibility aliases or wrappers.

## What it provides

- Pluggable prompt/completion/preference response renderers.
- Consistent Hugging Face chat-prompt formatting.
- Separate prompt/completion tokenization so the loss boundary is explicit.
- Completion-only labels (`prompt=-100`, `completion=token_id`).
- Completion-priority truncation and left-truncated prompts.
- Dynamic padding for normal SFT batches.
- Stateful shortest-first sequence packing with reset `position_ids`.
- Exact system-prompt checkpoint sidecar helpers.

## Basic SFT

```python
from sft_tokenization import build_sft_collator

collate = build_sft_collator(
    tokenizer,
    max_sequence_length=2048,
)

batch = collate({
    "prompt": ["Question: 2 + 2 ="],
    "completion": [" 4"],
})
```

The returned dictionary contains `input_ids`, `attention_mask`, and `labels`.
Prompt and padding positions are masked with `-100` in `labels`.

## Packed SFT

```python
from sft_tokenization import build_packed_sft_collator

collate = build_packed_sft_collator(
    tokenizer,
    max_sequence_length=2048,
    packing_factor=8,
)
packed = collate(batch)
```

`packing_factor` means the single physical packed row has a token budget of
`max_sequence_length * packing_factor`.

Packed output has physical batch size 1 and contains `input_ids`, `labels`, and
reset `position_ids`. It intentionally does not return a normal
`attention_mask`.

### Important backend requirement

Reset `position_ids` only isolate packed examples when the model's attention
backend explicitly interprets those resets as independent sequences. This is
not universal across all FlashAttention-enabled or hybrid architectures. Test
backend compatibility before enabling packed training for a new model family.

The pure packing state is exposed as `ShortestFirstSequencePacker`; it retains
segment lengths and cumulative boundaries so future backend adapters can pass
explicit sequence metadata without changing the packing algorithm.

## Backward compatibility

Existing code using names such as `PromptTransform`, `GreedySequencePacker`,
`create_sft_collate_fn`, `max_seq_length`, or `max_packed_rows` continues to
work through compatibility aliases/wrappers. New code should prefer the API
shown above.
