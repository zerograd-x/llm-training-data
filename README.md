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

- `SourceRef`, `FamilySpec`, `DataSuiteSpec`, `PreparedArtifactRef`
- `FamilyPrepareSpec`, `PrepareSuiteSpec`, `PrepareSpec`, `PrepareRegistry`, `PreparePlan`, `PrepareStats`
- `PreparedExample`, `TaskBlendSpec`, `EvaluationCellSpec`, `BlendSpec`, `DataPlan`
- `blend_prepared_examples()`, `format_data_plan()`, `save_data_plan()`
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

Pretraining data stays semantic until the rendering/tokenization stage. The data
contract separates four identities that should not be conflated:

```text
source snapshot
    ↓
prepare family
    ↓
semantic task
    ↓
evaluation cell
```

A suite records source snapshots plus family/task ownership:

```python
from llm_training_data import DataSuiteSpec, FamilySpec, SourceRef

suite = DataSuiteSpec(
    suite_id="experiment-001",
    sources=(
        SourceRef(
            name="documents",
            uri="dataset://documents",
            snapshot="2026-09-09",
        ),
    ),
    families=(
        FamilySpec(
            name="generation",
            source_names=("documents",),
            task_names=("text_to_target",),
        ),
    ),
)
```

`SourceRef` requires at least one of `version`, `snapshot`, or `fingerprint`, so a
suite does not silently mean "whatever is latest". `DataSuiteSpec` validates source
references and requires every task to have exactly one prepare-family owner.

Prepare execution is an explicit plugin contract rather than a built-in storage or
workflow system. A run enables only the families it needs, and may select only a
subset of a family's declared tasks:

```python
from llm_training_data import (
    FamilyPrepareSpec,
    PrepareRegistry,
    PrepareSpec,
    PrepareSuiteSpec,
    plan_prepare_suite,
)

class GenericPrepareSpec(PrepareSpec):
    name = "generation"
    task_names = ("text_to_target",)

    def prepare(self, plan):
        ...  # materialize externally and return PrepareResult

registry = PrepareRegistry()
registry.register(GenericPrepareSpec)

prepare_suite = PrepareSuiteSpec(
    run_id="data-run-001",
    families={
        "generation": FamilyPrepareSpec(config={"mode": "standard"}),
    },
)
plans = plan_prepare_suite(suite, prepare_suite, registry)
```

`PrepareSpec.validate_config(...)` is an optional family-specific preflight hook.
`PrepareRegistry` is explicit and has no import-time registration side effects.
`PreparePlan` binds the resolved source snapshots, selected tasks, run config, and
suite fingerprint into a stable plan fingerprint. The returned artifact records that
prepare-plan fingerprint, closing the provenance chain from source snapshots through
family configuration to materialized data. `execute_prepare_plan(...)` validates the
plugin result before accepting it. Concrete prepare implementations remain outside the
core package and may use any execution or storage technology.

Prepare output has a versioned canonical semantic schema. Use
`validate_prepared_mappings(...)` when reading raw rows, or
`validate_prepared_examples(...)` after parsing them. These checks enforce exact
schema fields, selected task membership, allowed cells, generation/choice
invariants, duplicate rejection, and the rule that probe rows are created only by
the blend stage. `PrepareStats` records output counts, task/cell coverage, distinct
groups, and optional application-defined rejection accounting.

A materialized prepare-stage output can be represented without coupling the library
to a storage system:

```python
from llm_training_data import PreparedArtifactRef

prepared = PreparedArtifactRef(
    family="generation",
    uri="artifact://prepared/generation",
    schema_version="semantic-v1",
    row_count=100000,
    fingerprint="...",
    source_names=("documents",),
    prepare_plan_fingerprint=plans[0].fingerprint,
)
```

The preferred blend API makes train quota and supply policy explicit per task and
makes evaluation-cell meaning inspectable metadata:

```python
from llm_training_data import (
    BlendSpec,
    EvaluationCellSpec,
    PreparedExample,
    TaskBlendSpec,
    blend_prepared_examples,
)

rows = [
    PreparedExample(
        task_name="text_to_target",
        split="train",
        group_id="entity-123",
        input_text="Example input",
        target_text="example-target",
    ),
]

spec = BlendSpec(
    task_specs={
        "text_to_target": TaskBlendSpec(
            train_rows=100000,
            shortfall_policy="error",
        ),
    },
    evaluation_cells=(
        EvaluationCellSpec(
            name="validation",
            dimensions={"entity_exposure": "held_out"},
        ),
    ),
    eval_rows_per_cell=5000,
    probe_rows_per_task=5000,
    seed=42,
)

result = blend_prepared_examples(rows, spec)
print(result.plan.rows_per_cell)
```

`EvaluationCellSpec.task_names` can restrict a cell to only the tasks for which
that experimental axis is meaningful. An omitted `task_names` applies the cell to
all selected tasks. A row appearing in a cell where its task is explicitly not
applicable is rejected, so "not applicable" is not confused with zero data supply.

`shortfall_policy="error"` treats missing requested train supply as a broken data
contract. `shortfall_policy="cap"` records a warning and takes all available train
rows. Zero train supply always fails. Different tasks may use different policies.

`task_row_counts={...}` plus the legacy `cap_to_available` flag remains supported as
a compatibility surface; it is normalized internally into per-task
`TaskBlendSpec` objects. The original positional field order of `BlendSpec` is also
preserved.

Sampling uses a stable semantic `sample_id` plus a seeded deterministic hash rank,
so the same prepared corpus and seed produce the same blend. `train_probe` rows are
derived only from rows already selected into train, making `probe ⊆ train` a
construction invariant rather than a convention.

`DataPlan` records source/suite lineage when provided, prepared artifacts,
available/requested/selected counts by task and applicable cell, distinct-group
coverage, supply warnings, evaluation-cell dimensions, the sampling seed, and
fingerprint version. Persist it next to a dataset with `save_data_plan(...)` as
`data-plan.json`.

The implementation is intentionally platform independent. External data systems
can materialize the prepared corpus and implement the same prepare/blend contracts
without becoming dependencies of the core package.

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

class RecordPromptRenderer(TemplatePromptRenderer):
    template = "Header: {header}\nBody: {body}"

collator = build_sft_collator(
    tokenizer,
    max_sequence_length=2048,
    prompt_renderer=RecordPromptRenderer(),
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
ambiguous task ownership, stale/underspecified source references, prepare-plan
mismatch, prepared-schema/lineage mismatch, non-applicable evaluation rows,
zero/short train supply, non-positive sequence
budgets, missing pad/EOS token IDs, empty normal SFT batches, renderer row-count
mismatches, and inconsistent template fields.
