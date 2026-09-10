from __future__ import annotations

from pathlib import Path

import pytest

from llm_training_data import (
    BlendSpec,
    ChatPromptFormatter,
    DataSuiteSpec,
    DefaultCompletionRenderer,
    DefaultPromptRenderer,
    EvaluationCellSpec,
    FamilyPrepareSpec,
    FamilySpec,
    PREPARED_SCHEMA_VERSION,
    PreparePlan,
    PrepareRegistry,
    PrepareSpec,
    PrepareStats,
    PrepareSuiteSpec,
    PreparedArtifactRef,
    PreparedExample,
    SFTDataConfig,
    ShortestFirstSequencePacker,
    SourceRef,
    TaskBlendSpec,
    TemplatePromptRenderer,
    TRAIN_PROBE_SPLIT,
    TRAIN_SPLIT,
    blend_prepared_examples,
    build_packed_sft_collator,
    build_sft_collator,
    build_training_example,
    execute_prepare_plan,
    extract_template_fields,
    format_data_plan,
    make_prepare_result,
    plan_prepare_suite,
    read_system_prompt_metadata,
    resolve_system_prompt,
    save_data_plan,
    validate_prepared_artifact,
    validate_prepared_examples,
    validate_prepared_mappings,
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


def test_prepared_example_has_stable_semantic_sample_id():
    train = PreparedExample(
        task_name="choose_target",
        split=TRAIN_SPLIT,
        group_id="group-1",
        input_text="description",
        options=("ab", "c"),
        answer_index=0,
    )
    probe = train.with_split(TRAIN_PROBE_SPLIT)
    other_options = PreparedExample(
        task_name="choose_target",
        split=TRAIN_SPLIT,
        group_id="group-1",
        input_text="description",
        options=("a", "bc"),
        answer_index=0,
    )
    from_old_contract = PreparedExample.from_mapping(
        {
            "task_name": "generate_target",
            "input_text": "description",
            "options": [],
            "answer_index": -1,
            "target_text": "target",
            "example_id": "group-2",
            "split": "train",
        }
    )

    assert train.sample_id == probe.sample_id
    assert train.sample_id != other_options.sample_id
    assert from_old_contract.group_id == "group-2"


def test_pretrain_blend_is_deterministic_whitelisted_and_probe_is_train_subset(tmp_path: Path):
    rows = []
    for index in range(8):
        rows.append(
            PreparedExample(
                task_name="generate_target",
                split=TRAIN_SPLIT,
                group_id=f"group-{index}",
                input_text=f"input {index}",
                target_text=f"target-{index}",
            )
        )
    for index in range(3):
        rows.append(
            PreparedExample(
                task_name="generate_target",
                split="validation",
                group_id=f"validation-{index}",
                input_text=f"input {index}",
                target_text=f"target-{index}",
            )
        )
    rows.append(
        PreparedExample(
            task_name="not_requested",
            split="ignored_split",
            group_id="ignored",
            input_text="ignored",
            target_text="ignored",
        )
    )

    spec = BlendSpec(
        task_row_counts={"generate_target": 5},
        eval_rows_per_cell=2,
        probe_rows_per_task=3,
        seed=17,
    )
    first = blend_prepared_examples(rows, spec)
    second = blend_prepared_examples(reversed(rows), spec)

    first_identity = [(row.task_name, row.split, row.sample_id) for row in first.examples]
    second_identity = [(row.task_name, row.split, row.sample_id) for row in second.examples]
    assert first_identity == second_identity
    assert {row.task_name for row in first.examples} == {"generate_target"}

    train_ids = {row.sample_id for row in first.examples if row.split == TRAIN_SPLIT}
    probe_ids = {
        row.sample_id for row in first.examples if row.split == TRAIN_PROBE_SPLIT
    }
    assert len(train_ids) == 5
    assert len(probe_ids) == 3
    assert probe_ids <= train_ids
    assert first.plan.rows_per_cell["generate_target/validation"] == 2
    assert first.plan.selected_rows == 5 + 2 + 3
    assert first.plan.input_rows == 12
    assert first.plan.whitelisted_rows == 11

    train_cell = next(
        cell
        for cell in first.plan.cells
        if cell.task_name == "generate_target" and cell.split == TRAIN_SPLIT
    )
    assert train_cell.available == 8
    assert train_cell.available_groups == 8
    assert train_cell.selected == 5
    assert train_cell.selected_groups == 5

    rendered = format_data_plan(first.plan)
    assert "EFFECTIVE DATA PLAN" in rendered
    assert "generate_target/train" in rendered
    assert "groups=8" in rendered

    plan_path = save_data_plan(first.plan, tmp_path)
    assert plan_path.name == "data-plan.json"
    assert plan_path.is_file()


def test_pretrain_blend_distinguishes_zero_supply_from_shortfall():
    rows = [
        PreparedExample(
            task_name="task_a",
            split=TRAIN_SPLIT,
            group_id=f"g-{index}",
            input_text=str(index),
            target_text="x",
        )
        for index in range(2)
    ]

    with pytest.raises(ValueError, match="< requested 3"):
        blend_prepared_examples(rows, BlendSpec(task_row_counts={"task_a": 3}))

    capped = blend_prepared_examples(
        rows,
        BlendSpec(task_row_counts={"task_a": 3}, cap_to_available=True),
    )
    assert len(capped.plan.warnings) == 1
    assert capped.plan.warnings[0].code == "train_supply_shortfall"

    with pytest.raises(ValueError, match="has 0 rows"):
        blend_prepared_examples(rows, BlendSpec(task_row_counts={"missing": 1}))


def test_legacy_blend_positional_order_is_preserved():
    spec = BlendSpec(
        {"task_a": 10},
        7,
        3,
        99,
        True,
        "train",
        "probe",
        ("heldout",),
    )
    assert spec.eval_rows_per_cell == 7
    assert spec.probe_rows_per_task == 3
    assert spec.seed == 99
    assert spec.cap_to_available is True
    assert spec.eval_splits == ("heldout",)
    assert spec.task_specs["task_a"] == TaskBlendSpec(
        train_rows=10,
        shortfall_policy="cap",
    )


def test_pretrain_suite_lineage_cells_and_per_task_shortfall_policy():
    suite = DataSuiteSpec(
        suite_id="suite-001",
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
                task_names=("task_a", "task_b"),
            ),
        ),
    )
    artifact = PreparedArtifactRef(
        family="generation",
        uri="artifact://prepared/generation",
        schema_version="semantic-v1",
        row_count=11,
        fingerprint="prepared-sha256",
        source_names=("documents",),
    )
    spec = BlendSpec(
        task_specs={
            "task_a": TaskBlendSpec(train_rows=3, shortfall_policy="error"),
            "task_b": TaskBlendSpec(
                train_rows=3,
                shortfall_policy="cap",
                eval_rows_per_cell=1,
                probe_rows=1,
            ),
        },
        evaluation_cells=(
            EvaluationCellSpec(
                name="heldout",
                dimensions={"entity_exposure": "unseen"},
            ),
            EvaluationCellSpec(
                name="relation_holdout",
                dimensions={"relation_exposure": "unseen"},
                task_names=("task_b",),
            ),
        ),
        eval_rows_per_cell=2,
        probe_rows_per_task=2,
        seed=9,
    )

    rows = [
        *[
            PreparedExample(
                task_name="task_a",
                split=TRAIN_SPLIT,
                group_id=f"a-{index}",
                input_text=f"a {index}",
                target_text="x",
            )
            for index in range(3)
        ],
        *[
            PreparedExample(
                task_name="task_b",
                split=TRAIN_SPLIT,
                group_id=f"b-{index}",
                input_text=f"b {index}",
                target_text="x",
            )
            for index in range(2)
        ],
        *[
            PreparedExample(
                task_name=task_name,
                split="heldout",
                group_id=f"{task_name}-heldout-{index}",
                input_text=f"heldout {index}",
                target_text="x",
            )
            for task_name in ("task_a", "task_b")
            for index in range(2)
        ],
        *[
            PreparedExample(
                task_name="task_b",
                split="relation_holdout",
                group_id=f"task-b-relation-{index}",
                input_text=f"relation {index}",
                target_text="x",
            )
            for index in range(2)
        ],
    ]

    result = blend_prepared_examples(
        rows,
        spec,
        suite=suite,
        prepared_artifacts=(artifact,),
    )

    assert result.plan.suite == suite
    assert result.plan.prepared_artifacts == (artifact,)
    assert result.plan.task_specs["task_a"].shortfall_policy == "error"
    assert result.plan.task_specs["task_b"].shortfall_policy == "cap"
    assert result.plan.rows_per_cell["task_a/train"] == 3
    assert result.plan.rows_per_cell["task_b/train"] == 2
    assert result.plan.rows_per_cell["task_a/heldout"] == 2
    assert result.plan.rows_per_cell["task_b/heldout"] == 1
    assert "task_a/relation_holdout" not in result.plan.rows_per_cell
    assert result.plan.rows_per_cell["task_b/relation_holdout"] == 1
    assert result.plan.rows_per_cell["task_a/train_probe"] == 2
    assert result.plan.rows_per_cell["task_b/train_probe"] == 1
    assert len(result.plan.warnings) == 1

    heldout_cell = next(
        cell
        for cell in result.plan.cells
        if cell.task_name == "task_a" and cell.split == "heldout"
    )
    relation_cell = next(
        cell
        for cell in result.plan.cells
        if cell.task_name == "task_b" and cell.split == "relation_holdout"
    )
    assert heldout_cell.dimensions == {"entity_exposure": "unseen"}
    assert relation_cell.dimensions == {"relation_exposure": "unseen"}

    rendered = format_data_plan(result.plan)
    assert "SOURCES" in rendered
    assert "PREPARED ARTIFACTS" in rendered
    assert "shortfall=cap" in rendered
    assert "entity_exposure=unseen" in rendered
    assert "relation_exposure=unseen" in rendered


def test_pretrain_suite_rejects_non_applicable_cell_rows():
    rows = [
        PreparedExample(
            task_name="task_a",
            split=TRAIN_SPLIT,
            group_id="train-a",
            input_text="a",
            target_text="x",
        ),
        PreparedExample(
            task_name="task_a",
            split="specialized",
            group_id="bad-cell-a",
            input_text="a",
            target_text="x",
        ),
    ]
    spec = BlendSpec(
        task_specs={"task_a": TaskBlendSpec(train_rows=1)},
        evaluation_cells=(
            EvaluationCellSpec(name="specialized", task_names=("task_b",)),
        ),
    )
    with pytest.raises(ValueError, match="not applicable"):
        blend_prepared_examples(rows, spec)


def test_pretrain_suite_rejects_ambiguous_task_ownership():
    source = SourceRef(
        name="source",
        uri="dataset://source",
        version="v1",
    )
    with pytest.raises(ValueError, match="belongs to both"):
        DataSuiteSpec(
            sources=(source,),
            families=(
                FamilySpec(
                    name="family_a",
                    source_names=("source",),
                    task_names=("shared_task",),
                ),
                FamilySpec(
                    name="family_b",
                    source_names=("source",),
                    task_names=("shared_task",),
                ),
            ),
        )



def test_prepare_suite_registry_planning_and_task_pruning():
    suite = DataSuiteSpec(
        suite_id="suite-001",
        sources=(
            SourceRef(
                name="documents",
                uri="dataset://documents",
                snapshot="snapshot-001",
            ),
        ),
        families=(
            FamilySpec(
                name="family_a",
                source_names=("documents",),
                task_names=("task_a", "task_b"),
            ),
        ),
    )

    class FamilyA(PrepareSpec):
        name = "family_a"
        task_names = ("task_a", "task_b")

        def validate_config(self, config, *, sources, task_names):
            if config.get("mode") not in {"standard", "strict"}:
                raise ValueError("mode must be standard or strict")
            assert tuple(source.name for source in sources) == ("documents",)
            assert task_names == ("task_b",)

        def prepare(self, plan):
            artifact = PreparedArtifactRef(
                family=plan.family,
                uri="artifact://prepared/family-a",
                schema_version=PREPARED_SCHEMA_VERSION,
                row_count=1,
                fingerprint="artifact-fingerprint",
                source_names=tuple(source.name for source in plan.source_refs),
                prepare_plan_fingerprint=plan.fingerprint,
            )
            return make_prepare_result(
                plan,
                artifact,
                stats=PrepareStats(output_rows=1),
            )

    registry = PrepareRegistry()
    assert registry.register(FamilyA) is FamilyA
    assert registry.get("family_a") is not registry.get("family_a")

    prepare_suite = PrepareSuiteSpec(
        run_id="data-run-001",
        families={
            "family_a": FamilyPrepareSpec(
                config={"mode": "strict"},
                task_names=("task_b",),
            ),
        },
    )
    plans = plan_prepare_suite(suite, prepare_suite, registry)

    assert len(plans) == 1
    plan = plans[0]
    assert isinstance(plan, PreparePlan)
    assert plan.family == "family_a"
    assert plan.task_names == ("task_b",)
    assert plan.config == {"mode": "strict"}
    assert plan.run_id == "data-run-001"
    assert plan.suite_fingerprint == suite.fingerprint
    assert len(plan.fingerprint) == 64


def test_prepare_output_validation_and_accounting():
    suite = DataSuiteSpec(
        sources=(
            SourceRef(
                name="records",
                uri="dataset://records",
                version="v1",
            ),
        ),
        families=(
            FamilySpec(
                name="family_a",
                source_names=("records",),
                task_names=("task_a",),
            ),
        ),
    )

    class FamilyA(PrepareSpec):
        name = "family_a"
        task_names = ("task_a",)

        def prepare(self, plan):
            artifact = PreparedArtifactRef(
                family=plan.family,
                uri="artifact://prepared/family-a",
                schema_version=PREPARED_SCHEMA_VERSION,
                row_count=2,
                fingerprint="artifact-fingerprint",
                source_names=("records",),
                prepare_plan_fingerprint=plan.fingerprint,
            )
            return make_prepare_result(
                plan,
                artifact,
                stats=PrepareStats(output_rows=2),
            )

    registry = PrepareRegistry()
    registry.register(FamilyA)
    plan = plan_prepare_suite(
        suite,
        PrepareSuiteSpec(
            families={"family_a": FamilyPrepareSpec()},
        ),
        registry,
    )[0]

    rows = (
        PreparedExample(
            task_name="task_a",
            split="train",
            group_id="group-1",
            input_text="input 1",
            target_text="target 1",
        ),
        PreparedExample(
            task_name="task_a",
            split="validation",
            group_id="group-2",
            input_text="input 2",
            target_text="target 2",
        ),
    )
    stats = validate_prepared_examples(
        rows,
        plan,
        allowed_splits=("train", "validation"),
    )
    assert stats == PrepareStats(
        output_rows=2,
        rows_per_task={"task_a": 2},
        rows_per_split={"train": 1, "validation": 1},
        distinct_groups=2,
    )

    executed = execute_prepare_plan(plan, registry)
    artifact = executed.artifact
    validate_prepared_artifact(artifact, plan, stats=stats)
    result = make_prepare_result(plan, artifact, stats=stats)
    assert result.plan == plan
    assert result.artifact == artifact
    assert result.stats == stats

    parsed, parsed_stats = validate_prepared_mappings(
        (
            {
                "task_name": "task_a",
                "split": "train",
                "group_id": "group-3",
                "input_text": "input 3",
                "options": (),
                "answer_index": -1,
                "target_text": "target 3",
            },
        ),
        plan,
        allowed_splits=("train", "validation"),
    )
    assert parsed[0].group_id == "group-3"
    assert parsed_stats.output_rows == 1

    with pytest.raises(ValueError, match="canonical semantic schema"):
        validate_prepared_mappings(
            (
                {
                    "task_name": "task_a",
                    "split": "train",
                    "group_id": "group-4",
                    "input_text": "input 4",
                    "answer_index": -1,
                    "target_text": "target 4",
                },
            ),
            plan,
        )

    with pytest.raises(ValueError, match="rejection_counts"):
        PrepareStats(
            output_rows=1,
            rejected_rows=2,
            rejection_counts={"invalid_record": 1},
        )

    wrong_schema = PreparedArtifactRef(
        family="family_a",
        uri="artifact://prepared/family-a",
        schema_version="other-schema",
        row_count=2,
        fingerprint="artifact-fingerprint",
        source_names=("records",),
        prepare_plan_fingerprint=plan.fingerprint,
    )
    with pytest.raises(ValueError, match="schema"):
        validate_prepared_artifact(wrong_schema, plan, stats=stats)

    wrong_plan = PreparedArtifactRef(
        family="family_a",
        uri="artifact://prepared/family-a",
        schema_version=PREPARED_SCHEMA_VERSION,
        row_count=2,
        fingerprint="artifact-fingerprint",
        source_names=("records",),
        prepare_plan_fingerprint="different-plan",
    )
    with pytest.raises(ValueError, match="prepare_plan_fingerprint"):
        validate_prepared_artifact(wrong_plan, plan, stats=stats)


def test_prepare_registry_and_suite_fail_loudly_on_contract_mismatch():
    class FamilyA(PrepareSpec):
        name = "family_a"
        task_names = ("task_a",)

        def prepare(self, plan):
            raise AssertionError("not executed")

    class ConflictingFamily(PrepareSpec):
        name = "family_a"
        task_names = ("task_a",)

        def prepare(self, plan):
            raise AssertionError("not executed")

    registry = PrepareRegistry()
    registry.register(FamilyA)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(ConflictingFamily)
    with pytest.raises(KeyError, match="Registered families"):
        registry.get("missing")

    suite = DataSuiteSpec(
        sources=(
            SourceRef(name="records", uri="dataset://records", version="v1"),
        ),
        families=(
            FamilySpec(
                name="family_a",
                source_names=("records",),
                task_names=("task_a",),
            ),
        ),
    )
    with pytest.raises(ValueError, match="absent from DataSuiteSpec"):
        plan_prepare_suite(
            suite,
            PrepareSuiteSpec(
                families={"family_b": FamilyPrepareSpec()},
            ),
            registry,
        )

    with pytest.raises(ValueError, match="unknown tasks"):
        plan_prepare_suite(
            suite,
            PrepareSuiteSpec(
                families={
                    "family_a": FamilyPrepareSpec(
                        task_names=("task_b",),
                    )
                },
            ),
            registry,
        )
