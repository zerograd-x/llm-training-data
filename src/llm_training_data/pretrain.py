from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


TRAIN_SPLIT = "train"
TRAIN_PROBE_SPLIT = "train_probe"
DEFAULT_EVAL_SPLITS = ("validation",)
_FINGERPRINT_VERSION = "sha256-v1"


@dataclass(frozen=True)
class PreparedExample:
    """Semantic pretraining example before prompt rendering/tokenization.

    ``group_id`` identifies the entity/group used for lineage or holdout
    isolation (for example an entity or source document). ``sample_id`` below
    is the unique semantic-row fingerprint and deliberately has different
    semantics from ``group_id``.

    Generation examples use ``target_text`` with empty ``options`` and
    ``answer_index=-1``. Multiple-choice examples use ``options`` plus a valid
    ``answer_index`` and leave ``target_text=None``.
    """

    task_name: str
    split: str
    group_id: str
    input_text: str | None = None
    options: tuple[str, ...] = ()
    answer_index: int = -1
    target_text: str | None = None

    def __post_init__(self) -> None:
        for name in ("task_name", "split", "group_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.input_text is not None and not isinstance(self.input_text, str):
            raise TypeError("input_text must be a string or None")
        if self.target_text is not None and not isinstance(self.target_text, str):
            raise TypeError("target_text must be a string or None")
        if not isinstance(self.answer_index, int):
            raise TypeError("answer_index must be an integer")

        options = tuple(self.options)
        if any(not isinstance(option, str) for option in options):
            raise TypeError("options must contain strings")
        object.__setattr__(self, "options", options)

        if options:
            if self.target_text is not None:
                raise ValueError("Multiple-choice examples must leave target_text=None")
            if not 0 <= self.answer_index < len(options):
                raise ValueError("Multiple-choice answer_index must select one of options")
        else:
            if self.answer_index != -1:
                raise ValueError(
                    "Generation examples without options must use answer_index=-1"
                )
            if self.target_text is None:
                raise ValueError(
                    "Generation examples without options must define target_text"
                )

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "PreparedExample":
        """Build from semantic rows, accepting ``example_id`` as ``group_id``.

        The alias makes the reference contract directly usable with prepared
        datasets that expose the seven semantic columns
        ``task_name,input_text,options,answer_index,target_text,example_id,split``
        while keeping group identity distinct from semantic sample identity in
        the public API.
        """
        if "group_id" in row:
            group_id = row["group_id"]
        elif "example_id" in row:
            group_id = row["example_id"]
        else:
            raise KeyError("Prepared example requires group_id or example_id")
        return cls(
            task_name=row["task_name"],
            split=row["split"],
            group_id=group_id,
            input_text=row.get("input_text"),
            options=tuple(row.get("options") or ()),
            answer_index=row.get("answer_index", -1),
            target_text=row.get("target_text"),
        )

    @property
    def sample_id(self) -> str:
        """Stable identity of the semantic row, independent of split."""
        payload = {
            "version": _FINGERPRINT_VERSION,
            "task_name": self.task_name,
            "group_id": self.group_id,
            "input_text": self.input_text,
            "options": list(self.options),
            "answer_index": self.answer_index,
            "target_text": self.target_text,
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def with_split(self, split: str) -> "PreparedExample":
        return replace(self, split=split)


@dataclass(frozen=True)
class BlendSpec:
    """Requested task mixture for a prepared semantic corpus.

    ``task_row_counts`` is intentionally both the train quota and the task
    allowlist. Tasks absent from the mapping cannot enter train or evaluation
    output.
    """

    task_row_counts: Mapping[str, int]
    eval_rows_per_cell: int = 5_000
    probe_rows_per_task: int = 5_000
    seed: int = 42
    cap_to_available: bool = False
    train_split: str = TRAIN_SPLIT
    probe_split: str = TRAIN_PROBE_SPLIT
    eval_splits: tuple[str, ...] = DEFAULT_EVAL_SPLITS

    def __post_init__(self) -> None:
        counts = dict(self.task_row_counts)
        if not counts:
            raise ValueError("task_row_counts must be non-empty")
        if any(not isinstance(task, str) or not task.strip() for task in counts):
            raise ValueError("task_row_counts keys must be non-empty task names")
        if any(not isinstance(count, int) or count <= 0 for count in counts.values()):
            raise ValueError("task_row_counts values must be positive integers")
        object.__setattr__(self, "task_row_counts", counts)

        if not isinstance(self.eval_rows_per_cell, int) or self.eval_rows_per_cell <= 0:
            raise ValueError("eval_rows_per_cell must be a positive integer")
        if not isinstance(self.probe_rows_per_task, int) or self.probe_rows_per_task <= 0:
            raise ValueError("probe_rows_per_task must be a positive integer")
        if not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if not isinstance(self.cap_to_available, bool):
            raise TypeError("cap_to_available must be a boolean")
        if not isinstance(self.train_split, str) or not self.train_split.strip():
            raise ValueError("train_split must be non-empty")
        if not isinstance(self.probe_split, str) or not self.probe_split.strip():
            raise ValueError("probe_split must be non-empty")

        eval_splits = tuple(self.eval_splits)
        if not eval_splits or any(
            not isinstance(split, str) or not split.strip() for split in eval_splits
        ):
            raise ValueError("eval_splits must contain non-empty split names")
        if len(set(eval_splits)) != len(eval_splits):
            raise ValueError("eval_splits must not contain duplicates")
        if self.train_split == self.probe_split:
            raise ValueError("train_split and probe_split must differ")
        if self.train_split in eval_splits or self.probe_split in eval_splits:
            raise ValueError("train/probe splits must be distinct from eval_splits")
        object.__setattr__(self, "eval_splits", eval_splits)


@dataclass(frozen=True)
class DataPlanWarning:
    code: str
    message: str


@dataclass(frozen=True)
class CellPlan:
    task_name: str
    split: str
    available: int
    available_groups: int
    requested_cap: int
    selected: int
    selected_groups: int


@dataclass(frozen=True)
class DataPlan:
    """Resolved data distribution produced by ``blend_prepared_examples``."""

    fingerprint_algorithm: str
    seed: int
    task_row_counts: dict[str, int]
    eval_rows_per_cell: int
    probe_rows_per_task: int
    input_rows: int
    whitelisted_rows: int
    selected_rows: int
    cells: tuple[CellPlan, ...]
    warnings: tuple[DataPlanWarning, ...]

    @property
    def rows_per_cell(self) -> dict[str, int]:
        return {
            f"{cell.task_name}/{cell.split}": cell.selected
            for cell in self.cells
        }

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class BlendResult:
    examples: tuple[PreparedExample, ...]
    plan: DataPlan


def sampling_rank_key(example: PreparedExample, seed: int) -> int:
    """Deterministic pseudo-random rank key for one task/split cell.

    The semantic row is serialized canonically before hashing, avoiding the
    ambiguous string concatenation that can make distinct option arrays look
    identical. ``split`` is mixed into the rank so train and evaluation cells
    are independently ordered even when they contain the same semantic row.
    """
    payload = {
        "version": _FINGERPRINT_VERSION,
        "task_name": example.task_name,
        "split": example.split,
        "sample_id": example.sample_id,
        "seed": seed,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")


def resolve_cell_cap(task_name: str, split: str, spec: BlendSpec) -> int:
    """Resolve one task/cell cap; absent tasks are excluded everywhere."""
    if task_name not in spec.task_row_counts:
        return 0
    if split == spec.train_split:
        return spec.task_row_counts[task_name]
    if split in spec.eval_splits:
        return spec.eval_rows_per_cell
    return 0


def _ranked(examples: Iterable[PreparedExample], seed: int) -> list[PreparedExample]:
    return sorted(
        examples,
        key=lambda example: (sampling_rank_key(example, seed), example.sample_id),
    )


def _distinct_groups(examples: Iterable[PreparedExample]) -> int:
    return len({example.group_id for example in examples})


def _validate_unique_samples(examples: Iterable[PreparedExample]) -> None:
    seen: set[tuple[str, str, str]] = set()
    for example in examples:
        key = (example.task_name, example.split, example.sample_id)
        if key in seen:
            raise ValueError(
                "Prepared corpus contains an exact duplicate semantic sample in "
                f"task={example.task_name!r} split={example.split!r}: "
                f"sample_id={example.sample_id}"
            )
        seen.add(key)


def blend_prepared_examples(
    examples: Iterable[PreparedExample],
    spec: BlendSpec,
) -> BlendResult:
    """Resolve a deterministic task/split blend and its inspectable DataPlan.

    This is the platform-independent reference implementation. Large-scale
    executors can implement the same contract while preserving the fingerprint,
    quota, whitelist, supply, group-coverage, and probe-subset invariants.
    """
    source = tuple(examples)

    # Whitelist first. Excluded tasks must not influence split validation,
    # supply accounting, sampling, or evaluation output.
    whitelisted = tuple(
        example for example in source if example.task_name in spec.task_row_counts
    )

    if any(example.split == spec.probe_split for example in whitelisted):
        raise ValueError(
            f"Prepared corpus must not contain {spec.probe_split!r}; probe rows "
            "are derived from selected train rows"
        )

    allowed_source_splits = {spec.train_split, *spec.eval_splits}
    unknown_splits = sorted(
        {
            example.split
            for example in whitelisted
            if example.split not in allowed_source_splits
        }
    )
    if unknown_splits:
        raise ValueError(f"Prepared corpus contains unknown splits: {unknown_splits}")

    _validate_unique_samples(whitelisted)

    grouped: dict[tuple[str, str], list[PreparedExample]] = defaultdict(list)
    for example in whitelisted:
        grouped[(example.task_name, example.split)].append(example)

    warnings: list[DataPlanWarning] = []
    for task_name, requested in sorted(spec.task_row_counts.items()):
        available = len(grouped[(task_name, spec.train_split)])
        if available == 0:
            raise ValueError(
                f"task {task_name!r} has 0 rows in the {spec.train_split!r} cell"
            )
        if available < requested:
            message = (
                f"task {task_name!r} has {available} {spec.train_split} rows "
                f"< requested {requested}"
            )
            if not spec.cap_to_available:
                raise ValueError(message)
            warnings.append(DataPlanWarning("train_supply_shortfall", message))

    selected: list[PreparedExample] = []
    cell_plans: list[CellPlan] = []
    selected_train_by_task: dict[str, list[PreparedExample]] = {}

    for task_name in sorted(spec.task_row_counts):
        for split in (spec.train_split, *spec.eval_splits):
            pool = grouped[(task_name, split)]
            cap = resolve_cell_cap(task_name, split, spec)
            chosen = _ranked(pool, spec.seed)[:cap]
            selected.extend(chosen)
            cell_plans.append(
                CellPlan(
                    task_name=task_name,
                    split=split,
                    available=len(pool),
                    available_groups=_distinct_groups(pool),
                    requested_cap=cap,
                    selected=len(chosen),
                    selected_groups=_distinct_groups(chosen),
                )
            )
            if split == spec.train_split:
                selected_train_by_task[task_name] = chosen

    probes: list[PreparedExample] = []
    for task_name in sorted(spec.task_row_counts):
        train_rows = selected_train_by_task[task_name]
        probe_cap = min(spec.probe_rows_per_task, spec.task_row_counts[task_name])
        probe_count = min(len(train_rows), probe_cap)
        task_probes = [
            example.with_split(spec.probe_split)
            for example in train_rows[:probe_count]
        ]
        probes.extend(task_probes)
        cell_plans.append(
            CellPlan(
                task_name=task_name,
                split=spec.probe_split,
                available=len(train_rows),
                available_groups=_distinct_groups(train_rows),
                requested_cap=probe_cap,
                selected=probe_count,
                selected_groups=_distinct_groups(task_probes),
            )
        )

    blended = tuple(selected + probes)
    plan = DataPlan(
        fingerprint_algorithm=_FINGERPRINT_VERSION,
        seed=spec.seed,
        task_row_counts=dict(spec.task_row_counts),
        eval_rows_per_cell=spec.eval_rows_per_cell,
        probe_rows_per_task=spec.probe_rows_per_task,
        input_rows=len(source),
        whitelisted_rows=len(whitelisted),
        selected_rows=len(blended),
        cells=tuple(cell_plans),
        warnings=tuple(warnings),
    )
    return BlendResult(examples=blended, plan=plan)


def format_data_plan(plan: DataPlan) -> str:
    """Render the resolved data distribution for logs/reviews."""
    lines = [
        "EFFECTIVE DATA PLAN",
        "",
        "SELECTION",
        f"  fingerprint: {plan.fingerprint_algorithm}",
        f"  seed: {plan.seed}",
        f"  input rows: {plan.input_rows}",
        f"  whitelisted rows: {plan.whitelisted_rows}",
        f"  selected rows (including probe copies): {plan.selected_rows}",
        "",
        "CELLS",
    ]
    for cell in plan.cells:
        lines.append(
            f"  {cell.task_name}/{cell.split}: "
            f"available={cell.available} groups={cell.available_groups} "
            f"cap={cell.requested_cap} selected={cell.selected} "
            f"selected_groups={cell.selected_groups}"
        )
    lines.extend(["", "WARNINGS"])
    if not plan.warnings:
        lines.append("  none")
    else:
        for warning in plan.warnings:
            lines.append(f"  [{warning.code}] {warning.message}")
    return "\n".join(lines)


def save_data_plan(plan: DataPlan, output: str | Path) -> Path:
    """Persist ``data-plan.json`` next to prepared/blended training data."""
    path = Path(output)
    if path.suffix != ".json":
        path = path / "data-plan.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(plan.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path
