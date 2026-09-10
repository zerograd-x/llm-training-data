from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping


TRAIN_SPLIT = "train"
TRAIN_PROBE_SPLIT = "train_probe"
DEFAULT_EVAL_SPLITS = ("validation",)
_FINGERPRINT_VERSION = "sha256-v1"
ShortfallPolicy = Literal["error", "cap"]


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_non_empty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


@dataclass(frozen=True)
class SourceRef:
    """Reference to one upstream data snapshot.

    ``uri`` is storage-neutral. At least one snapshot/version/fingerprint field
    is required so a data suite does not silently mean "whatever is latest".
    """

    name: str
    uri: str
    version: str | None = None
    snapshot: str | None = None
    fingerprint: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.name, "source.name")
        _require_non_empty(self.uri, "source.uri")
        for name in ("version", "snapshot", "fingerprint"):
            value = getattr(self, name)
            if value is not None:
                _require_non_empty(value, f"source.{name}")
        if not any((self.version, self.snapshot, self.fingerprint)):
            raise ValueError(
                "SourceRef requires at least one of version, snapshot, or fingerprint"
            )
        metadata = dict(self.metadata)
        if any(
            not isinstance(key, str)
            or not key.strip()
            or not isinstance(value, str)
            for key, value in metadata.items()
        ):
            raise ValueError("source.metadata must map non-empty string keys to strings")
        object.__setattr__(self, "metadata", metadata)


@dataclass(frozen=True)
class FamilySpec:
    """One prepare-stage family and the semantic tasks it can materialize."""

    name: str
    source_names: tuple[str, ...]
    task_names: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_non_empty(self.name, "family.name")
        source_names = tuple(self.source_names)
        task_names = tuple(self.task_names)
        if not source_names or any(
            not isinstance(name, str) or not name.strip() for name in source_names
        ):
            raise ValueError("family.source_names must contain non-empty source names")
        if not task_names or any(
            not isinstance(name, str) or not name.strip() for name in task_names
        ):
            raise ValueError("family.task_names must contain non-empty task names")
        if len(set(source_names)) != len(source_names):
            raise ValueError("family.source_names must not contain duplicates")
        if len(set(task_names)) != len(task_names):
            raise ValueError("family.task_names must not contain duplicates")
        object.__setattr__(self, "source_names", source_names)
        object.__setattr__(self, "task_names", task_names)


@dataclass(frozen=True)
class DataSuiteSpec:
    """Resolved source snapshots and family/task ownership for one data suite."""

    sources: tuple[SourceRef, ...]
    families: tuple[FamilySpec, ...]
    suite_id: str | None = None

    def __post_init__(self) -> None:
        sources = tuple(self.sources)
        families = tuple(self.families)
        if not sources:
            raise ValueError("DataSuiteSpec.sources must be non-empty")
        if not families:
            raise ValueError("DataSuiteSpec.families must be non-empty")
        if self.suite_id is not None:
            _require_non_empty(self.suite_id, "suite_id")

        source_names = [source.name for source in sources]
        if len(set(source_names)) != len(source_names):
            raise ValueError("DataSuiteSpec source names must be unique")
        family_names = [family.name for family in families]
        if len(set(family_names)) != len(family_names):
            raise ValueError("DataSuiteSpec family names must be unique")

        known_sources = set(source_names)
        unknown_sources = sorted(
            {
                source_name
                for family in families
                for source_name in family.source_names
                if source_name not in known_sources
            }
        )
        if unknown_sources:
            raise ValueError(
                f"DataSuiteSpec families reference unknown sources: {unknown_sources}"
            )

        task_owners: dict[str, str] = {}
        for family in families:
            for task_name in family.task_names:
                previous = task_owners.get(task_name)
                if previous is not None:
                    raise ValueError(
                        f"task {task_name!r} belongs to both {previous!r} and "
                        f"{family.name!r}"
                    )
                task_owners[task_name] = family.name

        object.__setattr__(self, "sources", sources)
        object.__setattr__(self, "families", families)

    @property
    def task_to_family(self) -> dict[str, str]:
        return {
            task_name: family.name
            for family in self.families
            for task_name in family.task_names
        }

    @property
    def fingerprint(self) -> str:
        sources = [asdict(source) for source in sorted(self.sources, key=lambda x: x.name)]
        families = []
        for family in sorted(self.families, key=lambda x: x.name):
            families.append(
                {
                    "name": family.name,
                    "source_names": sorted(family.source_names),
                    "task_names": sorted(family.task_names),
                }
            )
        return _canonical_sha256({"sources": sources, "families": families})


@dataclass(frozen=True)
class PreparedArtifactRef:
    """Reference to a materialized family-level prepared semantic corpus."""

    family: str
    uri: str
    schema_version: str
    row_count: int
    fingerprint: str
    source_names: tuple[str, ...] = ()
    prepare_plan_fingerprint: str | None = None

    def __post_init__(self) -> None:
        for name in ("family", "uri", "schema_version", "fingerprint"):
            _require_non_empty(getattr(self, name), f"prepared_artifact.{name}")
        if self.prepare_plan_fingerprint is not None:
            _require_non_empty(
                self.prepare_plan_fingerprint,
                "prepared_artifact.prepare_plan_fingerprint",
            )
        if not isinstance(self.row_count, int) or self.row_count < 0:
            raise ValueError("prepared_artifact.row_count must be >= 0")
        source_names = tuple(self.source_names)
        if any(not isinstance(name, str) or not name.strip() for name in source_names):
            raise ValueError(
                "prepared_artifact.source_names must contain non-empty names"
            )
        if len(set(source_names)) != len(source_names):
            raise ValueError("prepared_artifact.source_names must not contain duplicates")
        object.__setattr__(self, "source_names", source_names)


@dataclass(frozen=True)
class EvaluationCellSpec:
    """Named evaluation cell plus explicit experimental dimensions.

    ``task_names=None`` means the cell applies to every selected task. Otherwise
    it explicitly identifies the tasks for which the cell is meaningful, making
    "not applicable" distinct from "expected data was missing".
    """

    name: str
    dimensions: Mapping[str, str] = field(default_factory=dict)
    description: str | None = None
    task_names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        _require_non_empty(self.name, "evaluation_cell.name")
        if self.description is not None:
            _require_non_empty(self.description, "evaluation_cell.description")
        dimensions = dict(self.dimensions)
        if any(
            not isinstance(key, str)
            or not key.strip()
            or not isinstance(value, str)
            or not value.strip()
            for key, value in dimensions.items()
        ):
            raise ValueError(
                "evaluation_cell.dimensions must map non-empty strings to non-empty strings"
            )
        object.__setattr__(self, "dimensions", dimensions)

        if self.task_names is not None:
            task_names = tuple(self.task_names)
            if not task_names or any(
                not isinstance(task_name, str) or not task_name.strip()
                for task_name in task_names
            ):
                raise ValueError(
                    "evaluation_cell.task_names must contain non-empty task names"
                )
            if len(set(task_names)) != len(task_names):
                raise ValueError("evaluation_cell.task_names must not contain duplicates")
            object.__setattr__(self, "task_names", task_names)

    def applies_to(self, task_name: str) -> bool:
        return self.task_names is None or task_name in self.task_names


@dataclass(frozen=True)
class TaskBlendSpec:
    """Train quota and supply behavior for one semantic task."""

    train_rows: int
    shortfall_policy: ShortfallPolicy = "error"
    eval_rows_per_cell: int | None = None
    probe_rows: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.train_rows, int) or self.train_rows <= 0:
            raise ValueError("task_blend.train_rows must be a positive integer")
        if self.shortfall_policy not in {"error", "cap"}:
            raise ValueError("task_blend.shortfall_policy must be 'error' or 'cap'")
        for name in ("eval_rows_per_cell", "probe_rows"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, int) or value < 0):
                raise ValueError(f"task_blend.{name} must be >= 0 when set")


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
        """Build from semantic rows, accepting ``example_id`` as ``group_id``."""
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
        return _canonical_sha256(payload)

    def with_split(self, split: str) -> "PreparedExample":
        return replace(self, split=split)


@dataclass(frozen=True)
class BlendSpec:
    """Requested task/cell mixture for a prepared semantic corpus.

    Legacy fields retain their original positional order. ``task_specs`` and
    ``evaluation_cells`` are the preferred richer API and are placed at the end
    so existing positional construction remains compatible.
    """

    task_row_counts: Mapping[str, int] | None = None
    eval_rows_per_cell: int = 5_000
    probe_rows_per_task: int = 5_000
    seed: int = 42
    cap_to_available: bool | None = None
    train_split: str = TRAIN_SPLIT
    probe_split: str = TRAIN_PROBE_SPLIT
    eval_splits: tuple[str, ...] | None = None
    task_specs: Mapping[str, TaskBlendSpec | Mapping[str, Any]] | None = None
    evaluation_cells: tuple[EvaluationCellSpec | Mapping[str, Any], ...] | None = None

    def __post_init__(self) -> None:
        if self.task_row_counts is None and self.task_specs is None:
            raise ValueError("Specify task_specs or task_row_counts")
        if self.task_row_counts is not None and self.task_specs is not None:
            raise ValueError("Specify task_specs or task_row_counts, not both")
        if not isinstance(self.eval_rows_per_cell, int) or self.eval_rows_per_cell <= 0:
            raise ValueError("eval_rows_per_cell must be a positive integer")
        if not isinstance(self.probe_rows_per_task, int) or self.probe_rows_per_task <= 0:
            raise ValueError("probe_rows_per_task must be a positive integer")
        if not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if self.cap_to_available is not None and not isinstance(
            self.cap_to_available, bool
        ):
            raise TypeError("cap_to_available must be a boolean or None")
        _require_non_empty(self.train_split, "train_split")
        _require_non_empty(self.probe_split, "probe_split")
        if self.train_split == self.probe_split:
            raise ValueError("train_split and probe_split must differ")

        normalized_tasks: dict[str, TaskBlendSpec] = {}
        if self.task_specs is not None:
            if self.cap_to_available is not None:
                raise ValueError(
                    "cap_to_available is a legacy global policy; do not combine it "
                    "with per-task task_specs"
                )
            for task_name, value in dict(self.task_specs).items():
                _require_non_empty(task_name, "task_specs key")
                if isinstance(value, TaskBlendSpec):
                    normalized = value
                elif isinstance(value, Mapping):
                    normalized = TaskBlendSpec(**dict(value))
                else:
                    raise TypeError(
                        "task_specs values must be TaskBlendSpec or mappings"
                    )
                normalized_tasks[task_name] = normalized
            if not normalized_tasks:
                raise ValueError("task_specs must be non-empty")
            object.__setattr__(self, "cap_to_available", None)
        else:
            counts = dict(self.task_row_counts or {})
            if not counts:
                raise ValueError("task_row_counts must be non-empty")
            if any(not isinstance(task, str) or not task.strip() for task in counts):
                raise ValueError("task_row_counts keys must be non-empty task names")
            if any(
                not isinstance(count, int) or count <= 0 for count in counts.values()
            ):
                raise ValueError("task_row_counts values must be positive integers")
            policy: ShortfallPolicy = "cap" if self.cap_to_available else "error"
            normalized_tasks = {
                task_name: TaskBlendSpec(
                    train_rows=count,
                    shortfall_policy=policy,
                )
                for task_name, count in counts.items()
            }
            object.__setattr__(
                self,
                "cap_to_available",
                bool(self.cap_to_available),
            )

        if self.eval_splits is not None and self.evaluation_cells is not None:
            raise ValueError("Specify evaluation_cells or eval_splits, not both")
        if self.evaluation_cells is not None:
            cells: list[EvaluationCellSpec] = []
            for value in self.evaluation_cells:
                if isinstance(value, EvaluationCellSpec):
                    cell = value
                elif isinstance(value, Mapping):
                    cell = EvaluationCellSpec(**dict(value))
                else:
                    raise TypeError(
                        "evaluation_cells must contain EvaluationCellSpec or mappings"
                    )
                cells.append(cell)
        elif self.eval_splits is not None:
            cells = [EvaluationCellSpec(name=name) for name in self.eval_splits]
        else:
            cells = [EvaluationCellSpec(name=name) for name in DEFAULT_EVAL_SPLITS]

        if not cells:
            raise ValueError("At least one evaluation cell is required")
        cell_names = [cell.name for cell in cells]
        if len(set(cell_names)) != len(cell_names):
            raise ValueError("evaluation cell names must be unique")
        if self.train_split in cell_names or self.probe_split in cell_names:
            raise ValueError("train/probe splits must be distinct from evaluation cells")

        object.__setattr__(self, "task_specs", normalized_tasks)
        object.__setattr__(
            self,
            "task_row_counts",
            {task: value.train_rows for task, value in normalized_tasks.items()},
        )
        object.__setattr__(self, "evaluation_cells", tuple(cells))
        object.__setattr__(self, "eval_splits", tuple(cell_names))


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
    dimensions: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class DataPlan:
    """Resolved data distribution produced by ``blend_prepared_examples``."""

    fingerprint_algorithm: str
    seed: int
    task_row_counts: dict[str, int]
    task_specs: dict[str, TaskBlendSpec]
    evaluation_cells: tuple[EvaluationCellSpec, ...]
    eval_rows_per_cell: int
    probe_rows_per_task: int
    input_rows: int
    whitelisted_rows: int
    selected_rows: int
    cells: tuple[CellPlan, ...]
    warnings: tuple[DataPlanWarning, ...]
    suite: DataSuiteSpec | None = None
    prepared_artifacts: tuple[PreparedArtifactRef, ...] = ()

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
    """Deterministic pseudo-random rank key for one task/split cell."""
    payload = {
        "version": _FINGERPRINT_VERSION,
        "task_name": example.task_name,
        "split": example.split,
        "sample_id": example.sample_id,
        "seed": seed,
    }
    return int.from_bytes(bytes.fromhex(_canonical_sha256(payload))[:8], "big")


def _evaluation_cell(spec: BlendSpec, split: str) -> EvaluationCellSpec | None:
    for cell in spec.evaluation_cells or ():
        if cell.name == split:
            return cell
    return None


def resolve_cell_cap(task_name: str, split: str, spec: BlendSpec) -> int:
    """Resolve one task/cell cap; absent or non-applicable tasks get zero."""
    task_spec = spec.task_specs.get(task_name) if spec.task_specs is not None else None
    if task_spec is None:
        return 0
    if split == spec.train_split:
        return task_spec.train_rows
    if split == spec.probe_split:
        return (
            task_spec.probe_rows
            if task_spec.probe_rows is not None
            else spec.probe_rows_per_task
        )
    cell = _evaluation_cell(spec, split)
    if cell is not None and cell.applies_to(task_name):
        return (
            task_spec.eval_rows_per_cell
            if task_spec.eval_rows_per_cell is not None
            else spec.eval_rows_per_cell
        )
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


def _validate_suite_lineage(
    spec: BlendSpec,
    suite: DataSuiteSpec | None,
    prepared_artifacts: tuple[PreparedArtifactRef, ...],
) -> None:
    families = [artifact.family for artifact in prepared_artifacts]
    if len(set(families)) != len(families):
        raise ValueError("prepared_artifacts must contain at most one artifact per family")
    if suite is None:
        return

    task_to_family = suite.task_to_family
    missing_tasks = sorted(
        task_name for task_name in spec.task_specs if task_name not in task_to_family
    )
    if missing_tasks:
        raise ValueError(f"BlendSpec contains tasks absent from DataSuiteSpec: {missing_tasks}")

    if not prepared_artifacts:
        return
    family_specs = {family.name: family for family in suite.families}
    unknown_families = sorted(
        artifact.family
        for artifact in prepared_artifacts
        if artifact.family not in family_specs
    )
    if unknown_families:
        raise ValueError(
            f"prepared_artifacts reference unknown families: {unknown_families}"
        )

    artifacts_by_family = {artifact.family: artifact for artifact in prepared_artifacts}
    required_families = {task_to_family[task_name] for task_name in spec.task_specs}
    missing_families = sorted(required_families - artifacts_by_family.keys())
    if missing_families:
        raise ValueError(
            f"prepared_artifacts are missing required families: {missing_families}"
        )

    for family_name in required_families:
        artifact = artifacts_by_family[family_name]
        if artifact.source_names:
            expected = set(family_specs[family_name].source_names)
            actual = set(artifact.source_names)
            if actual != expected:
                raise ValueError(
                    f"prepared artifact {family_name!r} source lineage mismatch: "
                    f"expected {sorted(expected)}, got {sorted(actual)}"
                )


def _validate_cell_applicability(examples: Iterable[PreparedExample], spec: BlendSpec) -> None:
    cells = {cell.name: cell for cell in spec.evaluation_cells or ()}
    for example in examples:
        cell = cells.get(example.split)
        if cell is not None and not cell.applies_to(example.task_name):
            raise ValueError(
                f"task {example.task_name!r} is not applicable to evaluation cell "
                f"{example.split!r}"
            )


def blend_prepared_examples(
    examples: Iterable[PreparedExample],
    spec: BlendSpec,
    *,
    suite: DataSuiteSpec | None = None,
    prepared_artifacts: Iterable[PreparedArtifactRef] = (),
) -> BlendResult:
    """Resolve a deterministic task/cell blend and its inspectable DataPlan.

    This in-memory implementation is the reference contract. Large-scale
    executors may materialize family-level prepared artifacts and perform the
    same selection externally while preserving the same lineage, quota,
    whitelist, supply, group-coverage, and probe-subset semantics.
    """
    source = tuple(examples)
    artifacts = tuple(prepared_artifacts)
    _validate_suite_lineage(spec, suite, artifacts)

    whitelisted = tuple(
        example for example in source if example.task_name in spec.task_specs
    )

    if any(example.split == spec.probe_split for example in whitelisted):
        raise ValueError(
            f"Prepared corpus must not contain {spec.probe_split!r}; probe rows "
            "are derived from selected train rows"
        )

    allowed_source_splits = {spec.train_split, *(spec.eval_splits or ())}
    unknown_splits = sorted(
        {
            example.split
            for example in whitelisted
            if example.split not in allowed_source_splits
        }
    )
    if unknown_splits:
        raise ValueError(f"Prepared corpus contains unknown splits: {unknown_splits}")

    _validate_cell_applicability(whitelisted, spec)
    _validate_unique_samples(whitelisted)

    grouped: dict[tuple[str, str], list[PreparedExample]] = defaultdict(list)
    for example in whitelisted:
        grouped[(example.task_name, example.split)].append(example)

    warnings: list[DataPlanWarning] = []
    for task_name, task_spec in sorted(spec.task_specs.items()):
        available = len(grouped[(task_name, spec.train_split)])
        if available == 0:
            raise ValueError(
                f"task {task_name!r} has 0 rows in the {spec.train_split!r} cell"
            )
        if available < task_spec.train_rows:
            message = (
                f"task {task_name!r} has {available} {spec.train_split} rows "
                f"< requested {task_spec.train_rows}"
            )
            if task_spec.shortfall_policy == "error":
                raise ValueError(message)
            warnings.append(DataPlanWarning("train_supply_shortfall", message))

    selected: list[PreparedExample] = []
    cell_plans: list[CellPlan] = []
    selected_train_by_task: dict[str, list[PreparedExample]] = {}

    for task_name in sorted(spec.task_specs):
        train_pool = grouped[(task_name, spec.train_split)]
        train_cap = resolve_cell_cap(task_name, spec.train_split, spec)
        train_rows = _ranked(train_pool, spec.seed)[:train_cap]
        selected.extend(train_rows)
        selected_train_by_task[task_name] = train_rows
        cell_plans.append(
            CellPlan(
                task_name=task_name,
                split=spec.train_split,
                available=len(train_pool),
                available_groups=_distinct_groups(train_pool),
                requested_cap=train_cap,
                selected=len(train_rows),
                selected_groups=_distinct_groups(train_rows),
            )
        )

        for cell in spec.evaluation_cells or ():
            if not cell.applies_to(task_name):
                continue
            pool = grouped[(task_name, cell.name)]
            cap = resolve_cell_cap(task_name, cell.name, spec)
            chosen = _ranked(pool, spec.seed)[:cap]
            selected.extend(chosen)
            cell_plans.append(
                CellPlan(
                    task_name=task_name,
                    split=cell.name,
                    available=len(pool),
                    available_groups=_distinct_groups(pool),
                    requested_cap=cap,
                    selected=len(chosen),
                    selected_groups=_distinct_groups(chosen),
                    dimensions=dict(cell.dimensions),
                )
            )

    probes: list[PreparedExample] = []
    for task_name, task_spec in sorted(spec.task_specs.items()):
        train_rows = selected_train_by_task[task_name]
        requested_probe = resolve_cell_cap(task_name, spec.probe_split, spec)
        probe_cap = min(requested_probe, task_spec.train_rows)
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
        task_row_counts=dict(spec.task_row_counts or {}),
        task_specs=dict(spec.task_specs),
        evaluation_cells=tuple(spec.evaluation_cells or ()),
        eval_rows_per_cell=spec.eval_rows_per_cell,
        probe_rows_per_task=spec.probe_rows_per_task,
        input_rows=len(source),
        whitelisted_rows=len(whitelisted),
        selected_rows=len(blended),
        cells=tuple(cell_plans),
        warnings=tuple(warnings),
        suite=suite,
        prepared_artifacts=artifacts,
    )
    return BlendResult(examples=blended, plan=plan)


def format_data_plan(plan: DataPlan) -> str:
    """Render source lineage and the resolved data distribution for review."""
    lines = ["EFFECTIVE DATA PLAN"]

    if plan.suite is not None:
        lines.extend(
            [
                "",
                "SUITE",
                f"  id: {plan.suite.suite_id or '-'}",
                f"  fingerprint: {plan.suite.fingerprint}",
                "",
                "SOURCES",
            ]
        )
        for source in plan.suite.sources:
            identity = source.snapshot or source.version or source.fingerprint or "-"
            lines.append(f"  {source.name}: {source.uri} [{identity}]")

    if plan.prepared_artifacts:
        lines.extend(["", "PREPARED ARTIFACTS"])
        for artifact in plan.prepared_artifacts:
            plan_suffix = (
                f" plan={artifact.prepare_plan_fingerprint}"
                if artifact.prepare_plan_fingerprint is not None
                else ""
            )
            lines.append(
                f"  {artifact.family}: rows={artifact.row_count} "
                f"schema={artifact.schema_version} uri={artifact.uri}"
                f"{plan_suffix}"
            )

    lines.extend(
        [
            "",
            "SELECTION",
            f"  fingerprint: {plan.fingerprint_algorithm}",
            f"  seed: {plan.seed}",
            f"  input rows: {plan.input_rows}",
            f"  whitelisted rows: {plan.whitelisted_rows}",
            f"  selected rows (including probe copies): {plan.selected_rows}",
            "",
            "TASKS",
        ]
    )
    for task_name, task_spec in sorted(plan.task_specs.items()):
        lines.append(
            f"  {task_name}: train={task_spec.train_rows} "
            f"shortfall={task_spec.shortfall_policy}"
        )

    lines.extend(["", "CELLS"])
    for cell in plan.cells:
        dimensions = ""
        if cell.dimensions:
            rendered = ", ".join(
                f"{key}={value}" for key, value in sorted(cell.dimensions.items())
            )
            dimensions = f" dimensions=[{rendered}]"
        lines.append(
            f"  {cell.task_name}/{cell.split}: "
            f"available={cell.available} groups={cell.available_groups} "
            f"cap={cell.requested_cap} selected={cell.selected} "
            f"selected_groups={cell.selected_groups}{dimensions}"
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
