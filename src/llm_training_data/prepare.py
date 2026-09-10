from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import asdict, dataclass, field
import hashlib
import inspect
import json
from typing import Any, ClassVar, Iterable, Mapping

from .pretrain import (
    TRAIN_PROBE_SPLIT,
    DataSuiteSpec,
    FamilySpec,
    PreparedArtifactRef,
    PreparedExample,
    SourceRef,
)


PREPARED_SCHEMA_VERSION = "semantic-v1"
PREPARED_SCHEMA_FIELDS = (
    "task_name",
    "split",
    "group_id",
    "input_text",
    "options",
    "answer_index",
    "target_text",
)


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


def _normalize_string_counts(
    values: Mapping[str, int],
    *,
    name: str,
) -> dict[str, int]:
    normalized = dict(values)
    for key, value in normalized.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{name} keys must be non-empty strings")
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} values must be non-negative integers")
    return normalized


@dataclass(frozen=True)
class FamilyPrepareSpec:
    """Run-specific configuration for one enabled prepare family.

    task_names=None means prepare every task declared by the family. A subset
    lets callers avoid materializing unused variants before the blend stage.
    """

    config: Mapping[str, Any] = field(default_factory=dict)
    task_names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        config = dict(self.config)
        try:
            _canonical_sha256(config)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "family prepare config must be JSON-serializable so it can be "
                "fingerprinted and persisted"
            ) from exc
        object.__setattr__(self, "config", config)

        if self.task_names is not None:
            task_names = tuple(self.task_names)
            if not task_names or any(
                not isinstance(task_name, str) or not task_name.strip()
                for task_name in task_names
            ):
                raise ValueError(
                    "family prepare task_names must contain non-empty task names"
                )
            if len(set(task_names)) != len(task_names):
                raise ValueError(
                    "family prepare task_names must not contain duplicates"
                )
            object.__setattr__(self, "task_names", task_names)


@dataclass(frozen=True)
class PrepareSuiteSpec:
    """Enabled prepare families and their run-specific configuration."""

    families: Mapping[str, FamilyPrepareSpec]
    run_id: str | None = None

    def __post_init__(self) -> None:
        if self.run_id is not None:
            _require_non_empty(self.run_id, "prepare_suite.run_id")

        families = dict(self.families)
        if not families:
            raise ValueError("prepare_suite.families must be non-empty")
        for name, value in families.items():
            _require_non_empty(name, "prepare_suite family name")
            if not isinstance(value, FamilyPrepareSpec):
                raise TypeError(
                    "prepare_suite.families values must be FamilyPrepareSpec"
                )
        object.__setattr__(self, "families", families)

    @property
    def fingerprint(self) -> str:
        return _canonical_sha256(
            {
                "run_id": self.run_id,
                "families": {
                    name: {
                        "config": dict(spec.config),
                        "task_names": (
                            sorted(spec.task_names)
                            if spec.task_names is not None
                            else None
                        ),
                    }
                    for name, spec in sorted(self.families.items())
                },
            }
        )


@dataclass(frozen=True)
class PreparePlan:
    """Resolved execution contract for one family-level prepare operation."""

    family: str
    source_refs: tuple[SourceRef, ...]
    task_names: tuple[str, ...]
    config: Mapping[str, Any]
    suite_fingerprint: str
    run_id: str | None = None

    def __post_init__(self) -> None:
        _require_non_empty(self.family, "prepare_plan.family")
        _require_non_empty(
            self.suite_fingerprint,
            "prepare_plan.suite_fingerprint",
        )
        if self.run_id is not None:
            _require_non_empty(self.run_id, "prepare_plan.run_id")

        source_refs = tuple(self.source_refs)
        if not source_refs:
            raise ValueError("prepare_plan.source_refs must be non-empty")
        source_names = [source.name for source in source_refs]
        if len(set(source_names)) != len(source_names):
            raise ValueError("prepare_plan.source_refs must have unique names")
        object.__setattr__(self, "source_refs", source_refs)

        task_names = tuple(self.task_names)
        if not task_names or any(
            not isinstance(task_name, str) or not task_name.strip()
            for task_name in task_names
        ):
            raise ValueError(
                "prepare_plan.task_names must contain non-empty task names"
            )
        if len(set(task_names)) != len(task_names):
            raise ValueError("prepare_plan.task_names must not contain duplicates")
        object.__setattr__(self, "task_names", task_names)

        config = dict(self.config)
        try:
            _canonical_sha256(config)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "prepare_plan.config must be JSON-serializable"
            ) from exc
        object.__setattr__(self, "config", config)

    @property
    def fingerprint(self) -> str:
        return _canonical_sha256(
            {
                "family": self.family,
                "sources": [asdict(source) for source in self.source_refs],
                "task_names": list(self.task_names),
                "config": dict(self.config),
                "suite_fingerprint": self.suite_fingerprint,
                "run_id": self.run_id,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PrepareStats:
    """Inspectable accounting for one prepare operation.

    Rejection reason names are application-defined. The core only validates
    that the accounting is well formed.
    """

    output_rows: int
    input_rows: int | None = None
    rejected_rows: int = 0
    rejection_counts: Mapping[str, int] = field(default_factory=dict)
    rows_per_task: Mapping[str, int] = field(default_factory=dict)
    rows_per_split: Mapping[str, int] = field(default_factory=dict)
    distinct_groups: int | None = None

    def __post_init__(self) -> None:
        for name in ("output_rows", "rejected_rows"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"prepare_stats.{name} must be >= 0")
        if self.input_rows is not None and (
            not isinstance(self.input_rows, int) or self.input_rows < 0
        ):
            raise ValueError("prepare_stats.input_rows must be >= 0 when set")
        if self.distinct_groups is not None and (
            not isinstance(self.distinct_groups, int)
            or self.distinct_groups < 0
        ):
            raise ValueError(
                "prepare_stats.distinct_groups must be >= 0 when set"
            )

        object.__setattr__(
            self,
            "rejection_counts",
            _normalize_string_counts(
                self.rejection_counts,
                name="prepare_stats.rejection_counts",
            ),
        )
        object.__setattr__(
            self,
            "rows_per_task",
            _normalize_string_counts(
                self.rows_per_task,
                name="prepare_stats.rows_per_task",
            ),
        )
        object.__setattr__(
            self,
            "rows_per_split",
            _normalize_string_counts(
                self.rows_per_split,
                name="prepare_stats.rows_per_split",
            ),
        )

        if self.rejection_counts and (
            sum(self.rejection_counts.values()) != self.rejected_rows
        ):
            raise ValueError(
                "prepare_stats.rejection_counts must sum to rejected_rows"
            )
        if self.rows_per_task and sum(self.rows_per_task.values()) != self.output_rows:
            raise ValueError(
                "prepare_stats.rows_per_task must sum to output_rows"
            )
        if self.rows_per_split and sum(self.rows_per_split.values()) != self.output_rows:
            raise ValueError(
                "prepare_stats.rows_per_split must sum to output_rows"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PrepareResult:
    plan: PreparePlan
    artifact: PreparedArtifactRef
    stats: PrepareStats

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PrepareSpec(ABC):
    """Application-provided implementation of one prepare family.

    The core package owns the contract, validation, registry, and planning.
    Concrete subclasses own source-specific execution and may use any external
    execution or storage technology.
    """

    name: ClassVar[str]
    task_names: ClassVar[tuple[str, ...]]

    @classmethod
    def validate_definition(cls) -> None:
        _require_non_empty(getattr(cls, "name", ""), "prepare spec name")
        task_names = tuple(getattr(cls, "task_names", ()))
        if not task_names or any(
            not isinstance(task_name, str) or not task_name.strip()
            for task_name in task_names
        ):
            raise ValueError(
                f"prepare spec {cls.__name__}.task_names must contain "
                "non-empty task names"
            )
        if len(set(task_names)) != len(task_names):
            raise ValueError(
                f"prepare spec {cls.__name__}.task_names must not contain "
                "duplicates"
            )

    def validate_config(
        self,
        config: Mapping[str, Any],
        *,
        sources: tuple[SourceRef, ...],
        task_names: tuple[str, ...],
    ) -> None:
        """Optional family-specific semantic preflight before execution."""

    def build_plan(
        self,
        *,
        suite: DataSuiteSpec,
        family: FamilySpec,
        request: FamilyPrepareSpec,
        run_id: str | None,
    ) -> PreparePlan:
        self.validate_definition()
        if self.name != family.name:
            raise ValueError(
                f"prepare spec name {self.name!r} does not match family "
                f"{family.name!r}"
            )

        declared = tuple(self.task_names)
        if set(declared) != set(family.task_names):
            raise ValueError(
                f"prepare spec {self.name!r} task_names do not match "
                "DataSuiteSpec family declaration"
            )

        requested_tasks = request.task_names or family.task_names
        unknown = sorted(set(requested_tasks) - set(family.task_names))
        if unknown:
            raise ValueError(
                f"prepare family {family.name!r} selected unknown tasks: {unknown}"
            )
        selected_set = set(requested_tasks)
        selected = tuple(
            task_name
            for task_name in family.task_names
            if task_name in selected_set
        )

        sources_by_name = {source.name: source for source in suite.sources}
        sources = tuple(sources_by_name[name] for name in family.source_names)
        self.validate_config(
            request.config,
            sources=sources,
            task_names=selected,
        )
        return PreparePlan(
            family=family.name,
            source_refs=sources,
            task_names=selected,
            config=request.config,
            suite_fingerprint=suite.fingerprint,
            run_id=run_id,
        )

    @abstractmethod
    def prepare(self, plan: PreparePlan) -> PrepareResult:
        """Materialize one family and return its validated result."""
        raise NotImplementedError


class PrepareRegistry:
    """Explicit family-name registry with no import-time side effects."""

    def __init__(self) -> None:
        self._specs: dict[str, type[PrepareSpec]] = {}

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._specs))

    def register(self, spec_cls: type[PrepareSpec]) -> type[PrepareSpec]:
        if not isinstance(spec_cls, type) or not issubclass(spec_cls, PrepareSpec):
            raise TypeError("register expects a PrepareSpec subclass")
        if inspect.isabstract(spec_cls):
            raise TypeError("register expects a concrete PrepareSpec subclass")
        spec_cls.validate_definition()
        existing = self._specs.get(spec_cls.name)
        if existing is not None and existing is not spec_cls:
            raise ValueError(
                f"prepare family {spec_cls.name!r} is already registered "
                f"by {existing.__name__}"
            )
        self._specs[spec_cls.name] = spec_cls
        return spec_cls

    def get(self, name: str) -> PrepareSpec:
        _require_non_empty(name, "prepare family name")
        spec_cls = self._specs.get(name)
        if spec_cls is None:
            available = ", ".join(self.names) or "<none>"
            raise KeyError(
                f"Unknown prepare family {name!r}. Registered families: {available}"
            )
        return spec_cls()


def plan_prepare_suite(
    suite: DataSuiteSpec,
    prepare_suite: PrepareSuiteSpec,
    registry: PrepareRegistry,
) -> tuple[PreparePlan, ...]:
    """Resolve enabled families into deterministic family-level plans."""

    family_by_name = {family.name: family for family in suite.families}
    unknown = sorted(set(prepare_suite.families) - set(family_by_name))
    if unknown:
        raise ValueError(
            f"prepare suite enables families absent from DataSuiteSpec: {unknown}"
        )

    plans: list[PreparePlan] = []
    for family in suite.families:
        request = prepare_suite.families.get(family.name)
        if request is None:
            continue
        spec = registry.get(family.name)
        plans.append(
            spec.build_plan(
                suite=suite,
                family=family,
                request=request,
                run_id=prepare_suite.run_id,
            )
        )
    return tuple(plans)


def validate_prepared_mappings(
    rows: Iterable[Mapping[str, Any]],
    plan: PreparePlan,
    *,
    allowed_splits: Iterable[str] | None = None,
) -> tuple[tuple[PreparedExample, ...], PrepareStats]:
    """Validate canonical row fields, parse them, then validate semantics."""

    modern_fields = frozenset(PREPARED_SCHEMA_FIELDS)
    legacy_fields = frozenset(
        (set(PREPARED_SCHEMA_FIELDS) - {"group_id"}) | {"example_id"}
    )
    examples: list[PreparedExample] = []

    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("prepared rows must be mappings")
        fields = frozenset(row)
        if fields not in (modern_fields, legacy_fields):
            missing = sorted(modern_fields - fields)
            extra = sorted(fields - modern_fields - {"example_id"})
            raise ValueError(
                "prepared row does not match the canonical semantic schema: "
                f"missing={missing}, extra={extra}"
            )
        examples.append(PreparedExample.from_mapping(row))

    parsed = tuple(examples)
    stats = validate_prepared_examples(
        parsed,
        plan,
        allowed_splits=allowed_splits,
    )
    return parsed, stats


def validate_prepared_examples(
    examples: Iterable[PreparedExample],
    plan: PreparePlan,
    *,
    allowed_splits: Iterable[str] | None = None,
) -> PrepareStats:
    """Validate one materialized semantic corpus against its prepare plan."""

    rows = tuple(examples)
    allowed_tasks = set(plan.task_names)
    allowed_split_set = set(allowed_splits) if allowed_splits is not None else None
    seen: set[tuple[str, str, str]] = set()

    rows_per_task: Counter[str] = Counter()
    rows_per_split: Counter[str] = Counter()
    groups: set[str] = set()

    for row in rows:
        if not isinstance(row, PreparedExample):
            raise TypeError(
                "validate_prepared_examples expects PreparedExample rows"
            )
        if row.task_name not in allowed_tasks:
            raise ValueError(
                f"prepared row task {row.task_name!r} is not declared by "
                f"prepare plan {plan.family!r}"
            )
        if row.split == TRAIN_PROBE_SPLIT:
            raise ValueError(
                "prepare output must not contain train_probe; probe rows are "
                "derived by the blend stage"
            )
        if allowed_split_set is not None and row.split not in allowed_split_set:
            raise ValueError(
                f"prepared row split {row.split!r} is not allowed by this run"
            )

        key = (row.task_name, row.split, row.sample_id)
        if key in seen:
            raise ValueError(
                "prepare output contains an exact duplicate semantic row: "
                f"task={row.task_name!r} split={row.split!r} "
                f"sample_id={row.sample_id}"
            )
        seen.add(key)
        rows_per_task[row.task_name] += 1
        rows_per_split[row.split] += 1
        groups.add(row.group_id)

    return PrepareStats(
        output_rows=len(rows),
        rows_per_task=dict(rows_per_task),
        rows_per_split=dict(rows_per_split),
        distinct_groups=len(groups),
    )


def validate_prepared_artifact(
    artifact: PreparedArtifactRef,
    plan: PreparePlan,
    *,
    stats: PrepareStats | None = None,
    schema_version: str = PREPARED_SCHEMA_VERSION,
) -> None:
    """Validate artifact identity, schema version, lineage, and row accounting."""

    if artifact.family != plan.family:
        raise ValueError(
            f"prepared artifact family {artifact.family!r} does not match "
            f"prepare plan family {plan.family!r}"
        )
    if artifact.schema_version != schema_version:
        raise ValueError(
            f"prepared artifact schema {artifact.schema_version!r} does not "
            f"match expected {schema_version!r}"
        )
    if artifact.prepare_plan_fingerprint != plan.fingerprint:
        raise ValueError(
            "prepared artifact prepare_plan_fingerprint does not match "
            "the resolved prepare plan"
        )

    expected_sources = tuple(source.name for source in plan.source_refs)
    if tuple(artifact.source_names) != expected_sources:
        raise ValueError(
            f"prepared artifact source lineage mismatch: expected "
            f"{list(expected_sources)}, got {list(artifact.source_names)}"
        )

    if stats is not None and artifact.row_count != stats.output_rows:
        raise ValueError(
            f"prepared artifact row_count={artifact.row_count} does not match "
            f"validated output_rows={stats.output_rows}"
        )


def make_prepare_result(
    plan: PreparePlan,
    artifact: PreparedArtifactRef,
    *,
    stats: PrepareStats,
) -> PrepareResult:
    """Validate and bind a family plan, artifact, and accounting record."""

    validate_prepared_artifact(artifact, plan, stats=stats)
    return PrepareResult(plan=plan, artifact=artifact, stats=stats)


def execute_prepare_plan(
    plan: PreparePlan,
    registry: PrepareRegistry,
) -> PrepareResult:
    """Execute one resolved prepare plan and validate the returned result.

    Workflow systems can fan this function out concurrently without becoming a
    dependency of the core package.
    """

    spec = registry.get(plan.family)
    result = spec.prepare(plan)
    if not isinstance(result, PrepareResult):
        raise TypeError("PrepareSpec.prepare() must return PrepareResult")
    if result.plan != plan:
        raise ValueError("prepare result plan does not match the requested plan")
    validate_prepared_artifact(result.artifact, plan, stats=result.stats)
    return result
