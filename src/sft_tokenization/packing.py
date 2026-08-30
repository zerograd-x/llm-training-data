from __future__ import annotations

from dataclasses import dataclass
import warnings

TrainingExample = tuple[list[int], list[int]]


@dataclass(frozen=True)
class PackedSequence:
    input_ids: list[int]
    labels: list[int]
    position_ids: list[int]
    segment_lengths: tuple[int, ...]
    packed_example_count: int
    padding_length: int
    dropped_example_count: int = 0

    @property
    def total_length(self) -> int:
        return len(self.input_ids)

    @property
    def cu_seqlens(self) -> tuple[int, ...]:
        """Cumulative segment boundaries, including the final total length."""
        boundaries = [0]
        total = 0
        for length in self.segment_lengths:
            total += length
            boundaries.append(total)
        return tuple(boundaries)

    # Legacy attribute names.
    @property
    def packed_examples(self) -> int:
        return self.packed_example_count

    @property
    def dropped_examples(self) -> int:
        return self.dropped_example_count


class ShortestFirstSequencePacker:
    """Stateful shortest-first single-bin sequence packer.

    Unplaced examples remain pending for later calls. The pending pool is capped
    by token count to prevent unbounded growth.
    """

    def __init__(self, token_budget: int, *, pending_budget_multiplier: int = 4) -> None:
        if token_budget <= 0:
            raise ValueError("token_budget must be > 0")
        if pending_budget_multiplier <= 0:
            raise ValueError("pending_budget_multiplier must be > 0")
        self.token_budget = token_budget
        self.max_pending_tokens = token_budget * pending_budget_multiplier
        self._pending: list[TrainingExample] = []

    @property
    def pending(self) -> tuple[TrainingExample, ...]:
        return tuple(self._pending)

    @property
    def pending_tokens(self) -> int:
        return sum(len(ids) for ids, _ in self._pending)

    def clear(self) -> list[TrainingExample]:
        pending = self._pending
        self._pending = []
        return pending

    def pack(self, new_examples: list[TrainingExample], *, pad_id: int) -> PackedSequence:
        examples = self._pending + [(list(ids), list(labels)) for ids, labels in new_examples]
        examples.sort(key=lambda pair: len(pair[0]))

        packed_ids: list[int] = []
        packed_labels: list[int] = []
        packed_position_ids: list[int] = []
        segment_lengths: list[int] = []
        placed_count = 0

        for ids, labels in examples:
            if len(ids) != len(labels):
                raise ValueError("Each example must have equally sized input_ids and labels")
            length = len(ids)
            if length > self.token_budget:
                raise ValueError(
                    f"Example length {length} exceeds packing budget {self.token_budget}; "
                    "truncate examples before packing."
                )
            if len(packed_ids) + length > self.token_budget:
                break
            packed_ids.extend(ids)
            packed_labels.extend(labels)
            packed_position_ids.extend(range(length))
            segment_lengths.append(length)
            placed_count += 1

        self._pending = examples[placed_count:]
        dropped_count = self._cap_pending()

        padding_length = self.token_budget - len(packed_ids)
        if padding_length > 0:
            packed_ids.extend([pad_id] * padding_length)
            packed_labels.extend([-100] * padding_length)
            packed_position_ids.extend(range(padding_length))
            segment_lengths.append(padding_length)

        return PackedSequence(
            input_ids=packed_ids,
            labels=packed_labels,
            position_ids=packed_position_ids,
            segment_lengths=tuple(segment_lengths),
            packed_example_count=placed_count,
            padding_length=padding_length,
            dropped_example_count=dropped_count,
        )

    def _cap_pending(self) -> int:
        total = self.pending_tokens
        if total <= self.max_pending_tokens:
            return 0

        before = len(self._pending)
        kept: list[TrainingExample] = []
        kept_tokens = 0
        for example in self._pending:
            length = len(example[0])
            if kept_tokens + length > self.max_pending_tokens:
                break
            kept.append(example)
            kept_tokens += length

        self._pending = kept
        dropped = before - len(kept)
        warnings.warn(
            f"Pending buffer exceeded {self.max_pending_tokens} tokens "
            f"({total} pending); dropped {dropped} longer examples from the "
            "length-sorted pending pool.",
            RuntimeWarning,
            stacklevel=3,
        )
        return dropped

    # Legacy attribute names.
    @property
    def total_budget(self) -> int:
        return self.token_budget

    @property
    def max_carryover_tokens(self) -> int:
        return self.max_pending_tokens

    @property
    def carryover(self) -> tuple[TrainingExample, ...]:
        return self.pending

    @property
    def carryover_tokens(self) -> int:
        return self.pending_tokens


class GreedySequencePacker(ShortestFirstSequencePacker):
    """Backward-compatible constructor using legacy parameter names."""

    def __init__(self, total_budget: int, *, carryover_budget_multiplier: int = 4) -> None:
        super().__init__(
            token_budget=total_budget,
            pending_budget_multiplier=carryover_budget_multiplier,
        )


PackedSequenceLayout = PackedSequence
