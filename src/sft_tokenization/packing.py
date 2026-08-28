from __future__ import annotations

from dataclasses import dataclass
import warnings

TrainingExample = tuple[list[int], list[int]]


@dataclass(frozen=True)
class PackedSequenceLayout:
    input_ids: list[int]
    labels: list[int]
    position_ids: list[int]
    segment_lengths: tuple[int, ...]
    packed_examples: int
    padding_length: int
    dropped_examples: int = 0

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


class GreedySequencePacker:
    """Stateful shortest-first single-bin sequence packer.

    Unplaced examples carry over to later calls. The carry-over pool is capped
    by token count to prevent unbounded growth.
    """

    def __init__(self, total_budget: int, *, carryover_budget_multiplier: int = 4) -> None:
        if total_budget <= 0:
            raise ValueError("total_budget must be > 0")
        if carryover_budget_multiplier <= 0:
            raise ValueError("carryover_budget_multiplier must be > 0")
        self.total_budget = total_budget
        self.max_carryover_tokens = total_budget * carryover_budget_multiplier
        self._carryover: list[TrainingExample] = []

    @property
    def carryover(self) -> tuple[TrainingExample, ...]:
        return tuple(self._carryover)

    @property
    def carryover_tokens(self) -> int:
        return sum(len(ids) for ids, _ in self._carryover)

    def clear(self) -> list[TrainingExample]:
        pending = self._carryover
        self._carryover = []
        return pending

    def pack(self, new_examples: list[TrainingExample], *, pad_id: int) -> PackedSequenceLayout:
        examples = self._carryover + [
            (list(ids), list(labels)) for ids, labels in new_examples
        ]
        examples.sort(key=lambda pair: len(pair[0]))

        packed_ids: list[int] = []
        packed_labels: list[int] = []
        packed_position_ids: list[int] = []
        segment_lengths: list[int] = []
        n_placed = 0

        for ids, labels in examples:
            if len(ids) != len(labels):
                raise ValueError("Each example must have equally sized input_ids and labels")
            n = len(ids)
            if n > self.total_budget:
                raise ValueError(
                    f"Example length {n} exceeds packing budget {self.total_budget}; "
                    "truncate examples before packing."
                )
            if len(packed_ids) + n > self.total_budget:
                break
            packed_ids.extend(ids)
            packed_labels.extend(labels)
            packed_position_ids.extend(range(n))
            segment_lengths.append(n)
            n_placed += 1

        self._carryover = examples[n_placed:]
        dropped_examples = self._cap_carryover()

        remaining = self.total_budget - len(packed_ids)
        if remaining > 0:
            packed_ids.extend([pad_id] * remaining)
            packed_labels.extend([-100] * remaining)
            packed_position_ids.extend(range(remaining))
            segment_lengths.append(remaining)

        return PackedSequenceLayout(
            input_ids=packed_ids,
            labels=packed_labels,
            position_ids=packed_position_ids,
            segment_lengths=tuple(segment_lengths),
            packed_examples=n_placed,
            padding_length=remaining,
            dropped_examples=dropped_examples,
        )

    def _cap_carryover(self) -> int:
        total = self.carryover_tokens
        if total <= self.max_carryover_tokens:
            return 0

        before = len(self._carryover)
        kept: list[TrainingExample] = []
        kept_tokens = 0
        for example in self._carryover:
            n = len(example[0])
            if kept_tokens + n > self.max_carryover_tokens:
                break
            kept.append(example)
            kept_tokens += n

        self._carryover = kept
        dropped = before - len(kept)
        warnings.warn(
            f"Carry-over buffer exceeded {self.max_carryover_tokens} tokens "
            f"({total} pending); dropped {dropped} longer examples from the "
            "length-sorted pending pool.",
            RuntimeWarning,
            stacklevel=3,
        )
        return dropped
