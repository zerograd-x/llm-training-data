from __future__ import annotations

from sft_tokenization.packing import GreedySequencePacker


def ex(n: int, token: int):
    return ([token] * n, [token] * n)


def test_shortest_first_packing_and_padding_segment():
    packer = GreedySequencePacker(total_budget=12)
    layout = packer.pack([ex(5, 5), ex(3, 3), ex(4, 4)], pad_id=0)
    assert layout.input_ids == [3] * 3 + [4] * 4 + [5] * 5
    assert layout.position_ids == list(range(3)) + list(range(4)) + list(range(5))
    assert layout.segment_lengths == (3, 4, 5)
    assert layout.padding_length == 0
    assert layout.cu_seqlens == (0, 3, 7, 12)


def test_unplaced_examples_carry_over():
    packer = GreedySequencePacker(total_budget=7)
    first = packer.pack([ex(3, 3), ex(4, 4), ex(6, 6)], pad_id=0)
    assert first.packed_examples == 2
    assert [len(x[0]) for x in packer.carryover] == [6]

    second = packer.pack([], pad_id=0)
    assert second.input_ids[:6] == [6] * 6
    assert second.padding_length == 1
    assert second.position_ids == list(range(6)) + [0]


def test_clear_returns_pending_examples():
    packer = GreedySequencePacker(total_budget=4)
    packer.pack([ex(4, 1), ex(4, 2)], pad_id=0)
    pending = packer.clear()
    assert len(pending) == 1
    assert packer.carryover == ()
