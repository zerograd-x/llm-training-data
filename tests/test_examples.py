from __future__ import annotations

from sft_tokenization.examples import build_example_ids, round_up


def test_build_example_ids_normal():
    ids, labels = build_example_ids([10, 11, 12], [20, 21], 8, 2)
    assert ids == [10, 11, 12, 20, 21, 2]
    assert labels == [-100, -100, -100, 20, 21, 2]


def test_build_example_ids_left_truncates_prompt():
    ids, labels = build_example_ids([10, 11, 12, 13, 14], [20, 21], 5, 2)
    assert ids == [13, 14, 20, 21, 2]
    assert labels == [-100, -100, 20, 21, 2]


def test_completion_can_truncate_away_eos():
    ids, labels = build_example_ids([10, 11], [20, 21, 22, 23, 24], 4, 2)
    assert ids == [20, 21, 22, 23]
    assert labels == ids


def test_round_up():
    assert round_up(65, 64) == 128
    assert round_up(65, 1) == 65
