from __future__ import annotations

from sft_tokenization.checkpoint import (
    read_checkpoint_system_prompt,
    resolve_effective_system_prompt,
    stamp_checkpoint_system_prompt,
)


def test_checkpoint_system_prompt_round_trip(tmp_path):
    stamp_checkpoint_system_prompt(tmp_path, "rules\n")
    assert read_checkpoint_system_prompt(tmp_path) == "rules\n"


def test_none_does_not_create_sidecar(tmp_path):
    stamp_checkpoint_system_prompt(tmp_path, None)
    assert read_checkpoint_system_prompt(tmp_path) is None


def test_explicit_system_prompt_wins(tmp_path):
    stamp_checkpoint_system_prompt(tmp_path, "old")
    assert resolve_effective_system_prompt("new", tmp_path) == "new"


def test_checkpoint_is_default_when_config_is_none(tmp_path):
    stamp_checkpoint_system_prompt(tmp_path, "saved")
    assert resolve_effective_system_prompt(None, tmp_path) == "saved"
