from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)
SYSTEM_PROMPT_FILENAME = "system_prompt.txt"


def write_system_prompt_metadata(checkpoint_dir: str | Path, system_prompt: str | None) -> None:
    if system_prompt is None:
        return
    (Path(checkpoint_dir) / SYSTEM_PROMPT_FILENAME).write_text(system_prompt)


def read_system_prompt_metadata(checkpoint_dir: str | Path) -> str | None:
    path = Path(checkpoint_dir) / SYSTEM_PROMPT_FILENAME
    return path.read_text() if path.exists() else None


def resolve_system_prompt(
    configured_system_prompt: str | None,
    checkpoint_dir: str | Path | None = None,
) -> str | None:
    recorded = read_system_prompt_metadata(checkpoint_dir) if checkpoint_dir is not None else None
    if configured_system_prompt is None:
        return recorded
    if recorded is not None and recorded != configured_system_prompt:
        logger.warning("Configured system prompt differs from checkpoint metadata.")
    return configured_system_prompt
