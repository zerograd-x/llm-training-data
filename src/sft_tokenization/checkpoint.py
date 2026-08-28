from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_FILENAME = "system_prompt.txt"


def stamp_checkpoint_system_prompt(checkpoint_dir: str | Path, system_prompt: str | None) -> None:
    """Write the exact training system prompt to a checkpoint sidecar.

    ``None`` means no sidecar is written. An empty string intentionally creates
    an empty file so callers can distinguish "not recorded" from "recorded as
    empty".
    """
    if system_prompt is None:
        return
    path = Path(checkpoint_dir) / SYSTEM_PROMPT_FILENAME
    path.write_text(system_prompt)


def read_checkpoint_system_prompt(checkpoint_dir: str | Path) -> str | None:
    path = Path(checkpoint_dir) / SYSTEM_PROMPT_FILENAME
    if not path.exists():
        return None
    return path.read_text()


def resolve_effective_system_prompt(
    configured_system_prompt: str | None,
    checkpoint_dir: str | Path | None = None,
) -> str | None:
    """Resolve explicit configuration over checkpoint metadata."""
    checkpoint_prompt = (
        read_checkpoint_system_prompt(checkpoint_dir) if checkpoint_dir is not None else None
    )

    if configured_system_prompt is None:
        if checkpoint_prompt is not None:
            logger.info("Using system prompt recorded in checkpoint metadata.")
            return checkpoint_prompt
        logger.info("No system prompt configured or recorded in checkpoint metadata.")
        return None

    if checkpoint_dir is None or checkpoint_prompt is None:
        logger.warning("Using explicit system prompt; checkpoint metadata is unavailable for comparison.")
    elif checkpoint_prompt == configured_system_prompt:
        logger.info("Explicit system prompt matches checkpoint metadata.")
    else:
        logger.warning(
            "Explicit system prompt differs from checkpoint metadata; tokenization may be out of distribution."
        )
    return configured_system_prompt
