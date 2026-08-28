from __future__ import annotations

from pydantic import BaseModel

_DEFAULT_PROMPT_TRANSFORM = "sft_tokenization.transforms.DefaultPromptTransform"
_DEFAULT_COMPLETION_TRANSFORM = "sft_tokenization.transforms.DefaultCompletionTransform"


class TokenizationConfig(BaseModel):
    """Serializable tokenization configuration.

    Transform fields are dotted import paths and are intentionally resolved at
    runtime rather than when this model is constructed.
    """

    prompt_transform: str = _DEFAULT_PROMPT_TRANSFORM
    completion_transform: str | None = _DEFAULT_COMPLETION_TRANSFORM
    system_prompt: str | None = None
    enable_thinking: bool | None = None
    completion_prefix: str | None = None
