from __future__ import annotations

from pydantic import BaseModel

_DEFAULT_PROMPT_RENDERER = "sft_tokenization.transforms.DefaultPromptRenderer"
_DEFAULT_COMPLETION_RENDERER = "sft_tokenization.transforms.DefaultCompletionRenderer"


class SFTDataConfig(BaseModel):
    """Serializable configuration for SFT data preparation."""

    prompt_renderer: str = _DEFAULT_PROMPT_RENDERER
    completion_renderer: str | None = _DEFAULT_COMPLETION_RENDERER
    system_prompt: str | None = None
    enable_thinking: bool | None = None
    completion_prefix: str | None = None


class TokenizationConfig(BaseModel):
    """Legacy configuration preserved for backward compatibility."""

    prompt_transform: str = "sft_tokenization.transforms.DefaultPromptTransform"
    completion_transform: str | None = "sft_tokenization.transforms.DefaultCompletionTransform"
    system_prompt: str | None = None
    enable_thinking: bool | None = None
    completion_prefix: str | None = None
