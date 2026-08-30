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


class TokenizationConfig(SFTDataConfig):
    """Backward-compatible config using the legacy transform field names."""

    prompt_transform: str = _DEFAULT_PROMPT_RENDERER
    completion_transform: str | None = _DEFAULT_COMPLETION_RENDERER

    def model_post_init(self, __context: object) -> None:
        self.prompt_renderer = self.prompt_transform
        self.completion_renderer = self.completion_transform
