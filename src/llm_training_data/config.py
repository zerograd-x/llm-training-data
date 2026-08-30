from __future__ import annotations

from pydantic import BaseModel

_DEFAULT_PROMPT_RENDERER = "llm_training_data.renderers.DefaultPromptRenderer"
_DEFAULT_COMPLETION_RENDERER = "llm_training_data.renderers.DefaultCompletionRenderer"


class SFTDataConfig(BaseModel):
    prompt_renderer: str = _DEFAULT_PROMPT_RENDERER
    completion_renderer: str | None = _DEFAULT_COMPLETION_RENDERER
    system_prompt: str | None = None
    enable_thinking: bool | None = None
    completion_prefix: str | None = None
