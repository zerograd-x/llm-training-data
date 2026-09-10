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



class CausalLMDataConfig(BaseModel):
    """Controls plain causal-LM document tokenization and sequence construction."""

    append_eos: bool = True
    add_special_tokens: bool = False
    pack_across_documents: bool = True
    allow_document_split: bool = True
    drop_remainder: bool = False
