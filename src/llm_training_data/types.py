from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

ColumnBatch = Mapping[str, Sequence[Any]]


@runtime_checkable
class TokenizerLike(Protocol):
    """Minimal tokenizer interface required by the data-preparation API."""

    pad_token_id: int | None
    eos_token_id: int | None
    chat_template: str | None

    def __call__(self, texts: Sequence[str], *, add_special_tokens: bool) -> Mapping[str, Any]: ...

    def apply_chat_template(
        self,
        conversation: Sequence[Mapping[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        **kwargs: Any,
    ) -> str: ...
