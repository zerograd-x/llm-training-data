from __future__ import annotations

from typing import Any, Mapping, Sequence

import pytest


class FakeTokenizerImpl:
    pad_token_id = 0
    eos_token_id = 2
    chat_template = "fake"

    def __init__(self, *, supports_system: bool = True) -> None:
        self.supports_system = supports_system
        self.calls: list[tuple[list[str], bool]] = []
        self.chat_calls: list[tuple[list[Mapping[str, str]], dict[str, Any]]] = []

    def __call__(self, texts: Sequence[str], *, add_special_tokens: bool):
        texts = list(texts)
        self.calls.append((texts, add_special_tokens))
        encoded = []
        for text in texts:
            ids = [10 + sum(ord(ch) for ch in word) % 80 for word in text.split()]
            if add_special_tokens:
                ids = [1] + ids
            encoded.append(ids)
        return {"input_ids": encoded}

    def apply_chat_template(
        self,
        conversation,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        **kwargs,
    ) -> str:
        messages = list(conversation)
        self.chat_calls.append((messages, kwargs))
        if messages and messages[0]["role"] == "system" and not self.supports_system:
            raise ValueError("system role unsupported")
        body = "|".join(f"<{m['role']}>{m['content']}" for m in messages)
        if add_generation_prompt:
            body += "|<assistant>"
        return body


@pytest.fixture
def tokenizer_cls():
    return FakeTokenizerImpl
