from __future__ import annotations

import pytest

from sft_tokenization.chat import ChatTemplateWrapper


def test_raw_mode_does_not_use_chat_template(tokenizer_cls):
    tokenizer = tokenizer_cls()
    wrapper = ChatTemplateWrapper(tokenizer, None)
    assert wrapper("hello") == "hello"
    assert tokenizer.chat_calls == []


def test_system_role_is_used_when_supported(tokenizer_cls):
    tokenizer = tokenizer_cls(supports_system=True)
    wrapper = ChatTemplateWrapper(tokenizer, "rules", enable_thinking=False)
    rendered = wrapper("question")
    assert "<system>rules" in rendered
    assert "<user>question" in rendered
    assert tokenizer.chat_calls[-1][1] == {"enable_thinking": False}


def test_system_prompt_falls_back_into_user_message(tokenizer_cls):
    tokenizer = tokenizer_cls(supports_system=False)
    wrapper = ChatTemplateWrapper(tokenizer, "rules")
    rendered = wrapper("question")
    assert "<system>" not in rendered
    assert "<user>rules\n\nquestion" in rendered


def test_nonempty_system_prompt_requires_chat_template(tokenizer_cls):
    tokenizer = tokenizer_cls()
    tokenizer.chat_template = None
    with pytest.raises(ValueError):
        ChatTemplateWrapper(tokenizer, "rules")
