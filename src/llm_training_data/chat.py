from __future__ import annotations

from .types import TokenizerLike


class ChatPromptFormatter:
    """Apply one tokenizer chat-template policy consistently to prompts."""

    def __init__(
        self,
        tokenizer: TokenizerLike,
        system_prompt: str | None,
        enable_thinking: bool | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.enable_thinking = enable_thinking
        self.supports_system_role = False
        if self.system_prompt:
            if not getattr(tokenizer, "chat_template", None):
                raise ValueError("A non-empty system_prompt requires a tokenizer with a chat_template.")
            self.supports_system_role = self._probe_system_role()

    def _template_kwargs(self) -> dict[str, bool]:
        return {} if self.enable_thinking is None else {"enable_thinking": self.enable_thinking}

    def _probe_system_role(self) -> bool:
        messages = [{"role": "system", "content": "_"}, {"role": "user", "content": "_"}]
        try:
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **self._template_kwargs(),
            )
            return True
        except Exception:
            return False

    def __call__(self, prompt: str) -> str:
        if not self.system_prompt:
            return prompt
        if self.supports_system_role:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]
        else:
            messages = [{"role": "user", "content": f"{self.system_prompt}\n\n{prompt}"}]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **self._template_kwargs(),
        )
