from __future__ import annotations


def build_training_example(
    prompt_token_ids: list[int],
    response_token_ids: list[int],
    max_sequence_length: int,
    eos_token_id: int | None,
) -> tuple[list[int], list[int]]:
    """Build one completion-supervised causal-LM example."""
    if max_sequence_length <= 0:
        raise ValueError("max_sequence_length must be > 0")

    response = list(response_token_ids)
    if eos_token_id is not None:
        response.append(eos_token_id)
    response = response[:max_sequence_length]

    prompt_budget = max_sequence_length - len(response)
    prompt = list(prompt_token_ids[-prompt_budget:]) if prompt_budget > 0 else []
    input_ids = prompt + response
    labels = [-100] * len(prompt) + response.copy()
    return input_ids, labels


def round_up(value: int, multiple: int) -> int:
    if multiple <= 1:
        return value
    return -(-value // multiple) * multiple
