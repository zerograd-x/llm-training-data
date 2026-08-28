from __future__ import annotations


def build_example_ids(
    prompt_ids: list[int],
    completion_ids: list[int],
    max_seq_length: int,
    eos_id: int | None,
) -> tuple[list[int], list[int]]:
    """Build one causal-LM training example with completion-only supervision.

    Completion tokens have priority. Completion is right-truncated; prompt is
    left-truncated to consume any remaining sequence budget.
    """
    if max_seq_length <= 0:
        raise ValueError("max_seq_length must be > 0")

    completion = list(completion_ids)
    if eos_id is not None:
        completion.append(eos_id)
    completion = completion[:max_seq_length]

    keep_prompt = max_seq_length - len(completion)
    kept_prompt = list(prompt_ids[-keep_prompt:]) if keep_prompt > 0 else []

    input_ids = kept_prompt + completion
    labels = [-100] * len(kept_prompt) + completion.copy()
    return input_ids, labels


def round_up(value: int, multiple: int) -> int:
    if multiple <= 1:
        return value
    return -(-value // multiple) * multiple
