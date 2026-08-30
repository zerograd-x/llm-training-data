from __future__ import annotations

import pytest
import torch

transformers = pytest.importorskip("transformers")
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_training_data import build_packed_sft_collator


MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"


def _segment_starts(position_ids: torch.Tensor) -> list[int]:
    return torch.nonzero(position_ids[0] == 0, as_tuple=False).flatten().tolist()


def _find_subsequence(haystack: list[int], needle: list[int]) -> int:
    matches = [
        start
        for start in range(len(haystack) - len(needle) + 1)
        if haystack[start : start + len(needle)] == needle
    ]
    if len(matches) != 1:
        raise AssertionError(f"Expected exactly one packed segment match, found {len(matches)}")
    return matches[0]


@pytest.mark.integration
@pytest.mark.gpu
def test_packed_segments_are_attention_isolated():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for FlashAttention-2 integration testing")
    pytest.importorskip("flash_attn")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
    ).cuda()
    model.eval()

    def make_collator():
        return build_packed_sft_collator(
            tokenizer,
            max_sequence_length=64,
            packing_factor=2,
        )

    batch_ab = make_collator()(
        {
            "prompt": [
                "Answer with one word: sky color?",
                "Answer with one word: grass color?",
            ],
            "completion": ["blue", "green"],
        }
    )
    batch_b = make_collator()(
        {
            "prompt": ["Answer with one word: grass color?"],
            "completion": ["green"],
        }
    )

    standalone_starts = _segment_starts(batch_b["position_ids"])
    assert standalone_starts[0] == 0
    assert len(standalone_starts) >= 2
    b_length = standalone_starts[1]
    b_ids = batch_b["input_ids"][0, :b_length].tolist()

    ab_ids = batch_ab["input_ids"][0].tolist()
    b_start = _find_subsequence(ab_ids, b_ids)
    assert batch_ab["position_ids"][0, b_start].item() == 0

    batch_ab_cuda = {key: value.cuda() for key, value in batch_ab.items()}
    batch_b_cuda = {key: value.cuda() for key, value in batch_b.items()}

    with torch.no_grad():
        logits_ab = model(
            input_ids=batch_ab_cuda["input_ids"],
            position_ids=batch_ab_cuda["position_ids"],
            use_cache=False,
        ).logits
        logits_b = model(
            input_ids=batch_b_cuda["input_ids"],
            position_ids=batch_b_cuda["position_ids"],
            use_cache=False,
        ).logits

    torch.testing.assert_close(
        logits_ab[0, b_start : b_start + b_length],
        logits_b[0, :b_length],
        rtol=1e-2,
        atol=1e-2,
    )
