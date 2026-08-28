from __future__ import annotations

import pytest
import torch

transformers = pytest.importorskip("transformers")
from transformers import AutoModelForCausalLM, AutoTokenizer

from sft_tokenization import create_sft_collate_fn


MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_ID)


@pytest.fixture(scope="module")
def model():
    loaded = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
    )
    loaded.eval()
    return loaded


@pytest.mark.integration
def test_real_hf_model_forward(tokenizer, model):
    """Real tokenizer -> library collator -> real HF causal-LM forward/loss."""
    collate = create_sft_collate_fn(
        tokenizer,
        max_seq_length=128,
        system_prompt="You are a concise assistant.",
        pad_to_multiple_of=8,
    )

    batch = collate(
        {
            "prompt": [
                "What is 2 + 2?",
                "What is the capital of France?",
            ],
            "completion": ["4", "Paris"],
        }
    )

    assert batch["input_ids"].ndim == 2
    assert batch["attention_mask"].shape == batch["input_ids"].shape
    assert batch["labels"].shape == batch["input_ids"].shape
    assert (batch["labels"] == -100).any()
    assert (batch["labels"] != -100).any()

    with torch.no_grad():
        outputs = model(**batch)

    assert outputs.loss is not None
    assert torch.isfinite(outputs.loss)
    assert outputs.logits.shape[:2] == batch["input_ids"].shape


@pytest.mark.integration
def test_real_tokenizer_completion_boundary(tokenizer):
    """The supervised token span must equal separately tokenized completion + EOS."""
    completion = "Paris"
    collate = create_sft_collate_fn(
        tokenizer,
        max_seq_length=128,
        pad_to_multiple_of=1,
    )

    batch = collate(
        {
            "prompt": ["The capital of France is"],
            "completion": [completion],
        }
    )

    labels = batch["labels"][0]
    actual_supervised_ids = labels[labels != -100].tolist()

    expected_supervised_ids = tokenizer(
        completion,
        add_special_tokens=False,
    )["input_ids"]
    if tokenizer.eos_token_id is not None:
        expected_supervised_ids = [*expected_supervised_ids, tokenizer.eos_token_id]

    assert actual_supervised_ids == expected_supervised_ids
