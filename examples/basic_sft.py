"""Minimal usage example with a Hugging Face tokenizer.

Install optional Hugging Face support first:
    pip install -e '.[hf]'
"""
from transformers import AutoTokenizer

from llm_training_data import build_sft_collator


tokenizer = AutoTokenizer.from_pretrained("gpt2")
collate = build_sft_collator(tokenizer, max_sequence_length=128)

batch = {
    "prompt": ["Question: 2 + 2 ="],
    "completion": [" 4"],
}
print(collate(batch))
