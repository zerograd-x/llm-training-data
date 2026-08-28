"""Minimal usage example with a Hugging Face tokenizer.

Install optional Hugging Face support first:
    pip install -e '.[hf]'
"""
from transformers import AutoTokenizer

from sft_tokenization import create_sft_collate_fn


tokenizer = AutoTokenizer.from_pretrained("gpt2")
collate = create_sft_collate_fn(tokenizer, max_seq_length=128)

batch = {
    "prompt": ["Question: 2 + 2 ="],
    "completion": [" 4"],
}
print(collate(batch))
