"""IOI dataset loader (Step 3 비교 baseline).

Uses the official Easy-Transformer IOIDataset so our results stay
directly comparable to Wang et al. (2022)'s 26-head reference circuit.

Install:
    pip install git+https://github.com/redwoodresearch/Easy-Transformer.git
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.config import IOI_N, MODEL_NAME


def load_ioi(n: int = IOI_N, prompt_type: str = "mixed"):
    """Returns an IOIDataset instance.

    prompt_type options: "mixed" (ABBA + BABA), "ABBA", "BABA".
    """
    from easy_transformer.ioi_dataset import IOIDataset
    from transformers import GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    return IOIDataset(prompt_type=prompt_type, N=n, tokenizer=tok)


if __name__ == "__main__":
    ds = load_ioi()
    print(f"Loaded IOI dataset: N={len(ds.sentences)}")
    print(f"Example prompt: {ds.sentences[0]}")
