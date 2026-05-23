"""Week 1 Day 1: verify which variable names and answer values tokenize
as a single token under the GPT-2 BPE tokenizer.

Run BEFORE generating any code-binding data. Paste the output into
`shared/config.py` (SINGLE_TOKEN_NAMES, ANSWER_VALUES).

Why single-token? Activation patching requires clean/corrupted prompts to
have aligned token positions — variable names that split into different
numbers of tokens break alignment.
"""
from transformers import GPT2Tokenizer

CANDIDATES = [
    # single letters
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
    # short words — leading-space form often becomes one BPE token
    "cat", "dog", "sun", "red", "blue", "book", "desk", "car", "tree", "star",
    "apple", "banana", "mango", "kiwi", "lemon", "grape", "peach", "plum",
    "alpha", "beta", "gamma", "delta", "sigma", "theta",
]


def main():
    tok = GPT2Tokenizer.from_pretrained("gpt2")

    single_tok_names = [n for n in CANDIDATES
                        if len(tok.encode(" " + n)) == 1]

    single_tok_values = [v for v in range(1, 20)
                         if len(tok.encode(" " + str(v))) == 1]

    print(f"# === Day 1 verification (GPT-2 tokenizer) ===")
    print(f"# Single-token variable names ({len(single_tok_names)}):")
    print(f"SINGLE_TOKEN_NAMES = {single_tok_names}")
    print()
    print(f"# Single-token answer values ({len(single_tok_values)}):")
    print(f"ANSWER_VALUES = {single_tok_values}")
    print()
    print("# → Paste both lists into shared/config.py and commit.")


if __name__ == "__main__":
    main()
