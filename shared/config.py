"""Shared configuration. Import this everywhere instead of hardcoding.

All three tracks (grokking / code / ioi) must read seeds, sizes, and pools
from this file so results stay comparable.
"""
import os
import torch

# ---- Reproducibility ----
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Modular Arithmetic (Step 1~2) ----
MODULAR_P = 113
MODULAR_TRAIN_FRAC = 0.3

# ---- Code Variable Binding (Step 3) ----
VB_TIER1_N_PAIRS = 500      # → 1,000 samples (clean + corrupt)
VB_TIER2_N = 500
VB_TIER3_N = 500

# ---- IOI (Step 3 비교) ----
IOI_N = 500

# ---- Model ----
MODEL_NAME = "gpt2"

# ============================================================
# Day 1 verification results
# ------------------------------------------------------------
# Run `python -m data._day1_verify` and paste the output below.
# These pools are the ONLY allowed variable names / answer values.
# ============================================================

# Tier 1 (strictest): single-letter names only — guarantees identical
# token length between clean / corrupted prompts.
SINGLE_LETTER_POOL = ["a", "b", "c", "x", "y", "z"]

# Tier 2: single-token names verified by GPT-2 tokenizer (Day 1 output).
# Generated 2026-05-23 by `python -m data._day1_verify` — 47 names confirmed
# single-token after a leading-space encode. Re-run the verifier and update
# this list if the tokenizer version changes.
SINGLE_TOKEN_NAMES = [
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
    "cat", "dog", "sun", "red", "blue", "book", "desk", "car", "tree",
    "star", "apple", "banana", "mango", "lemon", "grape", "peach", "plum",
    "alpha", "beta", "gamma", "delta",
]

# Answer values: single-token integers. Day 1 verifier confirms 1..19 are all
# single-token under GPT-2 BPE; we keep 1..9 per datasets.md §4.6 so the
# distribution stays balanced and the answer slot stays decimal-digit only.
ANSWER_VALUES = list(range(1, 10))   # 1..9

# ---- Paths ----
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS_DIR = os.path.join(REPO_ROOT, "datasets")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
