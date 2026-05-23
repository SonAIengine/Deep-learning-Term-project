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
# Placeholder — update after running _day1_verify.
SINGLE_TOKEN_NAMES = [
    "a", "b", "c", "d", "e", "x", "y", "z", "p", "q", "r", "s",
]

# Answer values: single-token integers only.
ANSWER_VALUES = list(range(1, 10))   # 1..9

# ---- Paths ----
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS_DIR = os.path.join(REPO_ROOT, "datasets")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
