"""P2 var binding config — GPT-2 small activation patching.

Tier 1 cf pairs for circuit identification.
"""
from shared.config import DEVICE

# ---- Model (HookedTransformer) ----
MODEL_NAME = "gpt2-small"  # HookedTransformer alias
DEVICE = DEVICE  # from shared.config

# ---- Data ----
TIER1_FILE = "var_binding_tier1.jsonl"  # Just filename, DATASETS_DIR prepended in data.py
N_PAIRS_SAMPLE = 5  # First N pairs for initial cache shape check

# ---- Analysis ----
# Logit-diff metric: logit[answer] - mean(logit[distractor_answer_token_ids])
# Wang et al. 2022 IOI standard
PATCH_METRIC = "logit_diff"

# Activation patching: clean -> corrupt
# Patch at source_var_token_pos position
PATCH_COMPONENTS = [
    "attn_head",  # (layer, head)
    "mlp",        # layer
]

# ---- Output ----
RESULTS_DIR = "results/code"
