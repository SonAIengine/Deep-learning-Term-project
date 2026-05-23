"""P1 grokking config — canonical Nanda et al. (2023) reproduction.

Source: mechanistic-interpretability-grokking/progress-measures-paper,
transformers.py (Config dataclass, lines 28-66). All values pinned to that file.
"""
from shared.config import MODULAR_P

# ---- Model (HookedTransformer) ----
N_LAYERS = 1                # transformers.py:46
D_MODEL = 128               # transformers.py:32
N_HEADS = 4                 # transformers.py:51
D_HEAD = D_MODEL // N_HEADS # 32
D_MLP = 4 * D_MODEL         # 512, transformers.py:50
N_CTX = 3                   # tokens: a, b, '='
D_VOCAB = MODULAR_P + 1     # 114
ACT_FN = "relu"             # transformers.py:53
NORMALIZATION_TYPE = None   # Nanda commented out all LN call sites
POS_EMB_TYPE = "standard"   # learned, transformers.py:181-183

# ---- Train ----
LR = 1e-3                   # transformers.py:29
WEIGHT_DECAY = 1.0          # transformers.py:30, critical for grokking
BETAS = (0.9, 0.98)         # transformers.py:470
WARMUP_STEPS = 10           # transformers.py:471
NUM_EPOCHS = 50_000         # transformers.py:35

# ---- Eval ----
LOG_EVERY = 100             # ~500 log points over 50K
CKPT_EVERY = 1000           # transformers.py:37 (save_every=100; we use 1000 to limit disk)
