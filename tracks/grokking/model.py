"""Build the HookedTransformer for grokking."""
from transformer_lens import HookedTransformer, HookedTransformerConfig

from shared.config import DEVICE, SEED
from tracks.grokking import config as C


def build_model():
    cfg = HookedTransformerConfig(
        n_layers=C.N_LAYERS,
        d_model=C.D_MODEL,
        n_heads=C.N_HEADS,
        d_head=C.D_HEAD,
        d_mlp=C.D_MLP,
        n_ctx=C.N_CTX,
        d_vocab=C.D_VOCAB,
        act_fn=C.ACT_FN,
        attn_only=False,
        normalization_type=C.NORMALIZATION_TYPE,
        positional_embedding_type=C.POS_EMB_TYPE,
        seed=SEED,
        device=DEVICE,
        init_weights=True,
    )
    return HookedTransformer(cfg)
