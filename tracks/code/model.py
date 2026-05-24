"""Load GPT-2 small for circuit analysis."""
from transformer_lens import HookedTransformer

from tracks.code.config import MODEL_NAME, DEVICE


def load_gpt2_small() -> HookedTransformer:
    """Load GPT-2 small (HookedTransformer wrapper)."""
    model = HookedTransformer.from_pretrained(MODEL_NAME)
    return model.to(DEVICE)
