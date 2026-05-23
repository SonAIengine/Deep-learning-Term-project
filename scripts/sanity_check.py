"""Confirm TransformerLens loads GPT-2 small and we can inspect attention."""
import torch
from transformer_lens import HookedTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device = {device}")

model = HookedTransformer.from_pretrained("gpt2", device=device)
model.eval()

prompt = "When Mary and John went to the store, John gave a drink to"
tokens = model.to_tokens(prompt)
print(f"tokens shape = {tokens.shape}")

with torch.no_grad():
    logits, cache = model.run_with_cache(tokens)

print(f"logits shape = {logits.shape}")

attn = cache["pattern", 0]
print(f"layer-0 attention pattern shape = {tuple(attn.shape)}  (batch, head, q, k)")

top_token_id = logits[0, -1].argmax().item()
print(f"top next-token prediction: {model.to_string([top_token_id])!r}")
print("OK")
