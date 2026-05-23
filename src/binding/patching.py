"""Step 3-2: per-head activation patching on GPT-2 small with Tier 1 cf pairs.

Patches each attention head's `z` output (post-attention, pre-W_O), corrupt → clean,
and measures logit_diff recovery. High recovery = that head carries binding info.

Metric per (layer, head):
    recovery = (patched_ld - corrupt_ld) / max(clean_ld - corrupt_ld, eps)
averaged over cf pairs. 1.0 = full restoration. 0.0 = no effect.

Output:
  - results/binding_patching/head_effects.npy   shape [n_pairs, n_layers, n_heads]
  - results/binding_patching/head_mean.png      heatmap
  - results/binding_patching/top_heads.json     ranked list
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from shared.config import DATASETS_DIR, RESULTS_DIR


def load_pairs(tier_path):
    by_cf = defaultdict(dict)
    with open(tier_path) as f:
        for line in f:
            r = json.loads(line)
            by_cf[r["cf_id"]][r["role"]] = r
    return [{"cf_id": cf, "clean": d["clean"], "corrupt": d["corrupt"]}
            for cf, d in by_cf.items() if "clean" in d and "corrupt" in d]


def logit_diff(logits_last, answer_id, distractor_ids):
    ans = logits_last[answer_id]
    dist = logits_last[torch.tensor(distractor_ids, device=logits_last.device)].mean()
    return (ans - dist).item()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default=os.path.join(DATASETS_DIR, "var_binding_tier1.jsonl"))
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--out-dir", default=os.path.join(RESULTS_DIR, "binding_patching"))
    args = ap.parse_args()

    from transformer_lens import HookedTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[load] GPT-2 small on {device}")
    model = HookedTransformer.from_pretrained("gpt2", device=device)
    model.eval()

    pairs = load_pairs(args.tier)[: args.n]
    n_pairs = len(pairs)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    print(f"[data] {n_pairs} pairs, model {n_layers}L × {n_heads}H")

    effects = np.full((n_pairs, n_layers, n_heads), np.nan, dtype=np.float32)

    import time
    t0 = time.time()
    for i, p in enumerate(pairs):
        clean_r, corr_r = p["clean"], p["corrupt"]
        clean_tok = model.to_tokens(clean_r["prompt"], prepend_bos=False).to(device)
        corr_tok = model.to_tokens(corr_r["prompt"], prepend_bos=False).to(device)
        if clean_tok.shape != corr_tok.shape:
            continue
        # the answer is the clean answer (we want to recover clean signal)
        ans_id = clean_r["answer_token_id"]
        dist_ids = clean_r["distractor_answer_token_ids"]

        # baselines for this pair
        clean_logits = model(clean_tok)
        clean_ld = logit_diff(clean_logits[0, -1], ans_id, dist_ids)
        _, clean_cache = model.run_with_cache(clean_tok)

        corr_logits = model(corr_tok)
        corr_ld = logit_diff(corr_logits[0, -1], ans_id, dist_ids)

        gap = clean_ld - corr_ld
        if abs(gap) < 1e-4:
            continue                          # no signal to recover

        for L in range(n_layers):
            clean_z = clean_cache[f"blocks.{L}.attn.hook_z"]    # [1, T, H, Dh]
            for H in range(n_heads):
                def hook_fn(z, hook, _L=L, _H=H, _cz=clean_z):
                    z[:, :, _H, :] = _cz[:, :, _H, :]
                    return z
                patched = model.run_with_hooks(
                    corr_tok,
                    fwd_hooks=[(f"blocks.{L}.attn.hook_z", hook_fn)],
                )
                patched_ld = logit_diff(patched[0, -1], ans_id, dist_ids)
                effects[i, L, H] = (patched_ld - corr_ld) / gap

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            est_total = elapsed / (i + 1) * n_pairs
            print(f"  [{i+1:>3}/{n_pairs}] elapsed={elapsed:.1f}s est_total={est_total:.0f}s")

    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, "head_effects.npy"), effects)

    mean_eff = np.nanmean(effects, axis=0)         # [L, H]
    std_eff = np.nanstd(effects, axis=0)
    n_valid = np.sum(~np.isnan(effects[:, 0, 0]))

    # heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    vmax = float(np.nanmax(np.abs(mean_eff)))
    im = ax.imshow(mean_eff, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    for L in range(n_layers):
        for H in range(n_heads):
            v = mean_eff[L, H]
            ax.text(H, L, f"{v:+.2f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(v) > vmax * 0.5 else "black")
    ax.set_xlabel("head"); ax.set_ylabel("layer")
    ax.set_title(f"Mean recovery: corrupt→clean head-z patching\n"
                 f"Tier 1, n_valid={n_valid}/{n_pairs}, GPT-2 small")
    plt.colorbar(im, ax=ax, label="recovery fraction")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "head_mean.png"), dpi=120)
    plt.close(fig)

    # top heads
    flat = [(float(mean_eff[L, H]), float(std_eff[L, H]), L, H)
            for L in range(n_layers) for H in range(n_heads)]
    flat.sort(key=lambda t: -t[0])
    top = [{"layer": L, "head": H, "mean_recovery": m, "std": s}
           for m, s, L, H in flat]
    with open(os.path.join(args.out_dir, "top_heads.json"), "w") as f:
        json.dump({
            "n_valid": int(n_valid),
            "tier": "tier1",
            "ranked": top,
        }, f, indent=2)

    print("\n=== Top-10 heads by recovery ===")
    for r in top[:10]:
        print(f"  L{r['layer']:>2}H{r['head']:>2}: mean={r['mean_recovery']:+.4f} std={r['std']:.4f}")
    print(f"\n[done] {time.time() - t0:.1f}s  →  {args.out_dir}")


if __name__ == "__main__":
    main()
