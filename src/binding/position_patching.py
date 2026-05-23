"""Position-level patching for top binding heads.

For Tier 1 n_vars=3 cf pairs (length-14 prompts, 250 pairs):
  prompt token positions (typical):
    0: var_a_name    7: ';'
    1: '='           8: target_var_name
    2: var_a_value   9: '='
    3: ';'          10: source_var_ref   ← source_var_token_pos
    4: source_name  11: ';'
    5: '='          12: final_var_name
    6: source_value 13: '='   ← answer predicted here

For each top head (L, H), patch corrupt run at single position p with clean's
hook_z[:, p, H, :], measure logit_diff recovery. Average over 250 pairs.
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


TOP_HEADS = [(10, 7), (10, 2), (3, 0)]   # L10H7 NNMH, L10H2 BNMH, L3H0 DTH

POSITION_LABELS = [
    "var0_name", "=", "var0_val", ";",
    "src_name", "=", "src_val", ";",
    "tgt_name", "=", "src_ref", ";",
    "tgt_name2", "=",
]                                          # 14 positions


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
    ap.add_argument("--n", type=int, default=250)
    args = ap.parse_args()

    from transformer_lens import HookedTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[load] GPT-2 small on {device}")
    model = HookedTransformer.from_pretrained("gpt2", device=device)
    model.eval()

    pairs = load_pairs(os.path.join(DATASETS_DIR, "var_binding_tier1.jsonl"))
    # Filter to length-14 group (n_vars=3)
    pairs14 = []
    for p in pairs:
        clean_tok = model.to_tokens(p["clean"]["prompt"], prepend_bos=False)
        if clean_tok.shape[1] == 14:
            pairs14.append(p)
    pairs14 = pairs14[: args.n]
    print(f"[data] {len(pairs14)} length-14 cf pairs")

    n_pos = 14
    effects = np.full((len(pairs14), len(TOP_HEADS), n_pos), np.nan, dtype=np.float32)

    import time
    t0 = time.time()
    for i, pair in enumerate(pairs14):
        clean_tok = model.to_tokens(pair["clean"]["prompt"], prepend_bos=False).to(device)
        corr_tok = model.to_tokens(pair["corrupt"]["prompt"], prepend_bos=False).to(device)
        if clean_tok.shape != corr_tok.shape or clean_tok.shape[1] != 14:
            continue
        ans_id = pair["clean"]["answer_token_id"]
        dist_ids = pair["clean"]["distractor_answer_token_ids"]

        _, clean_cache = model.run_with_cache(clean_tok)
        clean_logits = model(clean_tok)
        corr_logits = model(corr_tok)
        clean_ld = logit_diff(clean_logits[0, -1], ans_id, dist_ids)
        corr_ld = logit_diff(corr_logits[0, -1], ans_id, dist_ids)
        gap = clean_ld - corr_ld
        if abs(gap) < 1e-4:
            continue

        for h_idx, (L, H) in enumerate(TOP_HEADS):
            clean_z = clean_cache[f"blocks.{L}.attn.hook_z"]   # [1, T, n_heads, d_head]
            for p in range(n_pos):
                def hook_fn(z, hook, _H=H, _p=p, _cz=clean_z):
                    z[:, _p, _H, :] = _cz[:, _p, _H, :]
                    return z
                patched = model.run_with_hooks(
                    corr_tok,
                    fwd_hooks=[(f"blocks.{L}.attn.hook_z", hook_fn)],
                )
                p_ld = logit_diff(patched[0, -1], ans_id, dist_ids)
                effects[i, h_idx, p] = (p_ld - corr_ld) / gap

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1:>3}/{len(pairs14)}] {elapsed:.0f}s")

    out_dir = os.path.join(RESULTS_DIR, "binding_position")
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "pos_effects.npy"), effects)

    mean_eff = np.nanmean(effects, axis=0)             # [n_heads, n_pos]
    std_eff = np.nanstd(effects, axis=0)
    n_valid = int(np.sum(~np.isnan(effects[:, 0, 0])))

    # Plot per-position recovery for each top head
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(n_pos)
    for h_idx, (L, H) in enumerate(TOP_HEADS):
        ax.plot(x, mean_eff[h_idx], "-o", label=f"L{L}H{H}", linewidth=2)
        ax.fill_between(x,
                        mean_eff[h_idx] - std_eff[h_idx] / np.sqrt(n_valid),
                        mean_eff[h_idx] + std_eff[h_idx] / np.sqrt(n_valid),
                        alpha=0.15)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}\n{lbl}" for p, lbl in enumerate(POSITION_LABELS)],
                       rotation=0, fontsize=8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(10, color="green", linewidth=0.8, linestyle="--", alpha=0.5,
               label="source_var_ref (pos 10)")
    ax.axvline(13, color="red", linewidth=0.8, linestyle="--", alpha=0.5,
               label="final position (pos 13)")
    ax.set_ylabel("mean recovery (±SE)")
    ax.set_title(f"Per-position patching: where does each head carry binding info?\n"
                 f"Tier 1, length-14 prompts, n={n_valid}")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "position_recovery.png"), dpi=120)
    plt.close(fig)

    # Print summary
    print("\n=== Top position per head ===")
    for h_idx, (L, H) in enumerate(TOP_HEADS):
        top_pos = int(np.argmax(mean_eff[h_idx]))
        print(f"  L{L}H{H}: best position = {top_pos} ({POSITION_LABELS[top_pos]}) "
              f"recovery = {mean_eff[h_idx, top_pos]:+.4f}")

    summary = {
        "n_pairs": n_valid,
        "heads": [f"L{L}H{H}" for (L, H) in TOP_HEADS],
        "position_labels": POSITION_LABELS,
        "mean_recovery": mean_eff.tolist(),
        "se_recovery": (std_eff / np.sqrt(n_valid)).tolist(),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[done] {time.time() - t0:.0f}s → {out_dir}")


if __name__ == "__main__":
    main()
