"""Step 3 extra (B): direct logit attribution for top binding heads.

Patching tells us "this head matters". Logit attribution tells us "this head
contributes +X to logit(answer) − logit(distractor) at the final position via
the direct path (residual → unembed)".

Method:
  - Run clean prompts with cache.
  - For each head (L, H), get its output at the final position
    (blocks.{L}.attn.hook_result, summed across heads doesn't help — we
    use per-head decomposition via cache["result", L][:, -1, H, :]).
  - Project onto W_U[:, answer] - W_U[:, distractor_mean] direction.
  - This is the *direct* contribution (ignoring downstream MLP/attn effects).
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from shared.config import DATASETS_DIR, RESULTS_DIR


TOP_HEADS = [(10, 7), (10, 2), (3, 0), (10, 10), (6, 1)]   # top-5 from patching


def load_pairs(path):
    by_cf = defaultdict(dict)
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            by_cf[r["cf_id"]][r["role"]] = r
    return [{"cf_id": cf, "clean": d["clean"], "corrupt": d["corrupt"]}
            for cf, d in by_cf.items() if "clean" in d and "corrupt" in d]


@torch.no_grad()
def main():
    from transformer_lens import HookedTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[load] GPT-2 small on {device}")
    model = HookedTransformer.from_pretrained("gpt2", device=device)
    model.eval()
    # We need per-head output decomposition
    model.set_use_attn_result(True)

    W_U = model.W_U                                   # [d_model, d_vocab]
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads

    pairs = load_pairs(os.path.join(DATASETS_DIR, "var_binding_tier1.jsonl"))[:500]
    print(f"[data] {len(pairs)} cf pairs")

    # per-head direct contribution to (logit_ans - mean logit_dist)
    per_head_attr = np.full((len(pairs), n_layers, n_heads), np.nan, dtype=np.float32)
    # also track total direct logit_diff for sanity
    total_direct_ld = []

    import time
    t0 = time.time()
    for i, pair in enumerate(pairs):
        tok = model.to_tokens(pair["clean"]["prompt"], prepend_bos=False).to(device)
        ans_id = pair["clean"]["answer_token_id"]
        dist_ids = pair["clean"]["distractor_answer_token_ids"]
        if not dist_ids:
            continue

        # answer direction in residual space
        dist_mean = W_U[:, dist_ids].mean(dim=1)      # [d_model]
        direction = W_U[:, ans_id] - dist_mean        # [d_model]

        _, cache = model.run_with_cache(tok)

        # final position
        T = tok.shape[1]
        ld_sum = 0.0
        for L in range(n_layers):
            # cache["result", L] shape: [batch, pos, n_heads, d_model]
            result = cache["result", L][0, T - 1]     # [n_heads, d_model]
            attr = (result @ direction).cpu().numpy() # [n_heads]
            per_head_attr[i, L, :] = attr
            ld_sum += attr.sum()
        total_direct_ld.append(float(ld_sum))

        if (i + 1) % 100 == 0:
            print(f"  [{i+1:>3}/{len(pairs)}] {time.time()-t0:.0f}s")

    out_dir = os.path.join(RESULTS_DIR, "binding_logit_attr")
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "per_head_attr.npy"), per_head_attr)

    mean_attr = np.nanmean(per_head_attr, axis=0)     # [12, 12]
    se_attr = np.nanstd(per_head_attr, axis=0) / np.sqrt(per_head_attr.shape[0])

    # === heatmap ===
    fig, ax = plt.subplots(figsize=(10, 8))
    vmax = float(np.nanmax(np.abs(mean_attr)))
    im = ax.imshow(mean_attr, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    for L in range(n_layers):
        for H in range(n_heads):
            v = mean_attr[L, H]
            ax.text(H, L, f"{v:+.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > vmax * 0.5 else "black")
    ax.set_xticks(range(n_heads)); ax.set_yticks(range(n_layers))
    ax.set_xlabel("head"); ax.set_ylabel("layer")
    ax.set_title(f"Direct logit attribution per head\n"
                 f"contribution to logit(answer) − mean logit(distractor) at final pos\n"
                 f"(Tier 1, n={per_head_attr.shape[0]})")
    plt.colorbar(im, ax=ax, label="logits")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "logit_attr_heatmap.png"), dpi=120)
    plt.close(fig)

    # === top heads summary ===
    print("\n=== Direct logit attribution for top patching heads ===")
    summary = {"n_pairs": int(per_head_attr.shape[0]),
               "mean_total_direct_ld": float(np.mean(total_direct_ld)),
               "top_heads": {}}
    print(f"  total direct logit_diff (sum over all heads) = "
          f"{np.mean(total_direct_ld):+.4f}")
    for (L, H) in TOP_HEADS:
        m, s = float(mean_attr[L, H]), float(se_attr[L, H])
        z = m / s if s > 0 else float("nan")
        summary["top_heads"][f"L{L}H{H}"] = {"mean": m, "se": s, "z_score": z}
        sign = "↑(supports answer)" if m > 0 else "↓(boosts distractor)"
        print(f"  L{L:>2}H{H:>2}: direct contribution = {m:+.4f} ± {s:.4f}  "
              f"(z={z:+.1f})  {sign}")

    # === full ranking ===
    flat = [(float(mean_attr[L, H]), L, H) for L in range(n_layers)
            for H in range(n_heads)]
    print("\n=== Top-10 positive (supports answer) ===")
    for m, L, H in sorted(flat, key=lambda t: -t[0])[:10]:
        print(f"  L{L:>2}H{H:>2}: {m:+.4f}")
    print("\n=== Top-10 negative (boosts distractor) ===")
    for m, L, H in sorted(flat, key=lambda t: t[0])[:10]:
        print(f"  L{L:>2}H{H:>2}: {m:+.4f}")

    summary["top10_positive"] = [{"layer": L, "head": H, "attr": m}
                                  for m, L, H in sorted(flat, key=lambda t: -t[0])[:10]]
    summary["top10_negative"] = [{"layer": L, "head": H, "attr": m}
                                  for m, L, H in sorted(flat, key=lambda t: t[0])[:10]]
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[done] {time.time()-t0:.0f}s → {out_dir}")


if __name__ == "__main__":
    main()
