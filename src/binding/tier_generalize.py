"""Step 3 extra: do Tier-1-discovered binding heads also matter on Tier 2/3?

Tier 2/3 have no cf pairs, so we use head-ablation rather than patching.
For each tier:
  baseline logit_diff = logit(answer) - mean(logit(distractor_answers))
  ablated logit_diff  = same metric with one head zeroed out
  effect = baseline - ablated  (positive = head was helping)

We compare:
  - Top-5 code-binding heads (from Tier 1 patching)
  - Random control heads (5 non-IOI heads)
to test whether the circuit generalizes to noisier data.
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


# Top-5 code-binding heads (from Step 3 patching results)
TOP5_HEADS = [(10, 7), (10, 2), (6, 1), (3, 0), (10, 10)]
# Random non-IOI controls (chosen from low-recovery cells in the heatmap)
CONTROL_HEADS = [(0, 4), (2, 6), (4, 2), (4, 8), (8, 0)]


def load_clean(tier_path):
    records = []
    with open(tier_path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("role", "single") in ("clean", "single"):
                records.append(r)
    return records


def logit_diff(logits_last, answer_id, distractor_ids):
    if not distractor_ids:
        return float("nan")
    ans = logits_last[answer_id]
    dist = logits_last[torch.tensor(distractor_ids, device=logits_last.device)].mean()
    return (ans - dist).item()


@torch.no_grad()
def measure_tier(model, records, head_list, device, label=""):
    """Returns dict head→(baseline_mean_ld, ablated_mean_ld)."""
    baseline_lds = []
    ablate_lds = {(L, H): [] for (L, H) in head_list}

    for i, r in enumerate(records):
        toks = model.to_tokens(r["prompt"], prepend_bos=False).to(device)
        ans_id = r["answer_token_id"]
        dist_ids = r["distractor_answer_token_ids"]
        if not dist_ids:
            continue

        base_logits = model(toks)
        baseline_lds.append(logit_diff(base_logits[0, -1], ans_id, dist_ids))

        for (L, H) in head_list:
            def hook_fn(z, hook, _H=H):
                z[:, :, _H, :] = 0
                return z
            patched = model.run_with_hooks(
                toks,
                fwd_hooks=[(f"blocks.{L}.attn.hook_z", hook_fn)],
            )
            ablate_lds[(L, H)].append(logit_diff(patched[0, -1], ans_id, dist_ids))

    base_mean = float(np.nanmean(baseline_lds))
    out = {"baseline_mean": base_mean, "n": len(baseline_lds), "heads": {}}
    for (L, H), lds in ablate_lds.items():
        m = float(np.nanmean(lds))
        out["heads"][f"L{L}H{H}"] = {
            "ablated_mean_logit_diff": m,
            "drop": base_mean - m,
        }
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()

    from transformer_lens import HookedTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[load] GPT-2 small on {device}")
    model = HookedTransformer.from_pretrained("gpt2", device=device)
    model.eval()

    out_dir = os.path.join(RESULTS_DIR, "binding_tier_generalize")
    os.makedirs(out_dir, exist_ok=True)

    results = {}
    tier_paths = {
        "tier1": os.path.join(DATASETS_DIR, "var_binding_tier1.jsonl"),
        "tier2": os.path.join(DATASETS_DIR, "var_binding_tier2.jsonl"),
        "tier3": os.path.join(DATASETS_DIR, "var_binding_tier3.jsonl"),
    }
    head_list = TOP5_HEADS + CONTROL_HEADS

    for tag, path in tier_paths.items():
        recs = load_clean(path)[: args.n]
        # Tier 3 has no distractor_answer_token_ids by design (multi-hop accuracy-only)
        recs = [r for r in recs if r.get("distractor_answer_token_ids")]
        if not recs:
            print(f"\n=== {tag}: skipped (no distractor field; multi-hop accuracy-only) ===")
            results[tag] = {"skipped": True, "reason": "no distractor_answer_token_ids"}
            continue
        print(f"\n=== {tag} (n={len(recs)}) ===")
        r = measure_tier(model, recs, head_list, device, label=tag)
        results[tag] = r
        print(f"baseline logit_diff mean = {r['baseline_mean']:+.4f}")
        print(f"{'head':>8} {'role':>10} {'ablated_LD':>12} {'drop':>10}")
        for (L, H) in head_list:
            role = "TOP5" if (L, H) in TOP5_HEADS else "control"
            d = r["heads"][f"L{L}H{H}"]
            print(f"  L{L:>2}H{H:>2}  {role:>9}  {d['ablated_mean_logit_diff']:+.4f}    {d['drop']:+.4f}")

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Plot: drop per head across tiers, group by role
    tiers = [t for t in results.keys() if not results[t].get("skipped")]
    head_labels = [f"L{L}H{H}" for (L, H) in head_list]
    role_color = {(L, H): "tab:red" if (L, H) in TOP5_HEADS else "tab:gray" for (L, H) in head_list}

    fig, axes = plt.subplots(1, len(tiers), figsize=(4 * len(tiers), 4), sharey=True)
    if len(tiers) == 1:
        axes = [axes]
    for ax, tag in zip(axes, tiers):
        drops = [results[tag]["heads"][lbl]["drop"] for lbl in head_labels]
        colors = [role_color[(L, H)] for (L, H) in head_list]
        ax.bar(range(len(head_labels)), drops, color=colors)
        ax.set_xticks(range(len(head_labels)))
        ax.set_xticklabels(head_labels, rotation=45)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(f"{tag}  (baseline LD={results[tag]['baseline_mean']:+.3f})")
        ax.set_ylabel("ablation drop (baseline − ablated logit_diff)")
    fig.suptitle("Do Tier-1-discovered heads generalize? Red=top5, gray=non-IOI control")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "tier_generalize.png"), dpi=120)
    plt.close(fig)
    print(f"\n[done] → {out_dir}")


if __name__ == "__main__":
    main()
