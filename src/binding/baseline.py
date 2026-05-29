"""Step 3-1 baseline: load Tier 1 cf pairs, compute clean/corrupt logit_diff on GPT-2 small.

logit_diff(prompt) = logit(answer) - mean(logit(distractor_answers))

A working binding signal requires clean logit_diff > 0 on average and clean > corrupt.
This script confirms whether GPT-2 small carries enough signal to make patching meaningful.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from shared.config import DATASETS_DIR, RESULTS_DIR


def load_pairs(tier_path):
    """Return list of dicts: {cf_id, clean, corrupt}."""
    by_cf = defaultdict(dict)
    with open(tier_path) as f:
        for line in f:
            r = json.loads(line)
            by_cf[r["cf_id"]][r["role"]] = r
    pairs = []
    for cf_id, d in by_cf.items():
        if "clean" in d and "corrupt" in d:
            pairs.append({"cf_id": cf_id, "clean": d["clean"], "corrupt": d["corrupt"]})
    return pairs


def logit_diff(logits_last, answer_id, distractor_ids):
    """logit_diff = logit(answer) - mean(logit(distractors))."""
    ans = logits_last[answer_id]
    dist = logits_last[distractor_ids].mean()
    return (ans - dist).item()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default=os.path.join(DATASETS_DIR, "var_binding_tier1.jsonl"))
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "binding_baseline.json"))
    args = ap.parse_args()

    from transformer_lens import HookedTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[load] GPT-2 small on {device}")
    model = HookedTransformer.from_pretrained("gpt2", device=device)
    model.eval()

    pairs = load_pairs(args.tier)[: args.n]
    print(f"[data] {len(pairs)} cf pairs from {args.tier}")

    clean_diffs, corrupt_diffs = [], []
    clean_top1, corrupt_top1 = 0, 0
    pct_pos_clean, pct_pos_corrupt = 0, 0
    examples = []

    for i, p in enumerate(pairs):
        for role, diff_list, top1_counter in [
            ("clean", clean_diffs, "clean"),
            ("corrupt", corrupt_diffs, "corrupt"),
        ]:
            r = p[role]
            toks = model.to_tokens(r["prompt"], prepend_bos=False).to(device)
            logits = model(toks)
            last = logits[0, -1]
            ld = logit_diff(last, r["answer_token_id"], r["distractor_answer_token_ids"])
            diff_list.append(ld)
            if last.argmax().item() == r["answer_token_id"]:
                if role == "clean": clean_top1 += 1
                else: corrupt_top1 += 1
            if ld > 0:
                if role == "clean": pct_pos_clean += 1
                else: pct_pos_corrupt += 1

        if i < 5:
            examples.append({
                "cf_id": p["cf_id"],
                "clean_prompt": p["clean"]["prompt"],
                "corrupt_prompt": p["corrupt"]["prompt"],
                "clean_logit_diff": clean_diffs[-1],
                "corrupt_logit_diff": corrupt_diffs[-1],
                "clean_answer": p["clean"]["answer"],
                "corrupt_answer": p["corrupt"]["answer"],
            })

    n = len(pairs)
    stats = {
        "n_pairs": n,
        "clean_logit_diff_mean": float(np.mean(clean_diffs)),
        "clean_logit_diff_std": float(np.std(clean_diffs)),
        "corrupt_logit_diff_mean": float(np.mean(corrupt_diffs)),
        "corrupt_logit_diff_std": float(np.std(corrupt_diffs)),
        "clean_top1_acc": clean_top1 / n,
        "corrupt_top1_acc": corrupt_top1 / n,
        "clean_pct_pos": pct_pos_clean / n,
        "corrupt_pct_pos": pct_pos_corrupt / n,
        "clean_minus_corrupt_diff": float(np.mean([c - cr for c, cr in zip(clean_diffs, corrupt_diffs)])),
        "examples": examples,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(stats, f, indent=2)

    print("\n=== Baseline (GPT-2 small, Tier 1) ===")
    print(f"clean   logit_diff: {stats['clean_logit_diff_mean']:+.4f} ± {stats['clean_logit_diff_std']:.4f}")
    print(f"corrupt logit_diff: {stats['corrupt_logit_diff_mean']:+.4f} ± {stats['corrupt_logit_diff_std']:.4f}")
    print(f"clean - corrupt   : {stats['clean_minus_corrupt_diff']:+.4f}  (patching upper bound)")
    print(f"clean top-1 acc   : {stats['clean_top1_acc']:.4f}")
    print(f"corrupt top-1 acc : {stats['corrupt_top1_acc']:.4f}")
    print(f"clean pct_pos     : {stats['clean_pct_pos']:.4f}")
    print(f"corrupt pct_pos   : {stats['corrupt_pct_pos']:.4f}")
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
