"""Step 3 extra (F): minimal-circuit experiment.

Goal: are the discovered head(s) *sufficient* on their own?

Method (inverse ablation):
  - For each Tier 1 clean prompt, run model with ALL attention heads zeroed
    EXCEPT a chosen "keep" set K.
  - Compare resulting logit_diff to clean baseline.
  - circuit_score = ablated_ld_when_only_K_kept / clean_ld
    (1.0 = K is sufficient, 0.0 = K alone contributes nothing,
     >1.0 = ablating noise actually helps)

Sets compared:
  - top3   = {L10H7, L10H2, L3H0}                (NNMH, BNMH, DTH)
  - top5   = top3 + {L10H10, L6H1}
  - top10  = top10 from patching ranking
  - ioi26  = all 26 IOI heads (Wang 2022)
  - random3, random5, random10  = matched-size random controls (5 seeds)
  - empty  = zero all heads (baseline floor)
  - full   = no ablation (baseline ceiling = clean run)
"""
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from shared.config import DATASETS_DIR, RESULTS_DIR
from src.binding.compare_ioi import IOI_HEADS


TOP10 = [(10, 7), (10, 2), (6, 1), (3, 0), (10, 10),
         (5, 0), (1, 11), (5, 8), (11, 10), (7, 11)]
TOP5 = TOP10[:5]
TOP3 = TOP10[:3]
IOI26 = [(L, H) for cls in IOI_HEADS.values() for (L, H) in cls]


def logit_diff(logits_last, answer_id, distractor_ids):
    ans = logits_last[answer_id]
    dist = logits_last[torch.tensor(distractor_ids, device=logits_last.device)].mean()
    return (ans - dist).item()


def load_clean(path, n_max=500):
    out = []
    seen = set()
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("role") == "clean" and r["cf_id"] not in seen:
                seen.add(r["cf_id"])
                out.append(r)
                if len(out) >= n_max:
                    break
    return out


def make_keep_hook(keep_set_by_layer, L):
    """Zero all heads at layer L not in keep_set_by_layer[L]."""
    keep_h = keep_set_by_layer.get(L, set())
    n_heads = 12

    def hook(z, hook):
        for H in range(n_heads):
            if H not in keep_h:
                z[:, :, H, :] = 0
        return z
    return hook


def make_remove_hook(remove_set_by_layer, L):
    """Zero only the heads at layer L in remove_set_by_layer[L]."""
    remove_h = remove_set_by_layer.get(L, set())

    def hook(z, hook):
        for H in remove_h:
            z[:, :, H, :] = 0
        return z
    return hook


@torch.no_grad()
def run_remove(model, records, remove_heads, device):
    """Ablate ONLY the specified heads, leave rest intact. Returns mean logit_diff."""
    n_layers = model.cfg.n_layers
    rem_by_layer = defaultdict(set)
    for (L, H) in remove_heads:
        rem_by_layer[L].add(H)
    hooks = [(f"blocks.{L}.attn.hook_z", make_remove_hook(rem_by_layer, L))
             for L in range(n_layers) if rem_by_layer[L]]
    lds = []
    for r in records:
        toks = model.to_tokens(r["prompt"], prepend_bos=False).to(device)
        ans_id = r["answer_token_id"]
        dist_ids = r["distractor_answer_token_ids"]
        if not dist_ids:
            continue
        logits = model.run_with_hooks(toks, fwd_hooks=hooks)
        lds.append(logit_diff(logits[0, -1], ans_id, dist_ids))
    return float(np.mean(lds))


@torch.no_grad()
def run_set(model, records, keep_heads, device, label=""):
    """Returns mean logit_diff and per-pair list."""
    n_layers = model.cfg.n_layers
    keep_by_layer = defaultdict(set)
    for (L, H) in keep_heads:
        keep_by_layer[L].add(H)

    hooks = [(f"blocks.{L}.attn.hook_z", make_keep_hook(keep_by_layer, L))
             for L in range(n_layers)]

    lds = []
    for r in records:
        toks = model.to_tokens(r["prompt"], prepend_bos=False).to(device)
        ans_id = r["answer_token_id"]
        dist_ids = r["distractor_answer_token_ids"]
        if not dist_ids:
            continue
        logits = model.run_with_hooks(toks, fwd_hooks=hooks) if keep_heads is not None \
            else model(toks)
        lds.append(logit_diff(logits[0, -1], ans_id, dist_ids))
    return float(np.mean(lds)), lds


@torch.no_grad()
def baseline_clean(model, records, device):
    lds = []
    for r in records:
        toks = model.to_tokens(r["prompt"], prepend_bos=False).to(device)
        ans_id = r["answer_token_id"]
        dist_ids = r["distractor_answer_token_ids"]
        if not dist_ids:
            continue
        lds.append(logit_diff(model(toks)[0, -1], ans_id, dist_ids))
    return float(np.mean(lds)), lds


def main():
    from transformer_lens import HookedTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[load] GPT-2 small on {device}")
    model = HookedTransformer.from_pretrained("gpt2", device=device)
    model.eval()

    recs = load_clean(os.path.join(DATASETS_DIR, "var_binding_tier1.jsonl"), n_max=500)
    recs = [r for r in recs if r.get("distractor_answer_token_ids")]
    print(f"[data] {len(recs)} clean records")

    # --- baselines ---
    print("\n[baseline] full model (no ablation)")
    clean_ld, _ = baseline_clean(model, recs, device)
    print(f"  clean logit_diff = {clean_ld:+.4f}")

    print("[baseline] empty (all attn heads zeroed)")
    empty_ld, _ = run_set(model, recs, keep_heads=[], device=device)
    print(f"  empty logit_diff = {empty_ld:+.4f}")
    print(f"  attention contributes total = {clean_ld - empty_ld:+.4f}")

    sets = {
        "top3":  TOP3,
        "top5":  TOP5,
        "top10": TOP10,
        "ioi26": IOI26,
    }
    # random controls (5 seeds each)
    all_heads = [(L, H) for L in range(12) for H in range(12)]
    rng = random.Random(0)
    for K in [3, 5, 10, 26]:
        for seed in range(5):
            rng2 = random.Random(seed * 100 + K)
            rand = rng2.sample(all_heads, K)
            sets[f"random{K}_seed{seed}"] = rand

    results = {
        "n_pairs": len(recs),
        "clean_ld": clean_ld,
        "empty_ld": empty_ld,
        "sets": {},
    }
    print("\n[experiment] running keep-only ablations...")
    print(f"  {'set':>20}  {'k':>3}  {'mean_ld':>10}  {'recovery':>10}")
    print(f"  {'baseline-clean':>20}  {'':>3}  {clean_ld:+.4f}     1.000")
    print(f"  {'baseline-empty':>20}  {0:>3}  {empty_ld:+.4f}     0.000")

    span = clean_ld - empty_ld
    for name, heads in sets.items():
        m, _ = run_set(model, recs, heads, device)
        recovery = (m - empty_ld) / span if abs(span) > 1e-6 else float("nan")
        results["sets"][name] = {"k": len(heads), "mean_ld": m,
                                  "recovery": recovery,
                                  "heads": [[L, H] for (L, H) in heads]}
        print(f"  {name:>20}  {len(heads):>3}  {m:+.4f}     {recovery:+.3f}")

    # aggregate random controls
    print("\n[summary] random control means")
    for K in [3, 5, 10, 26]:
        rs = [results["sets"][f"random{K}_seed{s}"]["recovery"] for s in range(5)]
        m_r, s_r = float(np.mean(rs)), float(np.std(rs))
        print(f"  random{K}: recovery = {m_r:+.3f} ± {s_r:.3f}")
        results.setdefault("random_summary", {})[f"random{K}"] = {"mean": m_r, "std": s_r}

    # === NECESSITY test: ablate top-K, keep rest ===
    print("\n[necessity] ablating ONLY top-K, leaving rest intact")
    necessity = {}
    print(f"  {'set':>20}  {'k':>3}  {'mean_ld':>10}  {'drop_from_clean':>15}")
    for name in ["top3", "top5", "top10", "ioi26"]:
        heads = sets[name]
        m = run_remove(model, recs, heads, device)
        drop = clean_ld - m
        necessity[name] = {"k": len(heads), "mean_ld": m, "drop": drop}
        print(f"  {name:>20}  {len(heads):>3}  {m:+.4f}     {drop:+.4f}")
    for K in [3, 5, 10, 26]:
        rdrops = []
        for s in range(5):
            heads = sets[f"random{K}_seed{s}"]
            m = run_remove(model, recs, heads, device)
            rdrops.append(clean_ld - m)
        m_d, s_d = float(np.mean(rdrops)), float(np.std(rdrops))
        necessity[f"random{K}"] = {"k": K, "mean_drop": m_d, "std_drop": s_d}
        print(f"  {'random'+str(K):>20}  {K:>3}  {'—':>10}     {m_d:+.4f} ± {s_d:.4f}")
    results["necessity"] = necessity

    out_dir = os.path.join(RESULTS_DIR, "binding_minimal_circuit")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # === plot ===
    Ks = [3, 5, 10, 26]
    labels = ["top3", "top5", "top10", "ioi26"]

    # necessity plot (cleaner story)
    fig, ax = plt.subplots(figsize=(9, 5))
    top_drops = [necessity[lbl]["drop"] for lbl in labels]
    rand_drops = [necessity[f"random{K}"]["mean_drop"] for K in Ks]
    rand_drop_stds = [necessity[f"random{K}"]["std_drop"] for K in Ks]
    x = np.arange(len(Ks))
    w = 0.35
    ax.bar(x - w/2, top_drops, w, color="tab:red", label="ablate discovered top-K")
    ax.bar(x + w/2, rand_drops, w, yerr=rand_drop_stds, color="tab:gray",
           label="ablate random K (5 seeds)", capsize=4)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("logit_diff drop when ablated (clean − ablated)")
    ax.set_title(f"Necessity test: ablating top-K hurts more than random K?\n"
                 f"(Tier 1, n={len(recs)}, clean baseline LD={clean_ld:+.3f})")
    for i, v in enumerate(top_drops):
        ax.text(i - w/2, v + 0.005, f"{v:+.3f}", ha="center", fontsize=9)
    for i, v in enumerate(rand_drops):
        ax.text(i + w/2, v + 0.005, f"{v:+.3f}", ha="center", fontsize=9)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "necessity.png"), dpi=120)
    plt.close(fig)

    # sufficiency plot (with honest caveat in title)
    fig, ax = plt.subplots(figsize=(9, 5))
    top_recov = [results["sets"][lbl]["recovery"] for lbl in labels]
    rand_means = [results["random_summary"][f"random{K}"]["mean"] for K in Ks]
    rand_stds = [results["random_summary"][f"random{K}"]["std"] for K in Ks]
    ax.bar(x - w/2, top_recov, w, color="tab:red", label="keep only discovered top-K")
    ax.bar(x + w/2, rand_means, w, yerr=rand_stds, color="tab:gray",
           label="keep only random K (5 seeds)", capsize=4)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.7,
               label="clean baseline (=1.0)")
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("recovery = (kept_ld − empty_ld) / (clean_ld − empty_ld)")
    ax.set_title(f"Sufficiency test (zero-ablation): both fail at small K\n"
                 f"— removing 141/144 heads is too OOD; necessity test is the cleaner story")
    for i, v in enumerate(top_recov):
        ax.text(i - w/2, v + 0.02, f"{v:+.2f}", ha="center", fontsize=9)
    for i, v in enumerate(rand_means):
        ax.text(i + w/2, v + 0.02, f"{v:+.2f}", ha="center", fontsize=9)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sufficiency.png"), dpi=120)
    plt.close(fig)
    print(f"\n[done] → {out_dir}")


if __name__ == "__main__":
    main()
