"""Step 3-3: compare code-binding heads vs Wang et al. 2022 IOI circuit.

IOI head taxonomy (Wang et al. 2022, Table 2 / Figure 2):
  - Duplicate Token Heads (DTH)
  - Previous Token Heads (PTH)
  - Induction Heads (IH)
  - S-Inhibition Heads (SIH)
  - Name Mover Heads (NMH)
  - Backup Name Mover Heads (BNMH)
  - Negative Name Mover Heads (NNMH)

Reference: https://arxiv.org/abs/2211.00593 Section 3 / Figure 2.

Compute:
  1. Overlap: top-K code-binding heads ∩ 26 IOI heads
  2. Per-class enrichment: are IOI heads of class C overrepresented in code top heads?
  3. Universality score = rank correlation between IOI importance and code recovery
"""
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from shared.config import RESULTS_DIR


# Wang et al. 2022 — IOI circuit heads (Figure 2 / Table 2 of arXiv:2211.00593)
IOI_HEADS = {
    "DTH":  [(0, 1), (0, 10), (3, 0)],
    "PTH":  [(2, 2), (4, 11)],
    "IH":   [(5, 5), (5, 8), (5, 9), (6, 9)],
    "SIH":  [(7, 3), (7, 9), (8, 6), (8, 10)],
    "NMH":  [(9, 6), (9, 9), (10, 0)],
    "BNMH": [(9, 0), (9, 7), (10, 1), (10, 2), (10, 6), (10, 10), (11, 2), (11, 9)],
    "NNMH": [(10, 7), (11, 10)],
}

# Importance ranking in IOI (qualitative, from Wang et al.):
# NMH > SIH ≈ NNMH > BNMH ≈ IH > DTH > PTH
IOI_CLASS_RANK = {"NMH": 6, "SIH": 5, "NNMH": 5, "BNMH": 4, "IH": 4, "DTH": 3, "PTH": 2}


def all_ioi_heads():
    return [(L, H, cls) for cls, lst in IOI_HEADS.items() for (L, H) in lst]


def load_effects(path):
    return np.load(path)                              # [n_pairs, 12, 12]


def main():
    eff = load_effects(os.path.join(RESULTS_DIR, "binding_patching/head_effects.npy"))
    mean_eff = np.nanmean(eff, axis=0)                # [12, 12]
    n = eff.shape[0]
    n_layers, n_heads = mean_eff.shape

    out_dir = os.path.join(RESULTS_DIR, "binding_compare")
    os.makedirs(out_dir, exist_ok=True)

    # ---- 1. Overlap: top-K code heads ∩ IOI 26 ----
    flat = [(float(mean_eff[L, H]), L, H) for L in range(n_layers) for H in range(n_heads)]
    flat.sort(key=lambda t: -t[0])
    ioi_set = {(L, H): cls for cls, lst in IOI_HEADS.items() for (L, H) in lst}

    print(f"=== Top-26 code-binding heads vs IOI 26 ===")
    top26_code = [(L, H, m) for m, L, H in flat[:26]]
    overlap = [(L, H, m, ioi_set[(L, H)]) for (L, H, m) in top26_code if (L, H) in ioi_set]
    print(f"Overlap |top26_code ∩ IOI26| = {len(overlap)}  (random expected ≈ 26*26/144 ≈ 4.7)")
    print()
    print(f"{'rank':>4} {'head':>6} {'recovery':>10}  {'IOI class':>10}")
    for rank, (m, L, H) in enumerate(flat[:30], 1):
        cls = ioi_set.get((L, H), "—")
        print(f"  {rank:>2} L{L:>2}H{H:>2}   {m:+.4f}  {cls:>10}")

    # ---- 2. Per-class enrichment ----
    print("\n=== Per-class mean recovery ===")
    class_stats = {}
    for cls, lst in IOI_HEADS.items():
        vals = [mean_eff[L, H] for (L, H) in lst]
        class_stats[cls] = {
            "heads": lst,
            "mean_recovery": float(np.mean(vals)),
            "max_recovery": float(np.max(vals)),
        }
        print(f"  {cls:>5} (n={len(lst)}): mean={np.mean(vals):+.4f}  max={np.max(vals):+.4f}  "
              f"members={[f'L{L}H{H}' for L, H in lst]}")

    non_ioi_vals = [mean_eff[L, H] for L in range(n_layers) for H in range(n_heads)
                    if (L, H) not in ioi_set]
    print(f"  {'NON-IOI':>5} (n={len(non_ioi_vals)}): mean={np.mean(non_ioi_vals):+.4f}")

    # ---- 3. Universality score (rank correlation IOI importance vs code recovery) ----
    # Score IOI heads by their class rank, then correlate with code recovery
    ioi_scores = []
    code_scores = []
    for cls, lst in IOI_HEADS.items():
        for (L, H) in lst:
            ioi_scores.append(IOI_CLASS_RANK[cls])
            code_scores.append(float(mean_eff[L, H]))
    from scipy.stats import spearmanr, pearsonr
    rho, p_rho = spearmanr(ioi_scores, code_scores)
    r, p_r = pearsonr(ioi_scores, code_scores)
    print(f"\n=== Universality score ===")
    print(f"Spearman ρ (IOI class rank vs code recovery on 26 IOI heads): "
          f"{rho:+.4f}  (p={p_rho:.3f})")
    print(f"Pearson  r: {r:+.4f}  (p={p_r:.3f})")

    # Alternative score: fraction of top-K code heads that are IOI heads
    print(f"\nFraction of top-K code heads that are IOI heads:")
    for K in [5, 10, 15, 20, 26]:
        topK = [(L, H) for _, L, H in flat[:K]]
        frac = sum(1 for h in topK if h in ioi_set) / K
        random = 26 / 144
        print(f"  K={K:>2}: frac={frac:.2%}  (random baseline {random:.2%}, enrichment ×{frac/random:.2f})")

    # ---- 4. Plot: code recovery heatmap with IOI heads marked ----
    fig, ax = plt.subplots(figsize=(9, 7))
    vmax = float(np.nanmax(np.abs(mean_eff)))
    im = ax.imshow(mean_eff, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    for L in range(n_layers):
        for H in range(n_heads):
            v = mean_eff[L, H]
            ax.text(H, L, f"{v:+.2f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(v) > vmax * 0.5 else "black")
    # Mark IOI heads with category abbrev
    for cls, lst in IOI_HEADS.items():
        for (L, H) in lst:
            ax.add_patch(plt.Rectangle((H - 0.5, L - 0.5), 1, 1, fill=False,
                                        edgecolor="lime", linewidth=1.5))
    ax.set_xlabel("head"); ax.set_ylabel("layer")
    ax.set_title(f"Code binding patching recovery — IOI heads outlined in green\n"
                 f"(GPT-2 small, Tier 1, n={n})")
    plt.colorbar(im, ax=ax, label="recovery fraction")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "code_vs_ioi.png"), dpi=120)
    plt.close(fig)

    # ---- 5. Save universality report ----
    report = {
        "n_pairs": int(n),
        "top26_code_heads": [{"layer": L, "head": H, "recovery": m,
                              "ioi_class": ioi_set.get((L, H))}
                             for m, L, H in flat[:26]],
        "overlap_top26": [{"layer": L, "head": H, "recovery": m, "ioi_class": cls}
                          for (L, H, m, cls) in overlap],
        "overlap_count": len(overlap),
        "random_baseline_overlap": 26 * 26 / 144,
        "ioi_class_stats": class_stats,
        "non_ioi_mean_recovery": float(np.mean(non_ioi_vals)),
        "universality": {
            "spearman_rho": float(rho),
            "spearman_p": float(p_rho),
            "pearson_r": float(r),
            "pearson_p": float(p_r),
        },
    }
    with open(os.path.join(out_dir, "universality_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[done] → {out_dir}")


if __name__ == "__main__":
    main()
