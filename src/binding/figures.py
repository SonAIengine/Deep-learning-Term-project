"""Polished publication-quality figures for Step 3 results.

Produces:
  - heatmap_classes.png — code recovery heatmap with IOI class-colored outlines
  - universality_summary.png — bar chart of per-class transfer + enrichment line
  - circuit_diagram.png — schematic of discovered circuit
"""
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from shared.config import RESULTS_DIR
from src.binding.compare_ioi import IOI_HEADS


CLASS_COLOR = {
    "DTH":  "#1f77b4",   # blue   — Duplicate Token
    "PTH":  "#17becf",   # cyan   — Previous Token
    "IH":   "#2ca02c",   # green  — Induction
    "SIH":  "#9467bd",   # purple — S-Inhibition
    "NMH":  "#ff7f0e",   # orange — Name Mover
    "BNMH": "#d62728",   # red    — Backup Name Mover
    "NNMH": "#8c564b",   # brown  — Negative Name Mover
}


def heatmap_classes():
    eff = np.load(os.path.join(RESULTS_DIR, "binding_patching/head_effects.npy"))
    mean_eff = np.nanmean(eff, axis=0)
    n_layers, n_heads = mean_eff.shape

    fig, ax = plt.subplots(figsize=(10, 8))
    vmax = float(np.nanmax(np.abs(mean_eff)))
    im = ax.imshow(mean_eff, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    for L in range(n_layers):
        for H in range(n_heads):
            v = mean_eff[L, H]
            ax.text(H, L, f"{v:+.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > vmax * 0.5 else "black")

    # color-coded IOI outlines
    for cls, lst in IOI_HEADS.items():
        for (L, H) in lst:
            ax.add_patch(plt.Rectangle((H - 0.5, L - 0.5), 1, 1,
                                       fill=False, edgecolor=CLASS_COLOR[cls],
                                       linewidth=2.2))

    ax.set_xticks(range(n_heads))
    ax.set_yticks(range(n_layers))
    ax.set_xlabel("attention head", fontsize=11)
    ax.set_ylabel("layer", fontsize=11)
    ax.set_title("Code variable binding — patching recovery (GPT-2 small, Tier 1 n=500)\n"
                 "IOI heads (Wang et al. 2022) outlined by category",
                 fontsize=11)

    handles = [mpatches.Patch(facecolor="none", edgecolor=CLASS_COLOR[c],
                              linewidth=2.2, label=c) for c in IOI_HEADS]
    ax.legend(handles=handles, loc="center left",
              bbox_to_anchor=(1.15, 0.5), fontsize=9, title="IOI class")
    plt.colorbar(im, ax=ax, label="recovery fraction", shrink=0.7)
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "binding_compare/heatmap_classes.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


def universality_summary():
    with open(os.path.join(RESULTS_DIR, "binding_compare/universality_report.json")) as f:
        rep = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Per-class mean recovery
    classes = ["NNMH", "BNMH", "NMH", "DTH", "PTH", "IH", "SIH"]
    means = [rep["ioi_class_stats"][c]["mean_recovery"] for c in classes]
    colors = [CLASS_COLOR[c] for c in classes]
    bars = ax1.bar(classes, means, color=colors)
    ax1.axhline(rep["non_ioi_mean_recovery"], color="gray", linestyle="--",
                label=f"non-IOI baseline ({rep['non_ioi_mean_recovery']:+.3f})")
    ax1.axhline(0, color="black", linewidth=0.5)
    for bar, m in zip(bars, means):
        ax1.text(bar.get_x() + bar.get_width() / 2, m + (0.005 if m >= 0 else -0.015),
                 f"{m:+.3f}", ha="center", fontsize=9)
    ax1.set_ylabel("mean recovery fraction", fontsize=11)
    ax1.set_title("Per-class transfer to code binding\n"
                  "(positive = head's role transfers from IOI to code)", fontsize=11)
    ax1.legend(loc="upper right")

    # Top-K enrichment
    K_values = [5, 10, 15, 20, 26]
    fracs = []
    random_base = 26 / 144
    for K in K_values:
        topK = [(t["layer"], t["head"]) for t in rep["top26_code_heads"][:K]]
        ioi_set = {(L, H) for cls, lst in IOI_HEADS.items() for (L, H) in lst}
        fracs.append(sum(1 for h in topK if h in ioi_set) / K)
    enrich = [f / random_base for f in fracs]

    ax2.bar([f"K={K}" for K in K_values], enrich, color="#d62728")
    ax2.axhline(1.0, color="black", linestyle="--", linewidth=0.7,
                label="random baseline (×1)")
    for i, (K, f, e) in enumerate(zip(K_values, fracs, enrich)):
        ax2.text(i, e + 0.1, f"{f*100:.0f}%\n×{e:.1f}", ha="center", fontsize=9)
    ax2.set_ylabel("enrichment over random baseline", fontsize=11)
    ax2.set_title("Top-K code heads ∩ IOI 26\n(% of top-K that are IOI heads / random expectation)",
                  fontsize=11)
    ax2.legend(loc="upper right")
    ax2.set_ylim(0, max(enrich) * 1.25)

    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "binding_compare/universality_summary.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  → {out}")


def circuit_diagram():
    """Schematic of the discovered code-binding circuit, aligned to position flow."""
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Token row
    tokens = ["x", "=", "5", ";", " z", "=", "9", ";", " a", "=", "z", ";", " a", "="]
    for i, t in enumerate(tokens):
        ax.add_patch(plt.Rectangle((i + 0.05, 0.2), 0.9, 0.8,
                                   facecolor="#eee", edgecolor="black", linewidth=0.5))
        ax.text(i + 0.5, 0.6, t, ha="center", va="center", fontsize=11,
                family="monospace")
        ax.text(i + 0.5, -0.1, f"{i}", ha="center", va="top", fontsize=8,
                color="gray")

    # Annotations
    def head_box(x, y, label, color, info=""):
        ax.add_patch(plt.Rectangle((x - 0.5, y - 0.3), 1.6, 0.6,
                                   facecolor=color, alpha=0.4, edgecolor=color, linewidth=1.5))
        ax.text(x + 0.3, y + 0.05, label, ha="center", va="center",
                fontsize=10, fontweight="bold")
        if info:
            ax.text(x + 0.3, y - 0.25, info, ha="center", va="center", fontsize=8)

    # L3H0 at pos 10 (src_ref)
    head_box(10, 2, "L3H0 (DTH)", CLASS_COLOR["DTH"], "duplicate token detect")
    ax.annotate("", xy=(10.5, 1.05), xytext=(10.5, 1.6),
                arrowprops=dict(arrowstyle="-|>", color=CLASS_COLOR["DTH"]))

    # L10H2 + L10H7 at pos 13 (final =)
    head_box(13, 3, "L10H2 (BNMH)", CLASS_COLOR["BNMH"], "+ correct value")
    ax.annotate("", xy=(13.5, 1.05), xytext=(13.5, 2.6),
                arrowprops=dict(arrowstyle="-|>", color=CLASS_COLOR["BNMH"]))

    head_box(13, 4.2, "L10H7 (NNMH)", CLASS_COLOR["NNMH"], "− distractor value")
    ax.annotate("", xy=(13.5, 3.5), xytext=(13.5, 3.9),
                arrowprops=dict(arrowstyle="-|>", color=CLASS_COLOR["NNMH"]))

    ax.text(7, 5.5, "Discovered circuit for `x=5; z=9; a=z; a=?`  →  answer: 9",
            ha="center", fontsize=12, fontweight="bold")
    ax.text(7, 5.0,
            "Same 3-stage shape as Wang et al. 2022 IOI circuit: early duplicate detection → late name-mover output",
            ha="center", fontsize=9, color="gray")

    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "binding_compare/circuit_diagram.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  → {out}")


if __name__ == "__main__":
    print("Building polished figures...")
    heatmap_classes()
    universality_summary()
    circuit_diagram()
    print("[done]")
