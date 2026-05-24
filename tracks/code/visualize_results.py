"""Visualization and comparison for var binding circuit analysis."""
import json
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np


def load_summary(results_dir: str) -> Dict[str, Any]:
    """Load analysis summary from results directory."""
    results_path = Path(results_dir)

    # Try .pt file first
    summary_file = results_path / "summary.pt"
    if summary_file.exists():
        return torch.load(summary_file)

    # Fall back to .json file
    summary_file = results_path / "summary.json"
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            data = json.load(f)

        # Convert JSON format to match torch format
        # JSON has [[layer, head, recovery], ...] format
        top_heads = [(l, h, r) for l, h, r in data["top_heads"]]

        # Reconstruct heatmap
        n_layers = 12
        n_heads = 12
        heatmap = torch.zeros(n_layers, n_heads)
        for l, h, r in top_heads:
            if l < n_layers and h < n_heads:
                heatmap[l, h] = r

        return {
            "top_heads": top_heads,
            "head_heatmap": heatmap,
        }

    raise FileNotFoundError(f"Summary not found in {results_dir}")


def compare_top_heads(old_summary: Dict, new_summary: Dict, top_n: int = 20) -> None:
    """Compare top heads between old and new analyses."""
    old_top = old_summary["top_heads"][:top_n]
    new_top = new_summary["top_heads"][:top_n]

    print(f"\n=== Top {top_n} Heads Comparison ===")
    print(f"{'Rank':<6} {'Old (5 pairs)':<20} {'New (500 pairs)':<20} {'Change':<10}")
    print("-" * 60)

    for i in range(top_n):
        old_str = f"L{old_top[i][0]}H{old_top[i][1]} ({old_top[i][2]:.3f})"
        new_str = f"L{new_top[i][0]}H{new_top[i][1]} ({new_top[i][2]:.3f})"

        # Check if head is in both top lists
        old_head = (old_top[i][0], old_top[i][1])
        new_head = (new_top[i][0], new_top[i][1])

        if old_head in [(h[0], h[1]) for h in new_top[:top_n]]:
            change = "✓ Stable"
        else:
            change = "✗ Changed"

        print(f"{i+1:<6} {old_str:<20} {new_str:<20} {change:<10}")


def plot_heatmap_comparison(old_summary: Dict, new_summary: Dict, save_path: str = None) -> None:
    """Plot heatmap comparison between old and new analyses."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Old heatmap (5 pairs)
    old_heatmap = old_summary["head_heatmap"].numpy()
    sns.heatmap(old_heatmap, annot=False, cmap="viridis", ax=axes[0], cbar_kws={"label": "Recovery"})
    axes[0].set_title("Var Binding Circuit (5 pairs)")
    axes[0].set_xlabel("Head")
    axes[0].set_ylabel("Layer")

    # New heatmap (500 pairs)
    new_heatmap = new_summary["head_heatmap"].numpy()
    sns.heatmap(new_heatmap, annot=False, cmap="viridis", ax=axes[1], cbar_kws={"label": "Recovery"})
    axes[1].set_title("Var Binding Circuit (500 pairs)")
    axes[1].set_xlabel("Head")
    axes[1].set_ylabel("Layer")

    # Difference
    diff = new_heatmap - old_heatmap
    sns.heatmap(diff, annot=False, cmap="RdBu_r", center=0, ax=axes[2],
                cbar_kws={"label": "Recovery Difference"})
    axes[2].set_title("Difference (500 - 5 pairs)")
    axes[2].set_xlabel("Head")
    axes[2].set_ylabel("Layer")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved heatmap comparison to {save_path}")

    plt.show()


def plot_recovery_distribution(summary: Dict, save_path: str = None) -> None:
    """Plot distribution of recovery scores across all heads."""
    recoveries = [r for _, _, r in summary["top_heads"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(recoveries, bins=50, edgecolor="black", alpha=0.7)
    axes[0].axvline(np.mean(recoveries), color="red", linestyle="--",
                    label=f"Mean: {np.mean(recoveries):.3f}")
    axes[0].axvline(np.median(recoveries), color="blue", linestyle="--",
                    label=f"Median: {np.median(recoveries):.3f}")
    axes[0].set_xlabel("Recovery Score")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Distribution of Recovery Scores")
    axes[0].legend()

    # Box plot by layer
    heatmap = summary["head_heatmap"].numpy()
    layer_means = heatmap.mean(axis=1)

    axes[1].bar(range(len(layer_means)), layer_means)
    axes[1].axhline(np.mean(recoveries), color="red", linestyle="--",
                    label=f"Overall Mean: {np.mean(recoveries):.3f}")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Mean Recovery")
    axes[1].set_title("Mean Recovery by Layer")
    axes[1].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved recovery distribution to {save_path}")

    plt.show()


def print_stability_analysis(old_summary: Dict, new_summary: Dict) -> None:
    """Analyze stability of top heads across sample sizes."""
    print("\n=== Stability Analysis ===")

    # Top 5 heads in each analysis
    old_top5 = set((h[0], h[1]) for h in old_summary["top_heads"][:5])
    new_top5 = set((h[0], h[1]) for h in new_summary["top_heads"][:5])

    # Top 10 heads in each analysis
    old_top10 = set((h[0], h[1]) for h in old_summary["top_heads"][:10])
    new_top10 = set((h[0], h[1]) for h in new_summary["top_heads"][:10])

    # Top 20 heads in each analysis
    old_top20 = set((h[0], h[1]) for h in old_summary["top_heads"][:20])
    new_top20 = set((h[0], h[1]) for h in new_summary["top_heads"][:20])

    print(f"Top 5 overlap: {len(old_top5 & new_top5)}/5 ({100*len(old_top5 & new_top5)/5:.0f}%)")
    print(f"Top 10 overlap: {len(old_top10 & new_top10)}/10 ({100*len(old_top10 & new_top10)/10:.0f}%)")
    print(f"Top 20 overlap: {len(old_top20 & new_top20)}/20 ({100*len(old_top20 & new_top20)/20:.0f}%)")

    # Heads that moved significantly
    print("\n=== Significant Rank Changes (5→20) ===")
    old_dict = {(h[0], h[1]): i for i, h in enumerate(old_summary["top_heads"][:20])}
    new_dict = {(h[0], h[1]): i for i, h in enumerate(new_summary["top_heads"][:20])}

    for head in old_top10 | new_top10:
        old_rank = old_dict.get(head, 99)
        new_rank = new_dict.get(head, 99)
        change = old_rank - new_rank  # Positive = improved

        if abs(change) >= 5:
            direction = "↑" if change > 0 else "↓"
            print(f"L{head[0]}H{head[1]}: {old_rank+1:2d} → {new_rank+1:2d} ({direction}{abs(change)})")


def main():
    """Main analysis workflow."""
    # Paths
    old_results = "results/code/run_20260524_185842"
    new_results = None  # Will be set when new analysis completes

    # Check if new results are available
    import sys
    if len(sys.argv) > 1:
        new_results = sys.argv[1]
    else:
        # Find most recent results
        results_dirs = sorted(Path("results/code").glob("run_*"), key=lambda x: x.name, reverse=True)
        if len(results_dirs) > 1:
            new_results = str(results_dirs[0])
        else:
            print("No new results found. Waiting for analysis to complete...")
            return

    print(f"Comparing:")
    print(f"  Old: {old_results}")
    print(f"  New: {new_results}")

    # Load summaries
    old_summary = load_summary(old_results)
    new_summary = load_summary(new_results)

    # Run comparisons
    compare_top_heads(old_summary, new_summary, top_n=20)
    print_stability_analysis(old_summary, new_summary)

    # Generate visualizations
    output_dir = Path(new_results) / "visualizations"
    output_dir.mkdir(exist_ok=True)

    plot_heatmap_comparison(old_summary, new_summary,
                           save_path=str(output_dir / "heatmap_comparison.png"))
    plot_recovery_distribution(new_summary,
                              save_path=str(output_dir / "recovery_distribution.png"))

    print(f"\nVisualizations saved to {output_dir}")


if __name__ == "__main__":
    main()
