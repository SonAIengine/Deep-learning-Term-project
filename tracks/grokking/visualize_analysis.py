"""Visualize grokking circuit analysis results.

Usage:
    python -m tracks.grokking.visualize_analysis
    python -m tracks.grokking.visualize_analysis --results_dir results/grokking/analysis
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_fourier_scores(results_dir: Path, output_path: Path = None) -> None:
    """Plot Fourier frequency scores."""
    data = np.load(results_dir / "fourier_scores.npz")
    frequencies = data["frequencies"]
    scores = data["scores"]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(frequencies, scores, width=0.8)
    ax.set_xlabel("Fourier Frequency")
    ax.set_ylabel("Correlation Score")
    ax.set_title("Fourier Basis Analysis - Embedding Frequency Representation")
    ax.grid(axis="y", alpha=0.3)

    if output_path is None:
        output_path = results_dir / "fourier_scores.png"
    else:
        output_path = Path(output_path)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_attention_patterns(results_dir: Path, output_path: Path = None) -> None:
    """Plot attention patterns for all heads."""
    data = np.load(results_dir / "attention_patterns.npz")

    n_heads = len([k for k in data.files if k.startswith("layer")])
    fig, axes = plt.subplots(1, n_heads, figsize=(4 * n_heads, 4))

    if n_heads == 1:
        axes = [axes]

    for i, key in enumerate(sorted([k for k in data.files if k.startswith("layer")])):
        pattern = data[key]
        ax = axes[i]

        im = ax.imshow(pattern, cmap="Blues", aspect="auto", vmin=0, vmax=1)
        ax.set_title(key.replace("_", " ").title())
        ax.set_xlabel("Key Position")
        ax.set_ylabel("Query Position")
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(["a", "b", "="])
        ax.set_yticklabels(["a", "b", "="])

        # Add text annotations
        for row in range(3):
            for col in range(3):
                text = ax.text(col, row, f"{pattern[row, col]:.2f}",
                              ha="center", va="center", color="red", fontsize=8)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("Attention Patterns by Head", fontsize=14)
    plt.tight_layout()

    if output_path is None:
        output_path = results_dir / "attention_patterns.png"
    else:
        output_path = Path(output_path)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_head_ablation(results_dir: Path, output_path: Path = None) -> None:
    """Plot head ablation results."""
    data = np.load(results_dir / "head_ablation.npz")
    baseline = float(data["baseline"])
    heads = data["heads"]
    accuracies = data["accuracies"]

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(heads))
    width = 0.6

    bars = ax.bar(x, accuracies, width, label="Ablated Accuracy")
    ax.axhline(y=baseline, color="r", linestyle="--", label=f"Baseline ({baseline:.2f})")

    # Color bars by impact
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        impact = baseline - acc
        if impact > 0.5:
            bar.set_color("crimson")
        elif impact > 0.2:
            bar.set_color("orange")
        else:
            bar.set_color("steelblue")

    ax.set_xlabel("Head")
    ax.set_ylabel("Accuracy")
    ax.set_title("Head Ablation Study - Performance Impact")
    ax.set_xticks(x)
    ax.set_xticklabels([h.replace("_", " ").title() for h in heads], rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.05)

    if output_path is None:
        output_path = results_dir / "head_ablation.png"
    else:
        output_path = Path(output_path)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize grokking analysis results")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/grokking/analysis",
        help="Directory containing analysis results",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save plots (default: same as results_dir)",
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    output_dir = Path(args.output_dir) if args.output_dir else results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating visualizations...")
    plot_fourier_scores(results_dir, output_dir / "fourier_scores.png")
    plot_attention_patterns(results_dir, output_dir / "attention_patterns.png")
    plot_head_ablation(results_dir, output_dir / "head_ablation.png")
    print(f"\nAll plots saved to: {output_dir}/")


if __name__ == "__main__":
    main()
