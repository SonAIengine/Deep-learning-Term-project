"""P3 IOI comparison visualization.

Creates visual comparisons between Wang et al. 2022 IOI circuit
and P2 var binding results.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.config import RESULTS_DIR


def load_data():
    """Load P2 var binding results and Wang et al. heads."""

    # Wang et al. 2022 IOI circuit heads
    wang_heads = {
        # Name Mover heads (primary)
        (9, 9), (10, 7), (11, 10),
        # Name Mover heads (secondary)
        (9, 6), (10, 0), (10, 6), (11, 2),
        # Backup name movers
        (9, 8), (10, 10), (10, 2), (11, 13), (11, 9),
        # S-Inhibition heads
        (9, 7), (11, 5), (11, 6), (11, 3),
        # Induction heads / Duplicate token
        (9, 5), (10, 1), (11, 4), (10, 8), (11, 1),
        # Previous token head
        (8, 9), (11, 11),
        # Negative name movers
        (10, 11), (8, 11), (9, 4)
    }

    # P2 var binding results
    results_path = Path(RESULTS_DIR) / "code" / "run_20260524_195653" / "top_heads.json"

    with open(results_path) as f:
        data = json.load(f)

    p2_results = [((layer, head), score) for layer, head, score in data]
    p2_results.sort(key=lambda x: x[1], reverse=True)

    return wang_heads, p2_results


def create_layer_distribution_comparison(wang_heads: Set[Tuple[int, int]],
                                        p2_results: List[Tuple[Tuple[int, int], float]]):
    """Compare layer distributions between Wang et al. and P2."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Wang et al. layer distribution
    wang_layers = [layer for layer, _ in wang_heads]
    wang_layer_counts = {}
    for layer in wang_layers:
        wang_layer_counts[layer] = wang_layer_counts.get(layer, 0) + 1

    # P2 layer distribution (Top-20)
    p2_top_20 = [head for head, _ in p2_results[:20]]
    p2_layers = [layer for layer, _ in p2_top_20]
    p2_layer_counts = {}
    for layer in p2_layers:
        p2_layer_counts[layer] = p2_layer_counts.get(layer, 0) + 1

    # Plot Wang et al. distribution
    layers_wang = sorted(wang_layer_counts.keys())
    counts_wang = [wang_layer_counts[l] for l in layers_wang]

    ax1.bar(layers_wang, counts_wang, color='coral', alpha=0.7, label='Wang et al. (2022)')
    ax1.set_xlabel('Layer', fontsize=12)
    ax1.set_ylabel('Number of Important Heads', fontsize=12)
    ax1.set_title('IOI Circuit Layer Distribution\n(Wang et al. 2022)', fontsize=13, fontweight='bold')
    ax1.set_xticks(range(8, 12))
    ax1.grid(axis='y', alpha=0.3)
    ax1.legend()

    # Plot P2 var binding distribution
    layers_p2 = sorted(p2_layer_counts.keys())
    counts_p2 = [p2_layer_counts[l] for l in layers_p2]

    ax2.bar(layers_p2, counts_p2, color='steelblue', alpha=0.7, label='P2 Var Binding')
    ax2.set_xlabel('Layer', fontsize=12)
    ax2.set_ylabel('Number of Important Heads', fontsize=12)
    ax2.set_title('Code Var Binding Layer Distribution\n(Top-20 Heads)', fontsize=13, fontweight='bold')
    ax2.set_xticks(range(1, 11))
    ax2.grid(axis='y', alpha=0.3)
    ax2.legend()

    plt.tight_layout()

    # Save figure
    output_path = Path(RESULTS_DIR) / "code" / "run_20260524_195653" / "visualizations" / "p3_layer_comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")

    plt.close()


def create_overlap_analysis(wang_heads: Set[Tuple[int, int]],
                           p2_results: List[Tuple[Tuple[int, int], float]]):
    """Create overlap analysis visualization."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Analyze overlap for different top-k values
    top_k_values = [10, 20, 26]
    overlap_percentages = []

    for top_k in top_k_values:
        p2_top_k = set(head for head, _ in p2_results[:top_k])
        overlap = wang_heads.intersection(p2_top_k)
        overlap_percentages.append(len(overlap) / top_k * 100)

    # Plot 1: Overlap percentage by top-k
    ax1 = axes[0, 0]
    bars = ax1.bar(range(len(top_k_values)), overlap_percentages,
                   color=['#e74c3c', '#f39c12', '#3498db'], alpha=0.7)
    ax1.set_xlabel('Top-K Heads', fontsize=12)
    ax1.set_ylabel('Overlap Percentage (%)', fontsize=12)
    ax1.set_title('Cross-Track Overlap Analysis', fontsize=13, fontweight='bold')
    ax1.set_xticks(range(len(top_k_values)))
    ax1.set_xticklabels([f'Top-{k}' for k in top_k_values])
    ax1.grid(axis='y', alpha=0.3)

    # Add percentage labels on bars
    for i, (bar, pct) in enumerate(zip(bars, overlap_percentages)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Plot 2: Venn diagram-style representation (Top-10)
    ax2 = axes[0, 1]
    p2_top_10 = set(head for head, _ in p2_results[:10])
    overlap_10 = wang_heads.intersection(p2_top_10)

    categories = ['Wang et al.\nOnly', 'Overlap', 'P2 Var Binding\nOnly']
    counts = [len(wang_heads) - len(overlap_10), len(overlap_10), len(p2_top_10) - len(overlap_10)]
    colors = ['#e74c3c', '#9b59b6', '#3498db']

    ax2.barh(categories, counts, color=colors, alpha=0.7)
    ax2.set_xlabel('Number of Heads', fontsize=12)
    ax2.set_title('Top-10 Heads Distribution', fontsize=13, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)

    # Plot 3: Layer heatmap for P2 Top-20
    ax3 = axes[1, 0]
    p2_top_20 = [head for head, _ in p2_results[:20]]

    # Create layer x head grid
    layer_grid = np.zeros((12, 13))  # 12 layers (0-11), 13 heads (0-12)

    for (layer, head), score in p2_results[:20]:
        layer_grid[layer, head] = score

    # Mask cells with no data
    masked_grid = np.ma.masked_where(layer_grid == 0, layer_grid)

    im = ax3.imshow(masked_grid, cmap='YlOrRd', aspect='auto', vmin=1.15, vmax=1.31)

    # Mark Wang et al. overlap heads
    for (layer, head) in wang_heads:
        if layer < 12 and head < 13:
            ax3.add_patch(plt.Rectangle((head - 0.5, layer - 0.5), 1, 1,
                                      fill=False, edgecolor='blue', linewidth=2))

    ax3.set_xlabel('Head Index', fontsize=12)
    ax3.set_ylabel('Layer', fontsize=12)
    ax3.set_title('P2 Var Binding Top-20 Heads\n(Blue box = Wang et al. overlap)', fontsize=13, fontweight='bold')
    ax3.set_xticks(range(13))
    ax3.set_yticks(range(12))

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label('Logit Diff Recovery Score', fontsize=11)

    # Plot 4: Head score distribution comparison
    ax4 = axes[1, 1]

    # Get scores for overlapping vs non-overlapping heads
    p2_top_20_heads = set(head for head, _ in p2_results[:20])
    overlap_20 = wang_heads.intersection(p2_top_20_heads)

    overlap_scores = [score for head, score in p2_results[:20] if head in overlap_20]
    non_overlap_scores = [score for head, score in p2_results[:20] if head not in overlap_20]

    ax4.hist(non_overlap_scores, bins=10, color='#3498db', alpha=0.6,
            label=f'P2 Only (n={len(non_overlap_scores)})', edgecolor='black')
    ax4.hist(overlap_scores, bins=5, color='#e74c3c', alpha=0.8,
            label=f'Overlap (n={len(overlap_scores)})', edgecolor='black')

    ax4.set_xlabel('Logit Diff Recovery Score', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.set_title('Score Distribution: Overlap vs P2 Only\n(Top-20 Heads)', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_path = Path(RESULTS_DIR) / "code" / "run_20260524_195653" / "visualizations" / "p3_overlap_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")

    plt.close()


def create_circuit_universality_score(wang_heads: Set[Tuple[int, int]],
                                     p2_results: List[Tuple[Tuple[int, int], float]]):
    """Calculate and visualize circuit universality metrics."""

    # Calculate Jaccard similarity for different top-k values
    top_k_values = list(range(5, 51, 5))
    jaccard_similarities = []
    overlap_counts = []

    for top_k in top_k_values:
        p2_top_k = set(head for head, _ in p2_results[:top_k])
        overlap = wang_heads.intersection(p2_top_k)
        union = wang_heads.union(p2_top_k)

        jaccard = len(overlap) / len(union) if union else 0
        jaccard_similarities.append(jaccard * 100)  # Convert to percentage
        overlap_counts.append(len(overlap))

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Jaccard similarity curve
    ax1.plot(top_k_values, jaccard_similarities, marker='o', linewidth=2,
            color='#e74c3c', markersize=8)
    ax1.fill_between(top_k_values, jaccard_similarities, alpha=0.3, color='#e74c3c')
    ax1.set_xlabel('Top-K Heads Considered', fontsize=12)
    ax1.set_ylabel('Jaccard Similarity (%)', fontsize=12)
    ax1.set_title('Circuit Universality: Jaccard Similarity', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, max(jaccard_similarities) * 1.2])

    # Add annotation for maximum similarity
    max_idx = np.argmax(jaccard_similarities)
    max_sim = jaccard_similarities[max_idx]
    max_k = top_k_values[max_idx]
    ax1.annotate(f'Max: {max_sim:.1f}% at Top-{max_k}',
                xy=(max_k, max_sim), xytext=(max_k + 5, max_sim + 2),
                fontsize=10, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='black'))

    # Plot 2: Absolute overlap count
    ax2.bar(top_k_values, overlap_counts, color='#3498db', alpha=0.7)
    ax2.set_xlabel('Top-K Heads Considered', fontsize=12)
    ax2.set_ylabel('Absolute Overlap Count', fontsize=12)
    ax2.set_title('Absolute Overlap: Wang et al. ∩ P2', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # Add count labels on bars
    for i, (k, count) in enumerate(zip(top_k_values, overlap_counts)):
        if count > 0:
            ax2.text(k, count + 0.1, str(count), ha='center', fontsize=9)

    plt.tight_layout()

    # Save figure
    output_path = Path(RESULTS_DIR) / "code" / "run_20260524_195653" / "visualizations" / "p3_circuit_universality.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")

    plt.close()


def main():
    """Main visualization function."""

    print("🎨 Creating P3 IOI comparison visualizations...")

    # Load data
    wang_heads, p2_results = load_data()
    print(f"✅ Loaded Wang et al. 2022: {len(wang_heads)} heads")
    print(f"✅ Loaded P2 var binding: {len(p2_results)} heads")

    # Create visualizations
    create_layer_distribution_comparison(wang_heads, p2_results)
    create_overlap_analysis(wang_heads, p2_results)
    create_circuit_universality_score(wang_heads, p2_results)

    print("✅ All visualizations created successfully!")


if __name__ == "__main__":
    main()