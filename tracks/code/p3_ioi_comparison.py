"""P3 IOI comparison with Wang et al. 2022.

Computes overlap between:
1. Wang et al. 26 heads from IOI circuit analysis
2. P2 var binding top heads from code analysis

Cross-track circuit universality investigation.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.config import RESULTS_DIR


def load_wang_2022_heads() -> Set[Tuple[int, int]]:
    """Wang et al. 2022 IOI circuit heads.

    From: Wang et al. (2022) "Interpretability at Scale:
    Indirect Object Identification in GPT-2"

    Returns:
        Set of (layer, head) tuples
    """
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
    return wang_heads


def load_p2_var_binding_results(run_dir: str = "run_20260524_195653") -> List[Tuple[Tuple[int, int], float]]:
    """Load P2 var binding results.

    Args:
        run_dir: Results directory name

    Returns:
        List of ((layer, head), score) tuples, sorted by score
    """
    results_path = Path(RESULTS_DIR) / "code" / run_dir / "top_heads.json"

    with open(results_path) as f:
        data = json.load(f)

    # Convert to list of (layer, head), score tuples
    results = [((layer, head), score) for layer, head, score in data]
    results.sort(key=lambda x: x[1], reverse=True)

    return results


def compute_overlap(wang_heads: Set[Tuple[int, int]],
                   p2_results: List[Tuple[Tuple[int, int], float]],
                   top_k: int = 20) -> Dict[str, any]:
    """Compute overlap between Wang et al. heads and P2 results.

    Args:
        wang_heads: Set of Wang et al. (layer, head) tuples
        p2_results: P2 results as ((layer, head), score) list
        top_k: Number of top P2 heads to consider

    Returns:
        Dictionary with overlap statistics
    """
    # Get top K P2 heads
    p2_top_k = set(head for head, _ in p2_results[:top_k])

    # Compute overlap
    overlap = wang_heads.intersection(p2_top_k)
    p2_only = p2_top_k - wang_heads
    wang_only = wang_heads - p2_top_k

    # Get scores for overlapping heads
    overlap_scores = [(head, dict(p2_results)[head]) for head in overlap]
    overlap_scores.sort(key=lambda x: x[1], reverse=True)

    return {
        "total_wang_heads": len(wang_heads),
        "p2_top_k": top_k,
        "overlap_count": len(overlap),
        "overlap_percentage": len(overlap) / top_k * 100,
        "overlap_heads": overlap,
        "overlap_with_scores": overlap_scores,
        "p2_only": sorted(list(p2_only)),
        "wang_only": sorted(list(wang_only)),
        "p2_top_k_heads": sorted(list(p2_top_k))
    }


def print_comparison_results(results: Dict[str, any], p2_results: List[Tuple[Tuple[int, int], float]], wang_heads: Set[Tuple[int, int]]):
    """Print formatted comparison results."""

    print("=" * 80)
    print("P3 IOI vs P2 Var Binding Cross-Track Comparison")
    print("=" * 80)

    print(f"\n📊 OVERVIEW:")
    print(f"  Wang et al. 2022 total heads: {results['total_wang_heads']}")
    print(f"  P2 var binding Top-{results['p2_top_k']} heads")
    print(f"  Overlap: {results['overlap_count']}/{results['p2_top_k']} ({results['overlap_percentage']:.1f}%)")

    print(f"\n🎯 OVERLAPPING HEADS ({len(results['overlap_heads'])}):")
    if results['overlap_heads']:
        for head, score in results['overlap_with_scores']:
            layer_idx, head_idx = head
            p2_rank = next(i for i, (h, _) in enumerate(p2_results) if h == head) + 1
            print(f"  L{layer_idx}H{head_idx}: P2 rank={p2_rank}, score={score:.3f}")
    else:
        print("  ❌ No overlapping heads found!")

    print(f"\n🔵 P2-VAR-BINDING ONLY TOP HEADS ({len(results['p2_only'])}):")
    for head in results['p2_only']:
        layer_idx, head_idx = head
        score = dict(p2_results)[head]
        p2_rank = next(i for i, (h, _) in enumerate(p2_results) if h == head) + 1
        print(f"  L{layer_idx}H{head_idx}: rank={p2_rank}, score={score:.3f}")

    print(f"\n🔴 WANG ET AL. ONLY HEADS (first 10 of {len(results['wang_only'])}):")
    for i, head in enumerate(sorted(results['wang_only'])[:10]):
        layer_idx, head_idx = head
        print(f"  L{layer_idx}H{head_idx}")
    if len(results['wang_only']) > 10:
        print(f"  ... and {len(results['wang_only']) - 10} more")

    # Layer distribution analysis
    print(f"\n📈 LAYER DISTRIBUTION ANALYSIS:")

    p2_layers = [h[0] for h in results['p2_top_k_heads']]
    wang_layers = [h[0] for h in wang_heads]
    overlap_layers = [h[0] for h in results['overlap_heads']]

    print(f"  P2 var binding layers: {sorted(set(p2_layers))}")
    print(f"  Wang et al. layers: {sorted(set(wang_layers))}")
    print(f"  Overlap layers: {sorted(set(overlap_layers))}")


def main():
    """Main analysis function."""

    # Load data
    wang_heads = load_wang_2022_heads()
    p2_results = load_p2_var_binding_results()

    print(f"✅ Loaded Wang et al. 2022: {len(wang_heads)} heads")
    print(f"✅ Loaded P2 var binding: {len(p2_results)} heads")

    # Compute overlap for different top-k values
    for top_k in [10, 20, 26]:
        print(f"\n{'─' * 80}")
        print(f"Top-{top_k} Comparison")
        print(f"{'─' * 80}")

        results = compute_overlap(wang_heads, p2_results, top_k=top_k)
        print_comparison_results(results, p2_results, wang_heads)

    print(f"\n{'=' * 80}")
    print("Analysis complete!")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()