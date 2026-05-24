"""Run activation patching analysis on P2 var binding Tier 1.

First N cf pairs for circuit identification.
"""
import os
import json
import torch
from datetime import datetime

from tracks.code.model import load_gpt2_small
from tracks.code.data import load_tier1, get_cf_pairs, sample_pairs
from tracks.code.analysis import analyze_first_n_pairs, summarize_patch_results
from tracks.code.config import RESULTS_DIR


def main():
    print("Loading GPT-2 small...")
    model = load_gpt2_small()
    print(f"Model loaded: {model.cfg.n_layers} layers, {model.cfg.n_heads} heads")

    print("Loading Tier 1 data...")
    records = load_tier1()
    pairs = get_cf_pairs(records)
    print(f"Loaded {len(pairs)} cf pairs")

    # Sample first N pairs
    n_pairs = 5
    sample = sample_pairs(pairs, n_pairs)
    print(f"Analyzing first {n_pairs} cf pairs...")

    # Run analysis
    print("Running activation patching...")
    results = analyze_first_n_pairs(model, sample, n_pairs)

    # Print cache shapes
    print("\n=== Cache Shapes (first clean prompt) ===")
    if results["cache_shapes"]:
        for name, shape in list(results["cache_shapes"].items())[:10]:
            print(f"  {name}: {shape}")

    # Print baseline logit diffs
    print("\n=== Baseline Logit Diffs ===")
    for cf_id, baselines in results["baseline_logit_diffs"].items():
        print(f"  {cf_id}: clean={baselines['clean']:.4f}, corrupt={baselines['corrupt']:.4f}")

    # Summarize patch results
    print("\n=== Summarizing Patch Results ===")
    summary = summarize_patch_results(results)

    print("\n=== Top 10 Heads by Recovery ===")
    for layer, head, avg in summary["top_heads"][:10]:
        print(f"  L{layer}H{head}: {avg:.4f}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(RESULTS_DIR, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Save JSON results
    results_path = os.path.join(run_dir, "patching_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Save summary
    summary_path = os.path.join(run_dir, "summary.json")
    summary_data = {
        "top_heads": [(l, h, float(r)) for l, h, r in summary["top_heads"]],
    }
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)

    # Save heatmap
    heatmap_path = os.path.join(run_dir, "head_heatmap.pt")
    torch.save(summary["head_heatmap"], heatmap_path)

    print(f"\nResults saved to {run_dir}")
    print(f"  - patching_results.json")
    print(f"  - summary.json")
    print(f"  - head_heatmap.pt")


if __name__ == "__main__":
    main()
