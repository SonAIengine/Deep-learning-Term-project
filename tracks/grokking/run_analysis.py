"""Run circuit analysis on trained grokking model.

Usage:
    python -m tracks.grokking.run_analysis
    python -m tracks.grokking.run_analysis --checkpoint <path>

This script:
1. Loads a trained grokking model checkpoint
2. Runs circuit analysis (Fourier basis, attention patterns, head ablation)
3. Saves results to results/grokking/analysis/
"""
import argparse
from pathlib import Path

import torch

from shared.config import DEVICE
from tracks.grokking.analysis import full_analysis, load_model_from_checkpoint, save_results


def main():
    parser = argparse.ArgumentParser(description="Analyze grokking circuit")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint .pt file (default: use latest from run_20260524_012428)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: results/grokking/analysis/)",
    )
    parser.add_argument(
        "--attn-samples",
        type=int,
        default=100,
        help="Number of samples for attention analysis",
    )
    parser.add_argument(
        "--ablate-samples",
        type=int,
        default=500,
        help="Number of samples for ablation study",
    )

    args = parser.parse_args()

    # Determine checkpoint path
    if args.checkpoint is None:
        # Use latest checkpoint from the grokking run
        run_dir = Path("results/grokking/run_20260524_012428")
        ckpt_dir = run_dir / "checkpoints"

        # Find the latest checkpoint (highest epoch)
        ckpts = sorted(ckpt_dir.glob("ckpt_*.pt"))
        if not ckpts:
            raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")

        ckpt_path = ckpts[-1]  # Last one is highest epoch
        print(f"Using latest checkpoint: {ckpt_path}")
    else:
        ckpt_path = Path(args.checkpoint)

    # Determine output directory
    if args.output is None:
        output_dir = Path("results/grokking/analysis")
    else:
        output_dir = Path(args.output)

    # Load model
    print(f"Loading model from {ckpt_path}...")
    model = load_model_from_checkpoint(ckpt_path)

    # Move to device
    model = model.to(DEVICE)

    # Run analysis
    print(f"Running circuit analysis on {DEVICE}...")
    results = full_analysis(
        model,
        n_attn_samples=args.attn_samples,
        n_ablate_samples=args.ablate_samples,
    )

    # Save results
    print("\nSaving results...")
    save_results(results, output_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)

    print(f"\nBaseline accuracy: {results.baseline_acc:.4f}")

    print("\nTop Fourier frequencies:")
    sorted_freq = sorted(results.fourier_scores.items(), key=lambda x: -x[1])
    for freq, score in sorted_freq[:5]:
        print(f"  Freq {freq:3d}: {score:.4f}")

    print("\nHead ablation impact:")
    for (layer, head), acc in sorted(results.head_ablation.items(), key=lambda x: x[1]):
        impact = results.baseline_acc - acc
        print(f"  L{layer}H{head}: {acc:.4f} (-{impact:.4f})")

    print(f"\nFull results saved to: {output_dir}/")
    print(f"  - fourier_scores.npz")
    print(f"  - attention_patterns.npz")
    print(f"  - head_ablation.npz")
    print(f"  - report.txt")


if __name__ == "__main__":
    main()
