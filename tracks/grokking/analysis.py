"""Circuit analysis for grokking model.

Key analyses:
1. Fourier basis verification — modular arithmetic should use Fourier frequencies
2. Attention pattern analysis — which positions do heads attend to?
3. Head ablation — performance impact when each head is removed

Reference: Nanda et al. (2023) "Progress Measures" — modular arithmetic
is solved using Fourier basis (frequencies 0, 1, ..., p-1).
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from transformer_lens import HookedTransformer
from tqdm import tqdm

from shared.config import DEVICE, MODULAR_P
from tracks.grokking import config as C


@dataclass
class AnalysisResult:
    """Container for analysis results."""

    fourier_scores: dict  # frequency -> correlation score
    attention_patterns: dict  # layer_head -> (n_positions, n_positions)
    head_ablation: dict  # layer_head -> accuracy after ablation
    baseline_acc: float  # accuracy with no ablation


def load_model_from_checkpoint(ckpt_path: str | Path) -> HookedTransformer:
    """Load a HookedTransformer from checkpoint.

    Args:
        ckpt_path: Path to .pt checkpoint file (contains 'epoch' and 'model' keys)

    Returns:
        Loaded HookedTransformer model
    """
    from tracks.grokking.model import build_model

    model = build_model()
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def compute_fourier_basis(model: HookedTransformer) -> dict:
    """Analyze if model uses Fourier frequencies to solve modular arithmetic.

    Nanda et al. (2023) shows that modular addition is solved using
    Fourier basis: each neuron/representation encodes a frequency component.

    Method: Project residual stream embeddings onto Fourier basis vectors.
    For modular arithmetic mod p, the Fourier basis is:
        e_k(x) = exp(2πi * k * x / p) for k = 0, 1, ..., p-1

    Returns:
        Dictionary with frequency keys (0, 1, ..., p-1) and correlation scores
        measuring how much each frequency is represented in the embeddings.
    """
    p = MODULAR_P

    # Get token embeddings for all numbers 0..p-1 (token 0 is padding/token)
    # In our setup: token i = number i for i in 0..p-1, token p is '='
    embed = model.embed.W_E.detach()  # [d_vocab, d_model]
    num_embeds = embed[:p]  # [p, d_model] — embeddings for numbers only

    # Compute Fourier basis vectors for each frequency k
    # For real embeddings, we use cos(2π*k*x/p) and sin(2π*k*x/p)
    fourier_scores = {}

    for k in range(p):
        # Fourier frequency k component: exp(2πi * k * x / p)
        x = torch.arange(p, dtype=torch.float32, device=DEVICE)
        phase = 2 * np.pi * k * x / p

        # Real and imaginary parts as basis vectors
        cos_basis = torch.cos(phase)  # [p]
        sin_basis = torch.sin(phase)  # [p]

        # Project each embedding dimension onto Fourier basis
        # For each dimension d, compute correlation with cos/sin patterns
        correlations = []
        for d in range(num_embeds.shape[1]):
            dim_values = num_embeds[:, d]  # [p]

            # Normalize for correlation
            dim_norm = (dim_values - dim_values.mean()) / (dim_values.std() + 1e-8)
            cos_norm = (cos_basis - cos_basis.mean()) / (cos_basis.std() + 1e-8)
            sin_norm = (sin_basis - sin_basis.mean()) / (sin_basis.std() + 1e-8)

            cos_corr = (dim_norm * cos_norm).mean().item()
            sin_corr = (dim_norm * sin_norm).mean().item()

            # Combined magnitude (R²-like)
            mag = np.sqrt(cos_corr**2 + sin_corr**2)
            correlations.append(mag)

        # Max correlation across all dimensions
        fourier_scores[k] = max(correlations) if correlations else 0.0

    return fourier_scores


def analyze_attention_patterns(
    model: HookedTransformer,
    n_samples: int = 100,
) -> dict:
    """Extract and analyze attention patterns.

    For modular addition (a + b = c mod p), the input format is:
    [a_token, b_token, eq_token, c_token]

    Key question: Which positions do each attention head attend to?

    Returns:
        Dictionary mapping (layer, head) -> attention pattern matrix [n_ctx, n_ctx]
    """
    p = MODULAR_P
    ctx_len = C.N_CTX

    attention_patterns = {}

    # Generate random samples
    torch.manual_seed(0)
    samples_a = torch.randint(0, p, (n_samples,))
    samples_b = torch.randint(0, p, (n_samples,))

    # Run model and cache attention patterns
    # Input format: [a, b, =] where a, b are numbers and = is the equality token
    x = torch.stack([samples_a, samples_b, torch.full_like(samples_a, p)], dim=1)
    _, cache = model.run_with_cache(x)

    # Extract attention patterns for each head
    # cache["blocks.0.attn.hook_pattern"] has shape [batch, n_heads, n_ctx, n_ctx]
    attn = cache["blocks.0.attn.hook_pattern"]  # [n_samples, n_heads, n_ctx, n_ctx]

    # Average across samples
    avg_attn = attn.mean(dim=0).detach().cpu()  # [n_heads, n_ctx, n_ctx]

    for head in range(C.N_HEADS):
        attention_patterns[(0, head)] = avg_attn[head].numpy()

    return attention_patterns


def ablate_head(
    model: HookedTransformer,
    layer: int,
    head: int,
    x: torch.Tensor,
    y: torch.Tensor,
) -> float:
    """Ablate a specific attention head and compute accuracy.

    Ablation = zero out the output of the attention head.

    Args:
        model: The model
        layer: Layer index (0-based)
        head: Head index within the layer
        x: Input tokens [batch, 2] (a, b values)
        y: Target answers [batch]

    Returns:
        Accuracy after ablation
    """
    p = MODULAR_P

    # Hook function to zero out specific head output
    def ablate_hook(z, hook):
        # z has shape [batch, n_ctx, n_heads, d_head]
        z[:, :, head, :] = 0
        return z

    # Run with ablation hook
    with model.hooks(fwd_hooks=[(f"blocks.{layer}.attn.hook_z", ablate_hook)]):
        logits = model(x)

    # Compute accuracy
    preds = logits[:, -1].argmax(dim=-1)  # Predictions at last position
    acc = (preds == y).float().mean().item()

    return acc


def run_head_ablation(
    model: HookedTransformer,
    n_test_samples: int = 500,
) -> dict:
    """Run head ablation study.

    For each head, measure accuracy when that head is disabled.

    Returns:
        Dictionary with baseline accuracy and per-head ablation results
    """
    p = MODULAR_P

    # Generate test data
    torch.manual_seed(42)
    a = torch.randint(0, p, (n_test_samples,), device=DEVICE)
    b = torch.randint(0, p, (n_test_samples,), device=DEVICE)
    c = (a + b) % p

    # Format input: [a_token, b_token, eq_token]
    x = torch.stack([a, b, torch.full_like(a, p)], dim=1)
    y = c

    # Baseline accuracy (no ablation)
    with torch.no_grad():
        logits = model(x)
    baseline_preds = logits[:, -1].argmax(dim=-1)
    baseline_acc = (baseline_preds == y).float().mean().item()

    # Ablate each head
    head_results = {}

    for head in range(C.N_HEADS):
        acc = ablate_head(model, layer=0, head=head, x=x, y=y)
        head_results[(0, head)] = acc

    return {
        "baseline": baseline_acc,
        "heads": head_results,
    }


def full_analysis(
    model: HookedTransformer,
    n_attn_samples: int = 100,
    n_ablate_samples: int = 500,
) -> AnalysisResult:
    """Run all circuit analyses.

    Args:
        model: Trained HookedTransformer
        n_attn_samples: Number of samples for attention analysis
        n_ablate_samples: Number of samples for ablation study

    Returns:
        AnalysisResult containing all analysis outputs
    """
    print("Computing Fourier basis analysis...")
    fourier_scores = compute_fourier_basis(model)

    print("Analyzing attention patterns...")
    attention_patterns = analyze_attention_patterns(model, n_samples=n_attn_samples)

    print("Running head ablation study...")
    ablation = run_head_ablation(model, n_test_samples=n_ablate_samples)

    return AnalysisResult(
        fourier_scores=fourier_scores,
        attention_patterns=attention_patterns,
        head_ablation=ablation["heads"],
        baseline_acc=ablation["baseline"],
    )


def save_results(results: AnalysisResult, output_dir: str | Path) -> None:
    """Save analysis results to disk.

    Args:
        results: AnalysisResult from full_analysis
        output_dir: Directory to save results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save Fourier scores
    np.savez(
        output_dir / "fourier_scores.npz",
        frequencies=list(results.fourier_scores.keys()),
        scores=list(results.fourier_scores.values()),
    )

    # Save attention patterns
    attn_dict = {
        f"layer{layer}_head{head}": pattern
        for (layer, head), pattern in results.attention_patterns.items()
    }
    np.savez(output_dir / "attention_patterns.npz", **attn_dict)

    # Save ablation results
    np.savez(
        output_dir / "head_ablation.npz",
        baseline=results.baseline_acc,
        heads=[f"layer{l}_head{h}" for l, h in results.head_ablation.keys()],
        accuracies=list(results.head_ablation.values()),
    )

    # Save text report
    report_path = output_dir / "report.txt"
    with open(report_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("GROKKING CIRCUIT ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")

        # Fourier analysis
        f.write("1. FOURIER BASIS ANALYSIS\n")
        f.write("-" * 40 + "\n")
        f.write("Top 10 frequencies by correlation:\n")
        sorted_freq = sorted(
            results.fourier_scores.items(), key=lambda x: x[1], reverse=True
        )
        for freq, score in sorted_freq[:10]:
            f.write(f"  Frequency {freq:3d}: {score:.4f}\n")
        f.write("\n")

        # Attention patterns
        f.write("2. ATTENTION PATTERN ANALYSIS\n")
        f.write("-" * 40 + "\n")
        for (layer, head), pattern in results.attention_patterns.items():
            f.write(f"\nLayer {layer}, Head {head}:\n")
            # Format: show attention from position to position
            # Positions: 0=a, 1=b, 2==
            f.write(f"  Attention to a (pos 0):   {pattern[:, 0]}\n")
            f.write(f"  Attention to b (pos 1):   {pattern[:, 1]}\n")
            f.write(f"  Attention to = (pos 2):  {pattern[:, 2]}\n")

        # Head ablation
        f.write("\n3. HEAD ABLATION STUDY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Baseline accuracy (no ablation): {results.baseline_acc:.4f}\n\n")
        f.write("Accuracy after ablating each head:\n")
        for (layer, head), acc in sorted(results.head_ablation.items()):
            impact = results.baseline_acc - acc
            importance = "CRITICAL" if impact > 0.5 else "Important" if impact > 0.1 else "Minor"
            f.write(f"  Layer {layer}, Head {head}: {acc:.4f} (-{impact:.4f}) [{importance}]\n")

    print(f"Results saved to {output_dir}")


def load_results(results_dir: str | Path) -> AnalysisResult:
    """Load analysis results from disk.

    Args:
        results_dir: Directory containing saved results

    Returns:
        AnalysisResult with loaded data
    """
    results_dir = Path(results_dir)

    # Load Fourier scores
    fourier_data = np.load(results_dir / "fourier_scores.npz")
    fourier_scores = dict(zip(fourier_data["frequencies"], fourier_data["scores"]))

    # Load attention patterns
    attn_data = np.load(results_dir / "attention_patterns.npz")
    attention_patterns = {}
    for key in attn_data.files:
        # Parse "layer0_head0" format
        parts = key.split("_")
        layer = int(parts[0][5:])
        head = int(parts[1][4:])
        attention_patterns[(layer, head)] = attn_data[key]

    # Load ablation results
    ablate_data = np.load(results_dir / "head_ablation.npz")
    baseline = float(ablate_data["baseline"])
    heads = ablate_data["heads"]
    accuracies = ablate_data["accuracies"]
    head_ablation = {}
    for head_str, acc in zip(heads, accuracies):
        parts = head_str.split("_")
        layer = int(parts[0][5:])
        head = int(parts[1][4:])
        head_ablation[(layer, head)] = float(acc)

    return AnalysisResult(
        fourier_scores=fourier_scores,
        attention_patterns=attention_patterns,
        head_ablation=head_ablation,
        baseline_acc=baseline,
    )
