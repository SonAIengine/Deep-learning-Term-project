"""Activation patching analysis for var binding circuit.

GPT-2 small + Tier 1 cf pairs.
Wang et al. 2022 IOI-style logit-diff metric.
"""
import torch
from typing import Dict, List, Tuple, Any, Callable
from jaxtyping import Float
from transformer_lens import HookedTransformer

from tracks.code.model import load_gpt2_small
from tracks.code.data import load_tier1, get_cf_pairs, sample_pairs
from tracks.code.config import DEVICE


def logit_diff_metric(
    logits: Float[torch.Tensor, "batch pos d_vocab"],
    answer_token_ids: torch.Tensor,
    distractor_token_ids: List[int],
) -> Float[torch.Tensor, "batch"]:
    """Compute logit[answer] - mean(logit[distractors]) per sample.

    Wang et al. 2022 IOI standard metric.

    Args:
        logits: [batch, pos, d_vocab] model output
        answer_token_ids: [batch] correct answer token ids
        distractor_token_ids: list of distractor token ids (same for all batch)

    Returns:
        [batch] logit differences
    """
    batch_size = logits.shape[0]
    last_pos_logits = logits[:, -1, :]  # [batch, d_vocab]

    # Get answer logits
    answer_logits = last_pos_logits[torch.arange(batch_size), answer_token_ids]

    # Get distractor logits and average
    distractor_logits = []
    for d_id in distractor_token_ids:
        # d_id: single int, use for all batch
        d_logits = last_pos_logits[:, d_id]  # [batch]
        distractor_logits.append(d_logits)

    # Stack and average: [batch, n_distractors] -> [batch]
    distractor_mean = torch.stack(distractor_logits, dim=-1).mean(dim=-1)

    return answer_logits - distractor_mean


def compute_cache_shape(
    model: HookedTransformer,
    prompts: List[str],
) -> Dict[str, Tuple[int, ...]]:
    """Run model with cache and return shape information.

    Returns dict mapping activation names to their shapes.
    """
    _, cache = model.run_with_cache(prompts)

    shapes = {}
    for name, activation in cache.items():
        shapes[name] = activation.shape

    return shapes


def run_prompt_baseline(
    model: HookedTransformer,
    prompts: List[str],
    answer_token_ids: torch.Tensor,
    distractor_token_ids: List[torch.Tensor],
) -> float:
    """Run model on prompts and return avg logit-diff."""
    logits = model(prompts)
    logit_diffs = logit_diff_metric(logits, answer_token_ids, distractor_token_ids)
    return logit_diffs.mean().item()


def activation_patching_experiment(
    model: HookedTransformer,
    clean_prompt: str,
    corrupt_prompt: str,
    clean_answer_id: int,
    corrupt_answer_id: int,
    clean_distractor_ids: List[int],
    corrupt_distractor_ids: List[int],
    source_pos: int,
) -> Dict[str, Any]:
    """Run activation patching on a single cf pair.

    Patch activations from corrupt run into clean run at source_pos.

    Returns:
        Dict with:
        - clean_logit_diff: baseline clean logit-diff
        - corrupt_logit_diff: baseline corrupt logit-diff
        - patch_results: {component_name: patched_logit_diff}
    """
    # Baseline runs
    clean_logits = model(clean_prompt)
    corrupt_logits = model(corrupt_prompt)

    clean_ld = logit_diff_metric(
        clean_logits,  # Already [1, pos, d_vocab]
        torch.tensor([clean_answer_id]).to(DEVICE),
        clean_distractor_ids,  # list of ints
    ).item()

    corrupt_ld = logit_diff_metric(
        corrupt_logits,  # Already [1, pos, d_vocab]
        torch.tensor([corrupt_answer_id]).to(DEVICE),
        corrupt_distractor_ids,  # list of ints
    ).item()

    # Full cache runs for patching
    _, clean_cache = model.run_with_cache(clean_prompt)
    _, corrupt_cache = model.run_with_cache(corrupt_prompt)

    patch_results = {}

    # Patch each attention head output
    # Use resid_post hook (after MLP) for each layer
    # Patch from source_pos to end (all subsequent positions)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # Get sequence length
    seq_len = clean_cache["blocks.0.hook_resid_post"].shape[1]

    for layer in range(n_layers):
        for head in range(n_heads):
            # Use resid_post hook (after attention + MLP)
            hook_name = f"blocks.{layer}.hook_resid_post"

            current_layer = layer
            current_head = head
            current_source_pos = source_pos
            d_head = model.cfg.d_head
            current_hook_name = hook_name
            current_corrupt_cache = corrupt_cache
            current_seq_len = seq_len

            def make_patch_fn(layer_idx, head_idx, start_pos, end_pos, d_head_val, hook_nm, corr_cache):
                def patch_fn(activation, hook):
                    # Patch specific head's contribution from start_pos to end
                    # activation shape: [batch, pos, d_model]
                    # Head contribution is at [head_idx * d_head_val : (head_idx + 1) * d_head_val]
                    start = head_idx * d_head_val
                    end = (head_idx + 1) * d_head_val
                    activation[:, start_pos:end_pos, start:end] = \
                        corr_cache[hook_nm][:, start_pos:end_pos, start:end]
                    return activation
                return patch_fn

            patch_fn = make_patch_fn(current_layer, current_head,
                                    current_source_pos, current_seq_len,
                                    d_head,
                                    current_hook_name, current_corrupt_cache)

            patched_logits = model.run_with_hooks(
                clean_prompt,
                fwd_hooks=[(hook_name, patch_fn)]
            )

            patched_ld = logit_diff_metric(
                patched_logits,  # Already [1, pos, d_vocab]
                torch.tensor([clean_answer_id]).to(DEVICE),
                clean_distractor_ids,  # list of ints
            ).item()

            patch_results[f"L{layer}H{head}"] = patched_ld

    return {
        "clean_logit_diff": clean_ld,
        "corrupt_logit_diff": corrupt_ld,
        "patch_results": patch_results,
    }


def analyze_first_n_pairs(
    model: HookedTransformer,
    pairs: Dict[str, List[Dict[str, Any]]],
    n: int = 5,
) -> Dict[str, Any]:
    """Analyze first n cf pairs with activation patching.

    Returns:
        Dict with:
        - cache_shapes: from first clean prompt
        - baseline_logit_diffs: {cf_id: {clean, corrupt}}
        - patch_effects: {cf_id: {component: logit_diff_recovery}}
    """
    results = {
        "cache_shapes": None,
        "baseline_logit_diffs": {},
        "patch_effects": {},
    }

    cf_ids = list(pairs.keys())[:n]

    for i, cf_id in enumerate(cf_ids):
        clean_rec, corrupt_rec = pairs[cf_id]

        # Get cache shapes from first sample
        if i == 0:
            results["cache_shapes"] = compute_cache_shape(model, [clean_rec["prompt"]])

        # Run activation patching
        patch_result = activation_patching_experiment(
            model,
            clean_rec["prompt"],
            corrupt_rec["prompt"],
            clean_rec["answer_token_id"],
            corrupt_rec["answer_token_id"],
            clean_rec["distractor_answer_token_ids"],
            corrupt_rec["distractor_answer_token_ids"],
            clean_rec["source_var_token_pos"],
        )

        results["baseline_logit_diffs"][cf_id] = {
            "clean": patch_result["clean_logit_diff"],
            "corrupt": patch_result["corrupt_logit_diff"],
        }

        # Compute logit diff recovery
        # Recovery: How much of the clean→corrupt change is reversed by patching?
        # Use absolute values to handle sign inconsistencies
        clean_ld = patch_result["clean_logit_diff"]
        corrupt_ld = patch_result["corrupt_logit_diff"]
        total_diff = abs(clean_ld - corrupt_ld)

        patch_effects = {}
        for comp, patched_ld in patch_result["patch_results"].items():
            # Recovery: (distance from patched to corrupt) / (distance from clean to corrupt)
            # Using abs to handle direction
            recovery = abs(patched_ld - corrupt_ld) / total_diff if total_diff != 0 else 0
            patch_effects[comp] = {
                "patched_logit_diff": patched_ld,
                "recovery": recovery,
            }

        results["patch_effects"][cf_id] = patch_effects

    return results


def summarize_patch_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Summarize patch results across all cf pairs.

    Returns:
        Dict with:
        - top_heads: [(layer, head, avg_recovery)]
        - head_heatmap: [n_layers, n_heads] avg recovery matrix
    """
    from collections import defaultdict

    head_recoveries = defaultdict(list)

    for cf_id, effects in results["patch_effects"].items():
        for comp, data in effects.items():
            if comp.startswith("L") and "H" in comp:
                layer = int(comp.split("H")[0][1:])
                head = int(comp.split("H")[1])
                recovery = data["recovery"]
                head_recoveries[(layer, head)].append(recovery)

    # Average recovery per head
    avg_recoveries = []
    for (layer, head), recs in head_recoveries.items():
        avg = sum(recs) / len(recs)
        avg_recoveries.append((layer, head, avg))

    # Sort by recovery
    avg_recoveries.sort(key=lambda x: x[2], reverse=True)

    # Build heatmap
    n_layers = max(l for l, _, _ in avg_recoveries) + 1 if avg_recoveries else 12
    n_heads = max(h for _, h, _ in avg_recoveries) + 1 if avg_recoveries else 12

    heatmap = torch.zeros(n_layers, n_heads)
    for layer, head, avg in avg_recoveries:
        heatmap[layer, head] = avg

    return {
        "top_heads": avg_recoveries,
        "head_heatmap": heatmap,
    }
