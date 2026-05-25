"""P4 Syntax Task Circuit Analysis.

Analyzes circuits for grammatical processing in GPT-2.
Focus: Subject-verb agreement task as a representative syntax task.

Task: Determine if a sentence is grammatically correct.
Example: "The cat runs" (correct) vs "The cat run" (incorrect)
"""

import torch
import json
from pathlib import Path
from typing import List, Dict, Tuple, Any
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.config import RESULTS_DIR, MODEL_NAME

try:
    from transformer_lens import HookedTransformer
    import torch.nn.functional as F
except ImportError:
    print("Warning: transformer_lens not available. Syntax analysis limited.")


def generate_sva_sentences(n: int = 100) -> List[Dict[str, Any]]:
    """Generate subject-verb agreement test sentences.

    Grammatical rule: Subject and verb must agree in number.

    Returns:
        List of sentence dictionaries with prompt, completion, correct label
    """

    # Singular/plural noun pairs
    nouns = {
        'singular': ['cat', 'dog', 'bird', 'fish', 'tree'],
        'plural': ['cats', 'dogs', 'birds', 'fishes', 'trees']
    }

    # Singular/plural verb pairs
    verbs = {
        'singular': ['runs', 'jumps', 'sleeps', 'eats', 'plays'],
        'plural': ['run', 'jump', 'sleep', 'eat', 'play']
    }

    sentences = []

    for i in range(n):
        # Alternate between singular and plural
        is_singular = i % 2 == 0

        if is_singular:
            noun = nouns['singular'][i % len(nouns['singular'])]
            verb = verbs['singular'][i % len(verbs['singular'])]
            correct = f"The {noun} {verb}"
            incorrect = f"The {noun} {verbs['plural'][i % len(verbs['plural'])]}"
        else:
            noun = nouns['plural'][i % len(nouns['plural'])]
            verb = verbs['plural'][i % len(verbs['plural'])]
            correct = f"The {noun} {verb}"
            incorrect = f"The {noun} {verbs['singular'][i % len(verbs['singular'])]}"

        # Add both correct and incorrect examples
        sentences.append({
            'prompt': "The",
            'completion': f" {noun} {verb if is_singular else verbs['plural'][i % len(verbs['plural'])]}",
            'correct': is_singular,
            'type': 'subject-verb_agreement',
            'number': 'singular' if is_singular else 'plural'
        })

    return sentences


def create_sva_pairs(n_pairs: int = 100) -> List[Dict[str, Any]]:
    """Create clean/corrupt pairs for SVA analysis.

    Similar to P2 var binding structure for consistency.

    Args:
        n_pairs: Number of sentence pairs to generate

    Returns:
        List of clean/corrupt pairs
    """

    pairs = []

    # Singular/plural noun pairs
    nouns = [
        ('cat', 'cats'), ('dog', 'dogs'), ('bird', 'birds'),
        ('fish', 'fishes'), ('tree', 'trees')
    ]

    # Singular/plural verb pairs
    verbs = [
        ('runs', 'run'), ('jumps', 'jump'), ('sleeps', 'sleep'),
        ('eats', 'eat'), ('plays', 'play')
    ]

    for i in range(n_pairs):
        noun_idx = i % len(nouns)
        verb_idx = i % len(verbs)

        is_singular = i % 2 == 0

        if is_singular:
            # Singular subject
            clean = f"The {nouns[noun_idx][0]} {verbs[verb_idx][0]}"
            corrupt = f"The {nouns[noun_idx][0]} {verbs[verb_idx][1]}"
        else:
            # Plural subject
            clean = f"The {nouns[noun_idx][1]} {verbs[verb_idx][1]}"
            corrupt = f"The {nouns[noun_idx][1]} {verbs[verb_idx][0]}"

        pairs.append({
            'clean': clean,
            'corrupt': corrupt,
            'correct': True,
            'task': 'sva',
            'number': 'singular' if is_singular else 'plural'
        })

    return pairs


def analyze_sva_with_model(model, pairs: List[Dict[str, Any]], n_samples: int = 50) -> Dict[str, Any]:
    """Analyze subject-verb agreement using GPT-2.

    Args:
        model: HookedTransformer instance
        pairs: Clean/corrupt sentence pairs
        n_samples: Number of samples to analyze

    Returns:
        Analysis results with logit differences
    """

    results = []

    for i, pair in enumerate(pairs[:n_samples]):
        clean_text = pair['clean']
        corrupt_text = pair['corrupt']

        # Get model predictions
        with torch.no_grad():
            clean_logits = model(clean_text)
            corrupt_logits = model(corrupt_text)

        # Calculate log probabilities
        clean_logprob = F.log_softmax(clean_logits, dim=-1)
        corrupt_logprob = F.log_softmax(corrupt_logits, dim=-1)

        # Get probability of correct completion
        clean_prob = clean_logprob[-1].exp().max().item()
        corrupt_prob = corrupt_logprob[-1].exp().max().item()

        # Logit difference (how much more likely is clean vs corrupt)
        logit_diff = clean_prob - corrupt_prob

        results.append({
            'pair_id': i,
            'clean': clean_text,
            'corrupt': corrupt_text,
            'clean_prob': clean_prob,
            'corrupt_prob': corrupt_prob,
            'logit_diff': logit_diff,
            'task': pair['task'],
            'number': pair['number']
        })

    return {
        'results': results,
        'avg_logit_diff': sum(r['logit_diff'] for r in results) / len(results),
        'clean_correct': sum(1 for r in results if r['clean_prob'] > r['corrupt_prob']),
        'total_pairs': len(results)
    }


def generate_sva_dataset(n_pairs: int = 100, output_path: str = None) -> List[Dict[str, Any]]:
    """Generate and save SVA dataset.

    Args:
        n_pairs: Number of pairs to generate
        output_path: Path to save dataset

    Returns:
        Generated pairs
    """

    pairs = create_sva_pairs(n_pairs)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(pairs, f, indent=2)

        print(f"✅ Saved {len(pairs)} SVA pairs to {output_path}")

    return pairs


def analyze_sva_activation_patching(model, pairs: List[Dict[str, Any]], top_k: int = 20) -> Dict[str, Any]:
    """Analyze SVA using activation patching (similar to P2).

    This identifies which heads are important for grammatical processing.

    Args:
        model: HookedTransformer instance
        pairs: Clean/corrupt pairs
        top_k: Number of top heads to return

    Returns:
        Head importance scores
    """

    try:
        from transformer_lens.utils import get_act_name
    except ImportError:
        print("Error: transformer_lens.utils not available")
        return {}

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # Store attention patterns and activations
    head_activations = {}

    for layer in range(n_layers):
        for head in range(n_heads):
            head_activations[(layer, head)] = []

    # Analyze each pair
    for i, pair in enumerate(pairs[:min(len(pairs), 100)]):
        clean_text = pair['clean']

        # Get attention patterns
        _, cache = model.run_with_cache(clean_text)

        # Extract attention patterns
        for layer in range(n_layers):
            attention_pattern = cache['pattern', layer]  # [batch, head, q, k]
            # Average attention across batch and positions
            avg_attention = attention_pattern.mean(dim=(0, 2, 3))

            for head in range(n_heads):
                head_activations[(layer, head)].append(avg_attention[head].item())

    # Calculate head importance scores
    head_scores = []
    for (layer, head), activations in head_activations.items():
        if activations:
            avg_activation = sum(activations) / len(activations)
            std_activation = (sum((a - avg_activation) ** 2 for a in activations) / len(activations)) ** 0.5

            head_scores.append({
                'layer': layer,
                'head': head,
                'avg_activation': avg_activation,
                'std_activation': std_activation,
                'importance': avg_activation / (std_activation + 1e-8)
            })

    # Sort by importance
    head_scores.sort(key=lambda x: x['importance'], reverse=True)

    return {
        'top_heads': head_scores[:top_k],
        'all_heads': head_scores,
        'task': 'subject-verb_agreement',
        'model': MODEL_NAME
    }


def main():
    """Main P4 analysis function."""

    print("🔍 P4 Syntax Task Circuit Analysis")
    print("Task: Subject-Verb Agreement")
    print("-" * 50)

    # Generate dataset
    print("\n📝 Generating SVA dataset...")
    pairs = generate_sva_dataset(
        n_pairs=100,
        output_path='datasets/sva_pairs.json'
    )

    print(f"✅ Generated {len(pairs)} sentence pairs")

    # Sample pairs
    print("\n📊 Sample pairs:")
    for i, pair in enumerate(pairs[:5]):
        print(f"  {i+1}. Clean:   \"{pair['clean']}\"")
        print(f"      Corrupt: \"{pair['corrupt']}\"")

    # Try to load model for analysis
    try:
        print("\n🧠 Loading GPT-2 model...")
        model = HookedTransformer.from_pretrained(MODEL_NAME)
        print(f"✅ Loaded {MODEL_NAME}")

        # Run analysis
        print("\n🔬 Running activation patching analysis...")
        results = analyze_sva_activation_patching(model, pairs)

        print("\n📊 Top 10 Important Heads (Syntax Task):")
        for i, head in enumerate(results['top_heads'][:10]):
            print(f"  {i+1}. L{head['layer']}H{head['head']}: "
                  f"importance={head['importance']:.4f}")

        # Save results
        output_path = Path(RESULTS_DIR) / "sva" / "analysis_results.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✅ Saved results to {output_path}")

    except Exception as e:
        print(f"\n⚠️ Model analysis skipped: {e}")
        print("Dataset generated successfully, but circuit analysis requires full dependencies.")


if __name__ == "__main__":
    main()