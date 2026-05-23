"""Code Variable Binding dataset (Step 3).

Tier 1: counterfactual pairs, single-letter vars, 1-hop — activation patching.
Tier 2: scaling, single-token names, 1-hop — circuit robustness across n_vars.
Tier 3: optional, multi-token names + multi-hop — accuracy only.

See docs/datasets.md §4 for design rationale.
"""
import os
import sys
import json
import random
from typing import List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.config import (
    SEED, DATASETS_DIR,
    SINGLE_LETTER_POOL, SINGLE_TOKEN_NAMES, ANSWER_VALUES,
    VB_TIER1_N_PAIRS, VB_TIER2_N, VB_TIER3_N,
)


# ---- Lazy tokenizer load (only needed to populate token_id / lens) ----
_TOK = None


def _get_tokenizer():
    global _TOK
    if _TOK is None:
        from transformers import GPT2Tokenizer
        _TOK = GPT2Tokenizer.from_pretrained("gpt2")
    return _TOK


def _make_record(sid, cf_id, role, tier, prompt, answer,
                 n_vars, target_var, source_var, distractors, hop,
                 tokenization_check="passed"):
    tok = _get_tokenizer()
    return {
        "id": sid,
        "cf_id": cf_id,
        "role": role,
        "tier": tier,
        "prompt": prompt,
        "answer": answer,
        "answer_token_id": tok.encode(" " + str(answer))[0],
        "n_vars": n_vars,
        "target_var": target_var,
        "source_var": source_var,
        "distractor_vars": distractors,
        "binding_hop": hop,
        "var_name_token_lens": {
            v: len(tok.encode(" " + v))
            for v in [target_var, source_var] + distractors
        },
        "tokenization_check": tokenization_check,
    }


# ---------------- Tier 1 ----------------
def gen_tier1_pair(n_vars: int, rng: random.Random, pair_idx: int):
    """Generate one counterfactual pair (clean + corrupted).

    Clean and corrupted prompts differ in exactly one token (the source var
    in the binding line), so token positions stay aligned.
    """
    names = rng.sample(SINGLE_LETTER_POOL, n_vars)
    values = {v: rng.choice(ANSWER_VALUES) for v in names}

    target = names[-1]
    candidates = [v for v in names if v != target]
    src_clean, src_corrupt = rng.sample(candidates, 2)

    # Force differing answers so the pair is informative
    if values[src_clean] == values[src_corrupt]:
        alt = [v for v in ANSWER_VALUES if v != values[src_clean]]
        values[src_corrupt] = rng.choice(alt)

    prefix = "; ".join(f"{v}={values[v]}" for v in names if v != target)
    clean = f"{prefix}; {target}={src_clean}; {target}="
    corrupt = f"{prefix}; {target}={src_corrupt}; {target}="

    cf_id = f"vb1_{pair_idx:04d}"
    distractors = [v for v in names
                   if v not in (target, src_clean, src_corrupt)]

    return [
        _make_record(f"{cf_id}_clean", cf_id, "clean", 1, clean,
                     values[src_clean], n_vars, target, src_clean,
                     distractors, 1),
        _make_record(f"{cf_id}_corrupt", cf_id, "corrupt", 1, corrupt,
                     values[src_corrupt], n_vars, target, src_corrupt,
                     distractors, 1),
    ]


def make_tier1(n_pairs=VB_TIER1_N_PAIRS, seed=SEED) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    samples = []
    for i in range(n_pairs):
        n_vars = 2 + (i % 2)   # alternate 2, 3
        samples.extend(gen_tier1_pair(n_vars, rng, i))
    return samples


# ---------------- Tier 2 ----------------
def make_tier2(n=VB_TIER2_N, seed=SEED + 1) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    pool = SINGLE_TOKEN_NAMES if SINGLE_TOKEN_NAMES else SINGLE_LETTER_POOL
    n_var_options = [2, 3, 4, 5, 6]
    samples = []
    for i in range(n):
        n_vars = min(n_var_options[i % len(n_var_options)], len(pool))
        names = rng.sample(pool, n_vars)
        values = {v: rng.choice(ANSWER_VALUES) for v in names}
        target = names[-1]
        source = rng.choice([v for v in names if v != target])
        prefix = "; ".join(f"{v}={values[v]}" for v in names if v != target)
        prompt = f"{prefix}; {target}={source}; {target}="
        distractors = [v for v in names if v not in (target, source)]
        sid = f"vb2_{i:04d}"
        samples.append(_make_record(sid, sid, "single", 2, prompt,
                                    values[source], n_vars, target, source,
                                    distractors, 1))
    return samples


# ---------------- Tier 3 (optional) ----------------
def make_tier3(n=VB_TIER3_N, seed=SEED + 2) -> List[Dict[str, Any]]:
    """Multi-token names + multi-hop chains. Accuracy-only, no patching."""
    rng = random.Random(seed)
    pool = ["my_var", "result_1", "data_x", "value", "tmp",
            "input_a", "output_b", "counter", "score_1"]
    samples = []
    for i in range(n):
        n_vars = rng.randint(3, 5)
        names = rng.sample(pool, min(n_vars, len(pool)))
        root_val = rng.choice(ANSWER_VALUES)
        lines = [f"{names[0]}={root_val}"]
        for j in range(1, len(names)):
            lines.append(f"{names[j]}={names[j-1]}")
        target = names[-1]
        prompt = "; ".join(lines) + f"; {target}="
        sid = f"vb3_{i:04d}"
        rec = _make_record(sid, sid, "single", 3, prompt, root_val,
                           len(names), target, names[0],
                           names[1:-1], hop=len(names) - 1,
                           tokenization_check="multi_token_allowed")
        samples.append(rec)
    return samples


# ---------------- IO ----------------
def _save_jsonl(records, path):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def save_all():
    os.makedirs(DATASETS_DIR, exist_ok=True)
    t1 = make_tier1()
    t2 = make_tier2()
    t3 = make_tier3()
    _save_jsonl(t1, os.path.join(DATASETS_DIR, "var_binding_tier1.jsonl"))
    _save_jsonl(t2, os.path.join(DATASETS_DIR, "var_binding_tier2.jsonl"))
    _save_jsonl(t3, os.path.join(DATASETS_DIR, "var_binding_tier3.jsonl"))
    print(f"Saved: Tier1={len(t1)} (={len(t1)//2} cf pairs), "
          f"Tier2={len(t2)}, Tier3={len(t3)}")


if __name__ == "__main__":
    save_all()
