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


def _verify_single_token_vars(tok, target_var, source_var, distractors):
    """Return 'passed' or 'failed:<reason>' for Tier 1/2 var-name constraint."""
    for v in [target_var, source_var] + list(distractors):
        if len(tok.encode(" " + v)) != 1:
            return f"failed:var_{v}_multi_token"
    return "passed"


def _next_token_for_answer(tok, prompt, answer):
    """Return (token_id, status). status is 'passed' when (prompt + str(answer))
    tokenizes as prompt's tokens + exactly one extra token — i.e. the answer
    fits as the next-token logit under this prefix.

    In code context (prompt ends with `=`), GPT-2 BPE picks the no-space digit
    (e.g. ID 18 for '3'), not the leading-space variant (ID 513 for ' 3').
    Computing the answer id from the joined string avoids guessing that.
    """
    prompt_ids = tok.encode(prompt)
    full_ids = tok.encode(prompt + str(answer))
    if (len(full_ids) == len(prompt_ids) + 1
            and full_ids[:len(prompt_ids)] == prompt_ids):
        return full_ids[-1], "passed"
    return None, f"failed:answer_{answer}_not_single_next_token"


def _binding_source_token_pos(tok, prompt, target_var, source_var):
    """Locate the source-var token inside the binding line.

    For 1-hop Tier 1/2 the prompt contains exactly one occurrence of
    "; {target}={source}" (the binding line). Re-tokenize the prefix up to
    and including "{target}=" to get the 0-indexed position of the source
    token. Returns None if the marker isn't found (e.g., multi-hop tier 3).
    """
    marker = f"; {target_var}={source_var}"
    idx = prompt.find(marker)
    if idx < 0:
        return None
    prefix_str = prompt[:idx] + f"; {target_var}="
    return len(tok.encode(prefix_str))


def _make_record(sid, cf_id, role, tier, prompt, answer,
                 n_vars, target_var, source_var, distractors, hop,
                 distractor_values=None):
    tok = _get_tokenizer()
    answer_token_id, ans_status = _next_token_for_answer(tok, prompt, answer)

    # Distractor answer token ids for logit-difference evaluation
    # (Wang et al. 2022 IOI standard). Computed per-sample so the prompt
    # context dictates the right BPE variant (no-space digit after `=`).
    distractor_values = list(distractor_values or [])
    distractor_answer_token_ids = []
    for dv in distractor_values:
        d_id, _ = _next_token_for_answer(tok, prompt, dv)
        distractor_answer_token_ids.append(d_id)

    if tier == 3:
        # Multi-hop tier 3 is accuracy-only; we still want a usable answer id
        # but skip the strict var-name and binding-line checks.
        check = "multi_token_allowed"
        source_pos = None
        if answer_token_id is None:  # extremely unlikely (digits only)
            answer_token_id = tok.encode(str(answer))[0]
    else:
        var_check = _verify_single_token_vars(tok, target_var, source_var,
                                              distractors)
        if var_check != "passed":
            check = var_check
        elif ans_status != "passed":
            check = ans_status
        else:
            check = "passed"
        source_pos = _binding_source_token_pos(tok, prompt,
                                               target_var, source_var)

    return {
        "id": sid,
        "cf_id": cf_id,
        "role": role,
        "tier": tier,
        "prompt": prompt,
        "answer": answer,
        "answer_token_id": answer_token_id,
        "n_vars": n_vars,
        "target_var": target_var,
        "source_var": source_var,
        "distractor_vars": distractors,
        "distractor_values": distractor_values,
        "distractor_answer_token_ids": distractor_answer_token_ids,
        "binding_hop": hop,
        "source_var_token_pos": source_pos,
        "answer_token_pos": len(tok.encode(prompt)),
        "var_name_token_lens": {
            v: len(tok.encode(" " + v))
            for v in [target_var, source_var] + distractors
        },
        "tokenization_check": check,
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

    # Logit-diff distractors include the *other* counterfactual source value
    # plus any pure distractors. For clean, the rival is src_corrupt's value;
    # for corrupt, it's src_clean's value.
    clean_distractor_vals = [values[src_corrupt]] + [values[d] for d in distractors]
    corrupt_distractor_vals = [values[src_clean]] + [values[d] for d in distractors]

    return [
        _make_record(f"{cf_id}_clean", cf_id, "clean", 1, clean,
                     values[src_clean], n_vars, target, src_clean,
                     distractors, 1, clean_distractor_vals),
        _make_record(f"{cf_id}_corrupt", cf_id, "corrupt", 1, corrupt,
                     values[src_corrupt], n_vars, target, src_corrupt,
                     distractors, 1, corrupt_distractor_vals),
    ]


def make_tier1(n_pairs=VB_TIER1_N_PAIRS, seed=SEED) -> List[Dict[str, Any]]:
    """Tier 1 cf pairs.

    n_vars alternates 3 and 4 — datasets.md §4.3 originally listed {2,3} but
    counterfactual generation needs ≥2 non-target vars (src_clean, src_corrupt),
    so the floor is 3. Deviation logged in datasets/_report.md.
    """
    rng = random.Random(seed)
    samples = []
    for i in range(n_pairs):
        n_vars = 3 + (i % 2)
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
        distractor_vals = [values[d] for d in distractors]
        sid = f"vb2_{i:04d}"
        samples.append(_make_record(sid, sid, "single", 2, prompt,
                                    values[source], n_vars, target, source,
                                    distractors, 1, distractor_vals))
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
                           names[1:-1], hop=len(names) - 1)
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
