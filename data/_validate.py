"""Validate generated var_binding JSONL files against datasets.md section 4.7.

Run after `python -m data.var_binding`. Exits non-zero on any failure so the
freeze step can be gated on this script.

Checks (per tier):
  - Tier 1: all tokenization_check == "passed"
  - Tier 1: cf pair clean/corrupt prompts have equal GPT-2 token length
  - Tier 1: cf pair source_var_token_pos identical
  - Tier 1: answer-value distribution spread <= 20%
  - Tier 1: variable-name usage spread <= 30%
  - Tier 2: tokenization_check == "passed", answer distribution spread <= 25%
"""
import json
import os
import sys
from collections import Counter
from typing import List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.config import DATASETS_DIR


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def check_tokenization(samples, label) -> bool:
    fails = [s for s in samples if s["tokenization_check"] != "passed"]
    if fails:
        print(f"  [FAIL] {label}: {len(fails)} samples failed tokenization_check")
        for s in fails[:3]:
            print(f"     {s['id']}: {s['tokenization_check']}")
        return False
    print(f"  [OK]   {label}: all {len(samples)} samples passed tokenization_check")
    return True


def check_cf_pair_alignment(samples) -> bool:
    """Tier 1 only. Pairs must share token length and source_var_token_pos."""
    from transformers import GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained("gpt2")

    by_cf: Dict[str, List[Dict[str, Any]]] = {}
    for s in samples:
        by_cf.setdefault(s["cf_id"], []).append(s)

    fails = 0
    for cf_id, pair in by_cf.items():
        if len(pair) != 2:
            print(f"  [FAIL] cf_id {cf_id}: expected 2 records, got {len(pair)}")
            fails += 1
            continue
        try:
            clean = next(s for s in pair if s["role"] == "clean")
            corrupt = next(s for s in pair if s["role"] == "corrupt")
        except StopIteration:
            print(f"  [FAIL] cf_id {cf_id}: missing clean/corrupt role")
            fails += 1
            continue

        clean_len = len(tok.encode(clean["prompt"]))
        corrupt_len = len(tok.encode(corrupt["prompt"]))
        if clean_len != corrupt_len:
            print(f"  [FAIL] {cf_id}: token length {clean_len} != {corrupt_len}")
            print(f"         clean   = {clean['prompt']!r}")
            print(f"         corrupt = {corrupt['prompt']!r}")
            fails += 1
            continue

        if clean["source_var_token_pos"] != corrupt["source_var_token_pos"]:
            print(f"  [FAIL] {cf_id}: source_var_token_pos "
                  f"{clean['source_var_token_pos']} != {corrupt['source_var_token_pos']}")
            fails += 1

    if fails == 0:
        print(f"  [OK]   all {len(by_cf)} cf pairs aligned (len + source_pos)")
        return True
    print(f"  [FAIL] {fails} alignment issues across {len(by_cf)} cf pairs")
    return False


def check_answer_distribution(samples, max_spread=0.20, label="answers") -> bool:
    counts = Counter(s["answer"] for s in samples)
    total = sum(counts.values())
    freqs = [c / total for c in counts.values()]
    spread = max(freqs) - min(freqs)
    if spread > max_spread:
        print(f"  [FAIL] {label} distribution spread {spread:.2%} > {max_spread:.0%}")
        print(f"         counts: {dict(sorted(counts.items()))}")
        return False
    print(f"  [OK]   {label} distribution spread {spread:.2%} (<= {max_spread:.0%})")
    return True


def check_name_distribution(samples, max_spread=0.30) -> bool:
    counts: Counter = Counter()
    for s in samples:
        names = [s["target_var"], s["source_var"]] + list(s["distractor_vars"])
        for v in names:
            counts[v] += 1
    if not counts:
        return True
    total = sum(counts.values())
    freqs = [c / total for c in counts.values()]
    spread = max(freqs) - min(freqs)
    if spread > max_spread:
        print(f"  [FAIL] var-name distribution spread {spread:.2%} > {max_spread:.0%}")
        print(f"         counts: {dict(sorted(counts.items()))}")
        return False
    print(f"  [OK]   var-name distribution spread {spread:.2%} (<= {max_spread:.0%})")
    return True


def main() -> int:
    ok = True

    t1_path = os.path.join(DATASETS_DIR, "var_binding_tier1.jsonl")
    t2_path = os.path.join(DATASETS_DIR, "var_binding_tier2.jsonl")

    if not os.path.exists(t1_path):
        print(f"[FAIL] {t1_path} not found. Run `python -m data.var_binding` first.")
        return 1

    print(f"\n=== Tier 1 ({os.path.basename(t1_path)}) ===")
    t1 = _load_jsonl(t1_path)
    ok &= check_tokenization(t1, "Tier 1")
    ok &= check_cf_pair_alignment(t1)
    ok &= check_answer_distribution(t1, 0.20, "Tier 1 answers")
    ok &= check_name_distribution(t1, 0.30)

    if os.path.exists(t2_path):
        print(f"\n=== Tier 2 ({os.path.basename(t2_path)}) ===")
        t2 = _load_jsonl(t2_path)
        ok &= check_tokenization(t2, "Tier 2")
        ok &= check_answer_distribution(t2, 0.25, "Tier 2 answers")
        # Tier 2 var-name distribution intentionally not enforced — larger
        # pool + variable n_vars makes uniform usage impractical.

    print()
    if not ok:
        print("[FAIL] Validation failed. Fix data before freeze.")
        return 1
    print("[OK]   All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
