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
  - Freeze guard: SHA-256 of frozen files must match FROZEN_HASHES below
    (warning only — does not block exit unless --strict is passed).
"""
import hashlib
import json
import os
import sys
from collections import Counter
from typing import List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.config import DATASETS_DIR


# ============================================================
# Freeze hashes — SHA-256 of canonical Day 1-3 datasets.
# Stamped at freeze (datasets/_report.md §9). If you regenerate
# data, either update these hashes (with team sign-off) or revert.
# ============================================================
FROZEN_HASHES = {
    "modular_train.pt":        "2034d5a356c95d57c034b16fbda9c427d94ccdf5018caff4b8ab79f3c92589f0",
    "modular_test.pt":         "99135adb90e06ec759d88077b4ffca093de7c2a0b5870c966449bf6242393918",
    "var_binding_tier1.jsonl": "296dd2c9c69597de0227b996e1d35c0e8dc44b38db72a856cce61b7fa71275f7",
    "var_binding_tier2.jsonl": "9024ccb99ed30c0f020cc4fe326d281c11934b1cd5ecd5e3f19007b0a69790c8",
    "var_binding_tier3.jsonl": "a870214c01c0afbd11e3f9b5c9ae3a60a1415bb68c1dbdf73afef3ae56e11ce0",
}


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


def check_freeze_hashes(strict: bool) -> bool:
    """Compare SHA-256 of frozen files against FROZEN_HASHES.

    Emits warnings on mismatch. Only fails the run when strict=True (used by
    CI/the freeze gate). Day-to-day runs report drift without blocking, so
    contributors can iterate locally before deciding to update the table.
    """
    print("\n=== Freeze guard (SHA-256) ===")
    all_match = True
    for fname, expected in FROZEN_HASHES.items():
        path = os.path.join(DATASETS_DIR, fname)
        if not os.path.exists(path):
            print(f"  [WARN] {fname}: file missing (skipped)")
            all_match = False
            continue
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        actual = h.hexdigest()
        if actual == expected:
            print(f"  [OK]   {fname}")
        else:
            print(f"  [WARN] {fname}: hash drift")
            print(f"         expected {expected}")
            print(f"         actual   {actual}")
            all_match = False
    if not all_match and strict:
        print("[FAIL] Freeze hash mismatch and --strict was set.")
    return all_match


def main() -> int:
    strict = "--strict" in sys.argv
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

    hashes_match = check_freeze_hashes(strict=strict)
    if strict and not hashes_match:
        ok = False

    print()
    if not ok:
        print("[FAIL] Validation failed. Fix data before freeze.")
        return 1
    print("[OK]   All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
