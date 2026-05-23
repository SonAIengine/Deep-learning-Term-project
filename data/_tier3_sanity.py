"""Day 4 sanity — examine Tier 3 (multi-hop) failure modes on GPT-2 small.

Reports three views to settle whether Tier 3 should stay accuracy-only or
adopt a logit-difference metric (datasets/_report.md §7.1):

1. Top-5 predictions for the first 10 Tier 3 samples (qualitative look).
2. Answer rank distribution by binding_hop (where does the correct digit sit?).
3. Option-B logit-diff — root_val vs the *other* ANSWER_VALUES digits — under
   the same prompt context. Tier 3 has no distractor-variable VALUES (all
   intermediate vars take the same value as the root), so the only available
   "rival set" is the rest of {1..9}. If GPT-2 prefers the right digit above
   chance against that set, logit-diff is informative; otherwise Tier 3 is
   accuracy-only.
"""
import json
import os
import sys
from collections import Counter, defaultdict
from typing import List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.config import DATASETS_DIR, MODEL_NAME, DEVICE, ANSWER_VALUES


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def _other_digit_token_ids(tok, prompt: str, answer: int) -> List[int]:
    """Token id for each ANSWER_VALUES digit other than `answer`, in this prompt
    context. Uses the same join trick as data.var_binding._next_token_for_answer
    so we hit the no-space BPE variant."""
    prompt_ids = tok.encode(prompt)
    ids = []
    for v in ANSWER_VALUES:
        if v == answer:
            continue
        full = tok.encode(prompt + str(v))
        if len(full) == len(prompt_ids) + 1 and full[:len(prompt_ids)] == prompt_ids:
            ids.append(full[-1])
    return ids


def main() -> int:
    import torch
    from transformer_lens import HookedTransformer
    from transformers import GPT2Tokenizer

    t3_path = os.path.join(DATASETS_DIR, "var_binding_tier3.jsonl")
    if not os.path.exists(t3_path):
        print(f"[FAIL] {t3_path} not found.")
        return 1
    samples = _load_jsonl(t3_path)

    print(f"Loading {MODEL_NAME} on {DEVICE}...")
    model = HookedTransformer.from_pretrained(MODEL_NAME)
    model.to(DEVICE).eval()
    tok = GPT2Tokenizer.from_pretrained(MODEL_NAME)

    # ----- 1. Top-5 predictions for first 10 samples -----
    print("\n=== Top-5 predictions, first 10 Tier 3 samples ===")
    with torch.no_grad():
        for s in samples[:10]:
            ids = torch.tensor([tok.encode(s["prompt"])], device=DEVICE)
            last = model(ids)[0, -1]
            top5 = torch.topk(last, 5)
            top_tokens = [
                (tok.decode([int(i)]), float(v))
                for i, v in zip(top5.indices.tolist(), top5.values.tolist())
            ]
            ans_logit = float(last[s["answer_token_id"]].item())
            rank = int((last > last[s["answer_token_id"]]).sum().item()) + 1
            tokens_pretty = ", ".join(f"{repr(t)}={lg:+.2f}" for t, lg in top_tokens)
            print(f"  {s['id']} hop={s['binding_hop']} n_vars={s['n_vars']} "
                  f"ans={s['answer']} (id={s['answer_token_id']}, "
                  f"logit={ans_logit:+.2f}, rank={rank})")
            print(f"    prompt: {s['prompt']}")
            print(f"    top5:   {tokens_pretty}")

    # ----- 2. Answer rank distribution, all 500, grouped by binding_hop -----
    print("\n=== Answer rank distribution by binding_hop (all N=500) ===")
    rank_by_hop: Dict[int, List[int]] = defaultdict(list)
    with torch.no_grad():
        for s in samples:
            ids = torch.tensor([tok.encode(s["prompt"])], device=DEVICE)
            last = model(ids)[0, -1]
            rank = int((last > last[s["answer_token_id"]]).sum().item()) + 1
            rank_by_hop[s["binding_hop"]].append(rank)

    def _bucket(rs: List[int]) -> Counter:
        c = Counter()
        for r in rs:
            if r == 1:
                c["1"] += 1
            elif r <= 5:
                c["2-5"] += 1
            elif r <= 20:
                c["6-20"] += 1
            elif r <= 100:
                c["21-100"] += 1
            else:
                c[">100"] += 1
        return c

    for hop in sorted(rank_by_hop):
        rs = rank_by_hop[hop]
        c = _bucket(rs)
        median = sorted(rs)[len(rs) // 2]
        line = " | ".join(
            f"{k}: {c[k]} ({c[k] / len(rs):.0%})"
            for k in ["1", "2-5", "6-20", "21-100", ">100"]
        )
        print(f"  hop={hop} N={len(rs)} median_rank={median} | {line}")

    # ----- 3. Option-B logit-diff: root_val vs other ANSWER_VALUES digits -----
    print("\n=== Option-B logit-diff: root_val vs other ANSWER_VALUES (N=500) ===")
    diffs: List[float] = []
    pos = 0
    with torch.no_grad():
        for s in samples:
            ids = torch.tensor([tok.encode(s["prompt"])], device=DEVICE)
            last = model(ids)[0, -1]
            rival_ids = _other_digit_token_ids(tok, s["prompt"], s["answer"])
            if not rival_ids:
                continue
            rival_logits = last[rival_ids]
            diff = (last[s["answer_token_id"]] - rival_logits.mean()).item()
            diffs.append(diff)
            if diff > 0:
                pos += 1

    if diffs:
        avg = sum(diffs) / len(diffs)
        pct_pos = pos / len(diffs)
        print(f"  N={len(diffs)} avg_diff={avg:+.3f} pct_pos={pct_pos:.2%} "
              f"(chance=50%)")
        if pct_pos > 0.50 and avg > 0:
            print("  -> Above chance: Option-B logit-diff carries signal. "
                  "Worth reporting alongside accuracy.")
        else:
            print("  -> At/below chance: Tier 3 should remain accuracy-only.")
    else:
        print("  no diffs computed")

    return 0


if __name__ == "__main__":
    sys.exit(main())
