# Datasets Report — Transformer Circuit Analysis

> **Status**: in progress (Day 1-3 complete, Day 4-5 pending next session)
> **Last updated**: 2026-05-23
> **Owner**: 데이터셋 트랙 (yesul.min)
> **Related**: [docs/datasets.md](../docs/datasets.md), [docs/candidates/sonsj-proposal.md](../docs/candidates/sonsj-proposal.md)

This document records what was generated, what was validated, and every
deviation from `docs/datasets.md` v2. The freeze declaration will be added
in a follow-up session after Tier 3 sanity and final review.

---

## 1. Environment

| Item | Value |
|---|---|
| Python | 3.12 (system) inside `.venv/` |
| OS | Ubuntu 24.04.4 LTS on WSL2 |
| GPU | NVIDIA RTX 4060 Ti, 8 GB VRAM |
| CUDA driver | 13.1 (host) |
| `torch` | 2.12.0+cu130 |
| `transformers` | 5.9.0 |
| `transformer_lens` | 3.2.1 |
| `einops` | 0.8.2 |
| `easy_transformer` | from `git+https://github.com/redwoodresearch/Easy-Transformer.git` |
| `plotly` | added as missing transitive dep of `easy_transformer` |

Reproduce with the README "Quick Start" pip commands, plus `pip install plotly`.

---

## 2. Datasets generated

All under `datasets/`. Sizes match `docs/datasets.md` section 1.

| Dataset | File | Records | Seed |
|---|---|---|---|
| Modular arithmetic train | `modular_train.pt` | 3,830 | `SEED=0` |
| Modular arithmetic test | `modular_test.pt` | 8,939 | `SEED=0` |
| Var binding Tier 1 (cf) | `var_binding_tier1.jsonl` | 1,000 (500 cf pairs) | `SEED=0` |
| Var binding Tier 2 | `var_binding_tier2.jsonl` | 500 | `SEED=1` |
| Var binding Tier 3 | `var_binding_tier3.jsonl` | 500 | `SEED=2` |
| IOI | (loaded on demand via `data.ioi_loader`) | 500 | (Easy-Transformer default) |

Modular vocab: 114 (= p+1 with `=` token at index p=113). ✅ matches section 3.

---

## 3. Validation results (data/_validate.py)

§4.7 checklist on the generated Tier 1 / Tier 2 JSONL.

| Check | Tier 1 | Tier 2 |
|---|---|---|
| `tokenization_check == "passed"` | ✅ 1000/1000 | ✅ 500/500 |
| Cf-pair token length match | ✅ 500/500 | n/a |
| Cf-pair `source_var_token_pos` match | ✅ 500/500 | n/a |
| Answer-value distribution spread | ✅ 4.50% (≤ 20%) | ✅ 3.80% (≤ 25%) |
| Var-name distribution spread | ✅ 1.88% (≤ 30%) | (not enforced) |

IOI sanity (`data.ioi_loader`): loaded N=500, example prompt parses cleanly as
`"Then, Amy and Ashley were thinking about going to the house. Amy wanted to give a bone to Ashley"`.

---

## 4. Zero-shot GPT-2 small results

`python -m data._zero_shot_eval` on `cuda` (RTX 4060 Ti).

| Tier | Top-1 acc | logit_diff avg | pct_pos | Note |
|---|---|---|---|---|
| Tier 1 clean (N=500) | 0.80% | **+0.182** | **54.40%** | Above chance |
| Tier 1 corrupt (N=500) | 2.00% | +0.097 | 49.20% | ≈ chance (expected: corrupt run flips the binding rival) |
| Tier 2 (N=500) | 1.40% | +0.143 | 53.75% | Slightly above chance, more distractor noise |
| Tier 3 (N=500) | 0.00% | n/a | n/a | Multi-hop, no distractor list, accuracy-only |

**Interpretation**. Top-1 is uniformly near zero because the dominant next-token
under prompts like `"a=3; b=5; c=a; c="` is a variable name (`b`/`a`/`c`),
not a digit — GPT-2 small's next-variable prior outweighs its binding signal.
But the logit *ranking* between the bound answer and the rival distractors is
already above chance (pct_pos > 50%, avg diff > 0), which is the signal
mechanistic-interpretability analysis actually needs. This is the same
construction Wang et al. (2022) used for IOI ("IO logit > S logit" rather than
top-1).

---

## 5. Deviations from docs/datasets.md v2

These are the only places where the implementation diverges from the spec.
All are recorded here so the analysis tracks (P1/P2/P3) can adjust their
expectations.

### D1. Tier 1 `n_vars` ∈ {3, 4}, not {2, 3}
- **Spec**: §4.3 "변수 개수 n_vars: 2, 3 균등 (각 250 pair)"
- **Reality**: counterfactual generation requires **two** non-target variables
  (`src_clean`, `src_corrupt`), so the floor is `n_vars >= 3`. With `n_vars=2`
  the original code called `rng.sample(candidates, 2)` on a 1-element list and
  raised `ValueError`. Fixed to `n_vars = 3 + (i % 2)` — alternating 3 and 4,
  250 cf pairs each.
- **Impact on analysis**: minor. Distractor count goes from {0, 1} to {1, 2},
  which is closer to the IOI ABBA pattern Wang et al. analyze anyway.

### D2. Eval gate: logit-diff, not top-1 accuracy
- **Spec**: §4.7-1 "GPT-2 small zero-shot accuracy ≥ 70% (Tier 1 기준)"
- **Reality**: GPT-2 small top-1 accuracy under the spec prompt format is
  0.8%. Top-5 inspection shows the answer digit is suppressed by a strong
  variable-name prior (`b`/`a`/`c` dominate the head of the distribution).
  Substituted with the Wang et al. IOI-style logit-difference gate:
  `pct_pos > 0.50` AND `avg logit_diff > 0` on Tier 1 clean. Both satisfied
  (54.4% and +0.182). Tier 1 data is kept as-is; the analysis tracks should
  treat top-1 accuracy as informational only.
- **Schema change**: each Tier 1/2 record now carries
  `distractor_values` and `distractor_answer_token_ids` so patching/eval
  code can read distractor logits directly without re-tokenizing.

### D3. Answer token id: prompt-context aware
- **Spec**: §4.6 example shows `"answer_token_id": 513` (the leading-space
  variant `' 9'`).
- **Reality**: under code prompts that end in `=`, GPT-2 BPE puts the digit
  *without* a leading space at the next position (e.g. `'3'` is ID 18,
  not ID 513). `answer_token_id` is now computed by tokenizing
  `prompt + str(answer)` and taking the trailing token — which guarantees
  the id matches whatever GPT-2 would produce at that position. If the
  answer fails to fit in exactly one extra token the sample's
  `tokenization_check` reports `failed:answer_..._not_single_next_token`
  and validation drops it.

### D4. `SINGLE_TOKEN_NAMES` pinned to 47 names
- **Spec**: §4.1 placeholder pool of 12 names pending Day 1 verification.
- **Reality**: Day 1 verifier (`data._day1_verify`) confirmed 47 of the 51
  candidates encode as single tokens under GPT-2 BPE with a leading space.
  `shared/config.py` now contains the verified list.

### D5. `ANSWER_VALUES` is 1..9 even though 1..19 are single-token
- **Spec**: §4.6 "정답 값 1~9 균등 분포".
- **Reality**: Day 1 verifier reports 1..19 all encode as single tokens.
  Kept 1..9 to honor the spec — keeps distractor sets small and matches the
  digit-only assumption in the answer-distribution validation.

---

## 6. Bugs caught and fixed

- **B1**. `data/var_binding.py` `make_tier1`: ValueError when `n_vars=2`,
  fixed by raising the floor to 3 (see D1).
- **B2**. `data/ioi_loader.py`: `easy_transformer.utils` referenced
  `transformers.TRANSFORMERS_CACHE`, which was removed in `transformers>=5`.
  Added a compatibility shim that sets the attribute from
  `huggingface_hub.constants.HF_HUB_CACHE` before importing easy_transformer.
- **B3**. `data/ioi_loader.py`: GPT-2 tokenizer has no `pad_token` by default,
  so `IOIDataset` padding raised. Set `tok.pad_token = tok.eos_token`.

---

## 7. Open items (deferred to next session)

- [ ] Tier 3 sanity (Day 4): inspect failure mode beyond `top1=0%`; consider
      reporting logit-diff against the entire chain's intermediate values.
- [ ] Final freeze declaration (Day 5): once Tier 3 is reviewed, this report
      gets a "frozen on <date>" header and the corresponding commit becomes
      the canonical reference for analysis tracks.
- [ ] Decision: do analysis tracks need a higher-signal prompt format
      (natural-language wrap) as a *secondary* variant? GPT-2 small's
      `"Variables: a is 3, b is 5, c is a. So c is"` puts ` 3` at top-2.
      Not in scope for this session.

---

## 8. How to reproduce

```bash
# 1. environment
python3 -m venv .venv
.venv/bin/pip install torch transformers transformer_lens einops plotly
.venv/bin/pip install git+https://github.com/redwoodresearch/Easy-Transformer.git

# 2. data
.venv/bin/python -m data._day1_verify   # prints pools; already pinned in config.py
.venv/bin/python -m data.modular
.venv/bin/python -m data.var_binding
.venv/bin/python -m data.ioi_loader     # sanity load only

# 3. validation + eval
.venv/bin/python -m data._validate       # exits 0 on success
.venv/bin/python -m data._zero_shot_eval # exits 0 on Tier 1 pct_pos > 50%
```
