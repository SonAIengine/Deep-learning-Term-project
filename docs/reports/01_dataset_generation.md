# 데이터셋 생성 및 검증

> **실험 ID**: EX-001
> **관련 Commit**: `95621c4`, `a3d4c82` (freeze)
> **작성일**: 2026-05-24
> **상태**: ✅ 완료
> **다음**: [`02_grokking_training.md`](./02_grokking_training.md)

---

## 1. 개요

Transformer circuit analysis를 위한 데이터셋 3종을 생성하고 검증한다:

1. **Modular Arithmetic** — Grokking 실험용
2. **Variable Binding** — Code-style var tracking 회로 분석용
3. **IOI** — Wang et al. (2022)와 비교용

---

## 2. 환경

| 항목 | 값 |
|---|---|
| Python | 3.12 (system) inside `.venv/` |
| OS | Ubuntu 24.04.4 LTS on WSL2 |
| GPU | NVIDIA RTX 4060 Ti, 8 GB VRAM |
| `torch` | 2.12.0+cu130 |
| `transformers` | 5.9.0 |
| `transformer_lens` | 3.2.1 |
| `einops` | 0.8.2 |
| `easy_transformer` | `git+https://github.com/redwoodresearch/Easy-Transformer.git` |

---

## 3. 생성된 데이터셋

### 3.1 Modular Arithmetic (P1용)

| 항목 | 값 |
|---|---|
| 파일 | `datasets/modular_train.pt`, `modular_test.pt` |
| 연산 | a × b mod p (p=113) |
| Train | 3,830 샘플 (전체의 ~30%) |
| Test | 8,939 샘플 (전체의 ~70%) |
| Seed | `SEED=0` |
| Vocab | 114 (= p + 1, `=` 토큰을 인덱스 113에 배치) |

### 3.2 Variable Binding (P2용)

| 항목 | Tier 1 | Tier 2 | Tier 3 |
|---|---|---|---|
| 파일 | `var_binding_tier1.jsonl` | `var_binding_tier2.jsonl` | `var_binding_tier3.jsonl` |
| 레코드 수 | 1,000 (500 cf pair) | 500 | 500 |
| Seed | 0 | 1 | 2 |
| `n_vars` | 3 or 4 | 3 | 3 |
| Hop 길이 | 1 | 1 | 2~4 |

**Counterfactual 구조 (Tier 1)**:
```
Clean:   "a=3; b=5; c=a; c="          → answer: 3
Corrupt: "a=3; b=7; c=a; c="          → answer: 3 (distractor: 7)
```

### 3.3 IOI (P3용)

| 항목 | 값 |
|---|---|
| 로더 | `data.ioi_loader` |
| N | 500 |
| 출처 | Easy-Transformer 내장 |

---

## 4. 검증 결과

### 4.1 Tier 1 / Tier 2 검증

| 체크 항목 | Tier 1 | Tier 2 |
|---|---|---|
| `tokenization_check == "passed"` | ✅ 1000/1000 | ✅ 500/500 |
| cf pair 토큰 길이 일치 | ✅ 500/500 | N/A |
| cf pair `source_var_token_pos` 일치 | ✅ 500/500 | N/A |
| 정답값 분포 스프레드 | ✅ 4.50% (≤ 20%) | ✅ 3.80% (≤ 25%) |
| 변수명 분포 스프레드 | ✅ 1.88% (≤ 30%) | N/A |

### 4.2 GPT-2 Small Zero-shot 결과

| Tier | Top-1 acc | logit_diff avg | pct_pos | 비고 |
|---|---|---|---|---|
| Tier 1 clean | 0.80% | **+0.182** | **54.40%** | chance(50%) 위 ✅ |
| Tier 1 corrupt | 2.00% | +0.097 | 49.20% | ≈ chance |
| Tier 2 | 1.40% | +0.143 | 53.75% | chance 위 ✅ |
| Tier 3 | 0.00% | +0.459† | 53.80%† | multi-hop |

† Tier 3는 root_val vs `ANSWER_VALUES` \ {root_val}로 계산.

**해석**: Top-1 accuracy가 낮은 이유는 GPT-2가 변수명(`a`/`b`/`c`)을 먼저 예측하기 때문. 하지만 logit-diff metric은 chance 위이므로 분석 가능한 신호가 존재한다.

### 4.3 Tier 3 Multi-hop 실패 양상

| Hop | N | median rank | top-5 비율 | rank>100 비율 |
|-----|---|-------------|------------|---------------|
| 2 | 152 | 15 | 7% | 11% |
| 3 | 175 | 31 | 5% | 25% |
| 4 | 173 | 60 | 5% | 35% |

Chain 길이가 길어질수록 정답 rank가 악화됨. Tier 3는 accuracy-only로 freeze되며, 보조 지표로 rank-by-hop 분포 사용.

---

## 5. 명세와의 일탈 (Deviation)

### D1. Tier 1 `n_vars` ∈ {3, 4} (명세는 {2, 3})
- **이유**: Counterfactual 생성에는 target이 아닌 변수가 2개 필요
- **영향**: Distractor 개수가 {0,1} → {1,2}로 증가. Wang et al. IOI ABBA 구조에 더 가까워짐.

### D2. Eval gate: top-1 → logit-diff
- **이유**: GPT-2 small의 top-1 accuracy는 0.8%
- **변경**: `pct_pos > 50%` AND `avg logit_diff > 0`로 gate 통과

### D3. `answer_token_id` 계산 방식
- **변경**: `prompt + str(answer)`를 함께 토큰화한 후 마지막 토큰 취득
- **이유**: `=` 다음 위치에서 GPT-2 BPE는 leading space 없는 숫자를 둠

### D4-D6: 상세 내용은 원본 `_report.md` §5 참조

---

## 6. 발견·수정된 버그

| 버그 | 위치 | 수정 |
|------|------|------|
| B1 | `data/var_binding.py` | `n_vars` 하한을 2 → 3으로 수정 |
| B2 | `data/ioi_loader.py` | `HF_HUB_CACHE` compat shim 추가 |
| B3 | `data/ioi_loader.py` | `pad_token = eos_token` 설정 |

---

## 7. 동결 정책 (Freeze)

**Freeze Commit**: `a3d4c82` (2026-05-24)

### 7.1 동결 대상 파일

| 파일 | SHA-256 |
|---|---|
| `modular_train.pt` | `2034d5a3…2589f0` |
| `modular_test.pt` | `99135adb…393918` |
| `var_binding_tier1.jsonl` | `296dd2c9…1275f7` |
| `var_binding_tier2.jsonl` | `9024ccb9…9790c8` |
| `var_binding_tier3.jsonl` | `a870214c…6e11ce0` |

### 7.2 동결 후 변경 절차

1. 변경 사유와 영향 범위를 명시
2. 분석 트랙 담당자 sign-off
3. Seed 고정 (`SEED=0`) 재생성 검증
4. `FROZEN_HASHES` 갱신
5. Commit 메시지에 "refreeze" 명시

### 7.3 신뢰할 수 있는 불변량

분석 트랙은 다음 값을 신뢰하고 사용 가능:
- `answer_token_id`: Prompt 문맥에서 GPT-2가 생성할 token ID
- `distractor_answer_token_ids`: Logit-diff 계산용 rival IDs
- `source_var_token_pos`: Binding line 내 source variable 위치
- `tier`, `role`, `cf_id`, `binding_hop`: 분류/필터링 키

---

## 8. 재현 방법

```bash
# 1. 환경 설정
python3 -m venv .venv
.venv/bin/pip install torch transformers transformer_lens einops plotly
.venv/bin/pip install git+https://github.com/redwoodresearch/Easy-Transformer.git

# 2. 데이터셋 생성
.venv/bin/python -m data.modular
.venv/bin/python -m data.var_binding
.venv/bin/python -m data.ioi_loader

# 3. 검증
.venv/bin/python -m data._validate       # SHA-256 hash 검증
.venv/bin/python -m data._zero_shot_eval # Logit-diff gate
```

---

## 9. 관련 문서

- **이전**: [`00_experiment_overview.md`](./00_experiment_overview.md)
- **다음**: [`02_grokking_training.md`](./02_grokking_training.md)
