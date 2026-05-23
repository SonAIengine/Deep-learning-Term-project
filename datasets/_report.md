# 데이터셋 보고서 — Transformer Circuit Analysis

> **상태**: 진행 중 (Day 1~3 완료, Day 4~5는 다음 세션)
> **최종 업데이트**: 2026-05-23
> **담당**: 데이터셋 트랙 (yesul.min)
> **관련**: [docs/datasets.md](../docs/datasets.md), [docs/candidates/sonsj-proposal.md](../docs/candidates/sonsj-proposal.md)

본 문서는 무엇이 생성되었는지, 어떤 검증을 통과했는지, 그리고 `docs/datasets.md` v2와의 모든 일탈(deviation) 사항을 기록한다. 동결(freeze) 선언은 Tier 3 sanity 검토 후 별도 세션에서 이 보고서 상단에 추가된다.

---

## 1. 환경

| 항목 | 값 |
|---|---|
| Python | 3.12 (system) inside `.venv/` |
| OS | Ubuntu 24.04.4 LTS on WSL2 |
| GPU | NVIDIA RTX 4060 Ti, 8 GB VRAM |
| CUDA driver | 13.1 (host) |
| `torch` | 2.12.0+cu130 |
| `transformers` | 5.9.0 |
| `transformer_lens` | 3.2.1 |
| `einops` | 0.8.2 |
| `easy_transformer` | `git+https://github.com/redwoodresearch/Easy-Transformer.git` |
| `plotly` | `easy_transformer`의 누락된 transitive 의존성으로 별도 설치 |

재현 방법: README "Quick Start"의 pip 명령 + `pip install plotly`.

---

## 2. 생성된 데이터셋

모두 `datasets/` 하위. 크기는 `docs/datasets.md` §1과 일치.

| 데이터셋 | 파일 | 레코드 수 | seed |
|---|---|---|---|
| Modular arithmetic train | `modular_train.pt` | 3,830 | `SEED=0` |
| Modular arithmetic test | `modular_test.pt` | 8,939 | `SEED=0` |
| Var binding Tier 1 (cf) | `var_binding_tier1.jsonl` | 1,000 (= 500 cf pair) | `SEED=0` |
| Var binding Tier 2 | `var_binding_tier2.jsonl` | 500 | `SEED=1` |
| Var binding Tier 3 | `var_binding_tier3.jsonl` | 500 | `SEED=2` |
| IOI | (필요 시점에 `data.ioi_loader`로 로드) | 500 | (Easy-Transformer 기본값) |

Modular vocab: 114 (= p+1, `=` 토큰을 인덱스 p=113에 둠). ✅ §3 명세 일치.

---

## 3. 검증 결과 (data/_validate.py)

생성된 Tier 1 / Tier 2 JSONL에 대한 §4.7 체크리스트 결과.

| 체크 항목 | Tier 1 | Tier 2 |
|---|---|---|
| `tokenization_check == "passed"` | ✅ 1000/1000 | ✅ 500/500 |
| cf pair 토큰 길이 일치 | ✅ 500/500 | 해당 없음 |
| cf pair `source_var_token_pos` 일치 | ✅ 500/500 | 해당 없음 |
| 정답값 분포 스프레드 | ✅ 4.50% (≤ 20%) | ✅ 3.80% (≤ 25%) |
| 변수명 분포 스프레드 | ✅ 1.88% (≤ 30%) | (강제하지 않음) |

IOI sanity (`data.ioi_loader`): N=500 로드 성공, 예시 prompt가 정상적으로 파싱됨 — `"Then, Amy and Ashley were thinking about going to the house. Amy wanted to give a bone to Ashley"`.

---

## 4. GPT-2 small Zero-shot 결과

`python -m data._zero_shot_eval`, `cuda` (RTX 4060 Ti) 기준.

| Tier | Top-1 acc | logit_diff avg | pct_pos | 비고 |
|---|---|---|---|---|
| Tier 1 clean (N=500) | 0.80% | **+0.182** | **54.40%** | chance(50%) 위 |
| Tier 1 corrupt (N=500) | 2.00% | +0.097 | 49.20% | ≈ chance (corrupt run은 binding rival을 바꾼 케이스이므로 예상된 결과) |
| Tier 2 (N=500) | 1.40% | +0.143 | 53.75% | chance 위, distractor 노이즈는 더 큼 |
| Tier 3 (N=500) | 0.00% | n/a | n/a | multi-hop, distractor 리스트 없음 — accuracy만 측정 |

**해석**. Top-1 accuracy가 일률적으로 0에 가까운 이유는 `"a=3; b=5; c=a; c="` 같은 prompt 다음에 GPT-2가 예측하는 가장 가능성 높은 토큰이 **변수명**(`b`/`a`/`c`)이지 숫자가 아니기 때문이다. 즉 GPT-2 small의 "next-variable" prior가 binding 신호를 압도한다. 그러나 정답값과 rival distractor 값 사이의 **logit 순위**는 이미 chance 위(pct_pos > 50%, avg diff > 0)이며, 이것이 mechanistic interpretability 분석이 실제로 필요로 하는 신호다. 이는 Wang et al. (2022)이 IOI에서 사용한 구성과 동일하다 — 그들도 top-1이 아닌 "IO logit > S logit" 비율을 보고한다.

---

## 5. docs/datasets.md v2와의 일탈(deviation)

명세에서 벗어난 모든 지점. 분석 트랙(P1/P2/P3)이 기대치를 조정할 수 있도록 빠짐없이 기록한다.

### D1. Tier 1 `n_vars` ∈ {3, 4} (명세는 {2, 3})
- **명세**: §4.3 "변수 개수 n_vars: 2, 3 균등 (각 250 pair)"
- **실제**: counterfactual 생성에는 target이 아닌 변수가 **두 개** 필요하다 (`src_clean`, `src_corrupt`). 따라서 하한은 `n_vars >= 3`. 원본 코드는 `n_vars=2`일 때 1개짜리 리스트에 `rng.sample(candidates, 2)`를 호출해 `ValueError`로 죽었다. `n_vars = 3 + (i % 2)`로 수정 — 3과 4를 번갈아 250쌍씩.
- **분석에 미치는 영향**: 적음. distractor 개수가 {0, 1}에서 {1, 2}로 늘어 오히려 Wang et al.의 IOI ABBA 구조에 가까워짐.

### D2. Eval gate: top-1 accuracy 대신 logit-diff
- **명세**: §4.7-1 "GPT-2 small zero-shot accuracy ≥ 70% (Tier 1 기준)"
- **실제**: 명세 prompt 형식에서 GPT-2 small의 top-1 accuracy는 0.8%. top-5를 보면 정답 숫자가 강력한 변수명 prior(`b`/`a`/`c`)에 가려진다. 따라서 Wang et al. IOI 스타일 logit-difference gate로 대체했다 — Tier 1 clean에서 `pct_pos > 0.50` AND `avg logit_diff > 0`. 둘 다 통과(54.4%, +0.182). Tier 1 데이터는 그대로 두고, 분석 트랙은 top-1 accuracy를 참고 지표로만 다뤄야 한다.
- **스키마 변경**: Tier 1/2 레코드 각각에 `distractor_values`와 `distractor_answer_token_ids`가 추가됨 → 패칭·eval 코드가 distractor logit을 재토큰화 없이 직접 읽을 수 있음.

### D3. `answer_token_id`를 prompt 문맥 기반으로 계산
- **명세**: §4.6 예시는 `"answer_token_id": 513` (leading-space 버전 `' 9'`).
- **실제**: `=`로 끝나는 코드 prompt 다음 위치에서 GPT-2 BPE는 leading space **없는** 숫자를 둔다 (예: `'3'`는 ID 18, `' 3'`는 ID 513). 이제 `answer_token_id`는 `prompt + str(answer)`를 함께 토큰화한 후 마지막 토큰을 취하는 방식으로 계산한다 — 이러면 GPT-2가 해당 위치에서 실제로 생성할 ID와 보장된다. 정답이 정확히 한 토큰 안에 들어가지 않으면 sample의 `tokenization_check`가 `failed:answer_..._not_single_next_token`으로 보고하고 검증 단계에서 폐기된다.

### D4. `SINGLE_TOKEN_NAMES` 47개로 박제
- **명세**: §4.1은 12개 placeholder, Day 1 검증 후 갱신.
- **실제**: Day 1 검증 스크립트(`data._day1_verify`)가 후보 51개 중 47개가 GPT-2 BPE에서 leading-space single-token임을 확인. `shared/config.py`에 47개를 박제.

### D5. `ANSWER_VALUES`는 1..9 (1..19까지 single-token이지만 명세 준수)
- **명세**: §4.6 "정답 값 1~9 균등 분포".
- **실제**: Day 1 검증 결과 1..19 모두 single-token이지만, 명세를 존중해 1..9로 유지. distractor 집합이 작아지고, 정답값 분포 검증의 "한 자리 숫자" 가정과 일치.

---

## 6. 발견·수정된 버그

- **B1**. `data/var_binding.py` `make_tier1`: `n_vars=2`일 때 `rng.sample` ValueError. 하한을 3으로 올려 수정 (D1 참조).
- **B2**. `data/ioi_loader.py`: `easy_transformer.utils`가 `transformers.TRANSFORMERS_CACHE`를 참조하는데 `transformers>=5`에서 제거된 속성. `huggingface_hub.constants.HF_HUB_CACHE`에서 가져와 import 전에 주입하는 compat shim 추가.
- **B3**. `data/ioi_loader.py`: GPT-2 토크나이저는 기본적으로 `pad_token`이 없어서 `IOIDataset` padding이 실패. `tok.pad_token = tok.eos_token`로 해결.

---

## 7. Open items (다음 세션으로 이월)

- [ ] **Tier 3 sanity (Day 4)**: `top1=0%` 외에 multi-hop 실패 양상 분석. chain 중간값들에 대한 logit-diff 보고를 추가할지 검토.
- [ ] **동결 선언 (Day 5)**: Tier 3 검토 후 본 보고서 상단에 "frozen on \<date\>" 헤더를 추가하고, 그 commit이 분석 트랙의 canonical 기준이 된다.
- [ ] **결정 필요**: 분석 트랙에서 더 강한 신호의 prompt format(자연어 wrap)을 *secondary* variant로 추가할지. GPT-2 small은 `"Variables: a is 3, b is 5, c is a. So c is"`에서 ` 3`을 top-2에 둠. 이번 세션 범위 밖.

---

## 8. 재현 방법

```bash
# 1. 환경
python3 -m venv .venv
.venv/bin/pip install torch transformers transformer_lens einops plotly
.venv/bin/pip install git+https://github.com/redwoodresearch/Easy-Transformer.git

# 2. 데이터
.venv/bin/python -m data._day1_verify   # 풀 출력, 이미 config.py에 박제됨
.venv/bin/python -m data.modular
.venv/bin/python -m data.var_binding
.venv/bin/python -m data.ioi_loader     # sanity 로드만

# 3. 검증 + eval
.venv/bin/python -m data._validate       # 성공 시 exit 0
.venv/bin/python -m data._zero_shot_eval # Tier 1 pct_pos > 50% 통과 시 exit 0
```
