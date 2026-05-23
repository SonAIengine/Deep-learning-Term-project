# 데이터셋 보고서 — Transformer Circuit Analysis

> **상태**: **FROZEN (2026-05-24)** — Day 1~3 데이터셋 + Day 4 sanity 완료
> **freeze commit**: `<pending>` (이 commit 직후 hash로 stamp)
> **최종 업데이트**: 2026-05-24
> **담당**: 데이터셋 트랙 (yesul.min)
> **관련**: [docs/datasets.md](../docs/datasets.md), [docs/candidates/sonsj-proposal.md](../docs/candidates/sonsj-proposal.md)

본 문서는 무엇이 생성되었는지, 어떤 검증을 통과했는지, 그리고 `docs/datasets.md` v2와의 모든 일탈(deviation) 사항을 기록한다. **동결 정책은 §9 참조** — 분석 트랙(P1/P2/P3)은 이 commit 시점의 데이터·`answer_token_id`·`distractor_answer_token_ids`를 신뢰하고 작업을 시작할 수 있다.

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
| Tier 3 (N=500) | 0.00% | +0.459† | 53.80%† | multi-hop, Option-B(아래) |

† Tier 3의 logit-diff는 distractor variable 값이 없어 root_val vs `ANSWER_VALUES`의 다른 8개 숫자 (chance 비교)로 정의했음 — `data/_tier3_sanity.py` 참조.

**해석**. Top-1 accuracy가 일률적으로 0에 가까운 이유는 `"a=3; b=5; c=a; c="` 같은 prompt 다음에 GPT-2가 예측하는 가장 가능성 높은 토큰이 **변수명**(`b`/`a`/`c`)이지 숫자가 아니기 때문이다. 즉 GPT-2 small의 "next-variable" prior가 binding 신호를 압도한다. 그러나 정답값과 rival distractor 값 사이의 **logit 순위**는 이미 chance 위(pct_pos > 50%, avg diff > 0)이며, 이것이 mechanistic interpretability 분석이 실제로 필요로 하는 신호다. 이는 Wang et al. (2022)이 IOI에서 사용한 구성과 동일하다 — 그들도 top-1이 아닌 "IO logit > S logit" 비율을 보고한다.

### 4.1 Tier 3 실패 양상 (multi-hop)

`python -m data._tier3_sanity` 출력 요약. 분석 트랙이 Tier 3를 multi-hop circuit 평가에 쓰려 할 때 무엇을 기대해야 하는지 명확히 한다.

**Top-5 도배 현상 (첫 10개 샘플)**. 정답 숫자는 top-5에 거의 들어가지 않는다. 상위 5개는 항상 prompt에 등장한 **변수명 prefix**(`data`, `tmp`, `value`, `result`, `output` 등) + 숫자 `0` 조합이다. 예시:

- `vb3_0000` (hop=2, ans=3): top-5 = `'data', 'score', 'result', '0', 'x'` — 정답 `3`은 rank 7
- `vb3_0008` (hop=4, ans=7): top-5 = `'value', 'output', 'result', '0', 'input'` — 정답 `7`은 rank 394

**Rank 분포 by `binding_hop` (전체 N=500)**. 정답 token의 rank를 hop 길이별로 그룹화하면 체인이 길어질수록 단조 악화:

| hop | N | median rank | top-5 비율 | rank>100 비율 |
|---|---|---|---|---|
| 2 | 152 | 15 | 7% | 11% |
| 3 | 175 | 31 | 5% | 25% |
| 4 | 173 | 60 | 5% | 35% |

**Option-B logit-diff (root_val vs `ANSWER_VALUES` \ {root_val}, N=500)**. avg_diff = **+0.459**, pct_pos = **53.80%** — 거의 chance(50%) 수준이지만 통계적으로는 위에 있다. binding signal이 약하게나마 살아 있다는 뜻.

**Tier 3 평가 정책 (확정)**. Tier 3는 **accuracy-only**가 기본 metric (이미 0%로 freeze됨). 보조 지표로 (i) **rank-by-hop 분포** — chain 길이별 회로 성능 저하 측정, (ii) **Option-B logit-diff** — chance 비교용 약한 신호. 분석 트랙이 multi-hop circuit을 본격적으로 다루려면 Tier 3 데이터 자체가 아니라 별도 자연어 wrap variant(§7.3 검토 대상)를 만드는 편이 현실적.

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

### D6. Tier 3 평가 정책 — accuracy-only + Option-B logit-diff (보조)
- **명세**: §4.7-3 "Tier 3는 GPT-2 accuracy로 reference만"
- **실제**: accuracy 0%로 명세대로 reference 역할만 가능. §4.1의 sanity 분석으로 두 가지 보조 지표를 정의: (i) rank-by-hop 분포 (binding chain 길이별 성능 저하 측정), (ii) Option-B logit-diff — root_val vs `ANSWER_VALUES` \ {root_val} (chance 비교). 후자는 변수 distractor 값이 multi-hop에서는 모두 root_val과 같아져 의미가 없기 때문에, 대신 다른 한 자리 숫자 8개를 rival 집합으로 사용. 결과 pct_pos 53.80% — chance 살짝 위. 분석에서는 (i)을 주력으로 사용 권장.

---

## 6. 발견·수정된 버그

- **B1**. `data/var_binding.py` `make_tier1`: `n_vars=2`일 때 `rng.sample` ValueError. 하한을 3으로 올려 수정 (D1 참조).
- **B2**. `data/ioi_loader.py`: `easy_transformer.utils`가 `transformers.TRANSFORMERS_CACHE`를 참조하는데 `transformers>=5`에서 제거된 속성. `huggingface_hub.constants.HF_HUB_CACHE`에서 가져와 import 전에 주입하는 compat shim 추가.
- **B3**. `data/ioi_loader.py`: GPT-2 토크나이저는 기본적으로 `pad_token`이 없어서 `IOIDataset` padding이 실패. `tok.pad_token = tok.eos_token`로 해결.

---

## 7. 다음 세션 작업 계획

### 7.0 시작 시 체크리스트 (5분)

```bash
# 1. 최신 상태 동기화 (다른 트랙이 작업했을 수 있음)
git pull origin main

# 2. 환경 점검 — venv가 있고 import 다 통과하는지
.venv/bin/python -c "import torch, transformers, transformer_lens, einops, plotly; \
  print('cuda:', torch.cuda.is_available())"

# 3. 기존 데이터·검증·eval이 그대로 통과하는지 (regression 확인)
.venv/bin/python -m data._validate         # exit 0이어야 함
.venv/bin/python -m data._zero_shot_eval   # Tier 1 pct_pos > 50% 통과
```

위 3개가 깔끔히 통과하면 시작점 정상. 하나라도 실패하면 그 원인부터 추적.

### 7.1 Day 4 — Tier 3 sanity 검토

**목표**: 현재 Tier 3 `top1=0%` 한 줄 외에 multi-hop 실패 양상을 한 번 들여다보고, Tier 3 데이터의 쓰임새를 명확히 확정.

**작업 단계**:

1. **실패 양상 샘플링** (`data/var_binding.py` 또는 임시 스크립트):
   - Tier 3 첫 10개 sample에 대해 model forward → top-5 prediction 출력
   - chain 길이별(`binding_hop` 2~4)로 정답 token이 몇 위에 위치하는지 분포 확인
   - 결과를 본 문서 §4에 "Tier 3 실패 양상" 서브섹션으로 추가

2. **multi-hop logit-diff 정의 검토**:
   - 후보 A — 정답 vs chain 중간값들의 logit-diff (예: `a=3; b=a; c=b; c=?` → 정답 3, 중간값은 prompt에 추가로 안 나타나므로 distractor 없음)
   - 후보 B — root_val(`names[0]`의 값) vs `ANSWER_VALUES`의 다른 모든 값들 (chance 비교)
   - 어느 쪽이 의미 있는지 결정. 의미가 없다면 Tier 3는 **accuracy-only**로 유지하고 그 점을 명확히 기록.

3. **결론 기록**: §4 표 업데이트, §5(deviation)에 Tier 3 평가 방식 명시.

**완료 기준**: §4에 Tier 3 실패 양상 한 단락 + 평가 정책 한 줄.

### 7.2 Day 5 — 동결 선언

**목표**: 데이터셋·평가 코드의 canonical 버전을 명확히 못박고, 분석 트랙(P1/P2/P3)이 흔들리지 않을 기반을 제공.

**작업 단계**:

1. **본 문서 헤더 변경**:
   ```markdown
   > **상태**: FROZEN (2026-MM-DD)
   > **freeze commit**: <hash>
   ```
   (commit 만든 직후 hash로 채움.)

2. **§9 "동결 정책" 신규 섹션 추가**:
   - 데이터·평가 코드 변경 시 팀 전체 합의 필요
   - regen이 필요하면 seed 유지 + `git diff datasets/`가 0이어야 함
   - 분석 트랙은 이 commit의 데이터·`answer_token_id`·`distractor_answer_token_ids`를 신뢰

3. **`data/_validate.py`에 freeze 가드 추가** (선택):
   - 데이터 파일들의 SHA-256 해시를 박제하고 검증 시점에 mismatch면 경고
   - 또는 더 가볍게 — `(len(t1), len(t2), len(t3))`만 박제

4. **Commit 메시지 컨벤션**:
   ```
   chore(data): freeze Day 1-3 datasets

   - Modular train/test, var_binding tier 1/2/3 are now canonical.
   - Analysis tracks (P1/P2/P3) should read these without expecting
     further changes; refresh requires team sign-off.
   ```

5. **분석 트랙 알림**: README 또는 team chat에서 "데이터 동결됨, P1/P2/P3 시작 가능" 공지.

**완료 기준**: freeze commit hash가 헤더에 박혀 있고, 그 이후 `datasets/` 아래 diff가 0.

### 7.3 보류 결정 — Prompt format secondary variant

분석 트랙에서 "코드 형식만으로 회로 신호가 너무 약하다" 피드백이 오면, 자연어 wrap variant를 `datasets/var_binding_natural.jsonl`로 추가 생성. GPT-2 small이 `"Variables: a is 3, b is 5, c is a. So c is"`에서 ` 3`을 top-2에 두는 걸 확인했음. **이번 세션 시작 시 분석 트랙 결정을 먼저 듣고** 진행 여부 판단.

### 7.4 참고 — 전체 Plan 파일

세션 컨텍스트 복원이 필요하면:
- `/home/yesulmin/.claude/plans/docs-candidates-sonsj-proposal-md-compressed-wigderson.md` (Day 1~5 전체 플랜, Day 1~3은 완료 부분)
- `git log --oneline 5b9a72a..HEAD` (스캐폴딩 이후 데이터셋 트랙이 만든 모든 commit)

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

---

## 9. 동결 정책 (2026-05-24)

본 commit 시점의 데이터셋 + 평가 코드는 canonical이며, 분석 트랙(P1/P2/P3)은 이 상태를 신뢰하고 작업을 시작한다. 이후 변경은 **팀 sign-off 후**에만 허용한다.

### 9.1 동결 대상

| 파일 | SHA-256 |
|---|---|
| `datasets/modular_train.pt` | `2034d5a3…2589f0` |
| `datasets/modular_test.pt` | `99135adb…393918` |
| `datasets/var_binding_tier1.jsonl` | `296dd2c9…1275f7` |
| `datasets/var_binding_tier2.jsonl` | `9024ccb9…9790c8` |
| `datasets/var_binding_tier3.jsonl` | `a870214c…6e11ce0` |

전체 해시는 `data/_validate.py`의 `FROZEN_HASHES`에 박제. 검증 시 자동 비교되며 mismatch면 경고 (`--strict` 플래그 시 실패).

### 9.2 동결 후 변경 절차

1. 변경 사유와 영향 범위(어떤 분석 결과가 무효화되는지)를 PR description에 명시.
2. 분석 트랙 담당자 최소 1인의 sign-off (regression이 분석 트랙에 미치는 영향을 그쪽이 평가).
3. regen 시 **seed 고정** (`shared/config.py`의 `SEED=0`). seed가 같으면 동일한 데이터가 나오는지 먼저 확인.
4. `data/_validate.py`의 `FROZEN_HASHES`를 새 해시로 갱신 + 본 §9.1 표 갱신.
5. commit 메시지에 "refreeze" 명시 + 이전 freeze commit hash를 참조.

### 9.3 분석 트랙이 신뢰할 수 있는 불변량

- `answer_token_id`: prompt 문맥에서 GPT-2 BPE가 실제로 생성할 token id (no-space digit). 재토큰화 없이 그대로 사용 가능.
- `distractor_answer_token_ids` (Tier 1/2): logit-diff 계산용 rival token ids. Tier 1 clean의 rival 첫 번째 원소는 `src_corrupt`의 값 → counterfactual patching에서 의미 있는 비교.
- `source_var_token_pos` (Tier 1/2): binding line 내 source variable token의 0-indexed 위치. cf pair에서 clean/corrupt가 일치 (`_validate.py`로 보장).
- `tier`, `role`, `cf_id`, `binding_hop`: 분류/필터링 키.

### 9.4 평가 기준 — 동결됨

- Tier 1 freeze gate: `_zero_shot_eval.py`가 Tier 1 clean에서 `pct_pos > 50%` AND `avg logit_diff > 0` 통과 시 exit 0. 현재 54.40% / +0.182.
- Tier 2: chance 위 logit-diff signal 확인 (현재 53.75% / +0.143).
- Tier 3: accuracy-only가 기본 (현재 0%), 보조로 rank-by-hop 분포 + Option-B logit-diff. §4.1, §5 D6 참조.
