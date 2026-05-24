# 분석 트랙 계획 — P1 / P2 / P3

> **역할**: 분석 트랙(P1 grokking, P2 var binding, P3 IOI 비교)의 forward-looking 계획.
> **짝 문서**: [datasets/_report.md](../../datasets/_report.md) — 동결된 데이터 결과 (freeze commit `a3d4c82`).
> **분리 사유**: `_report.md`는 동결된 사실(freeze 후 흔들리지 않음)만, 계획은 분석 진행에 따라 자주 갱신되므로 이 파일에서 관리.

원래 `datasets/_report.md`의 §7 (다음 세션 작업 계획) + §10 (분석 트랙 시작 가이드)을 이쪽으로 이관. _report.md 본문은 §1-6 + §7(재현 방법) + §8(동결 정책)으로 단순화됨.

---

## 1. 다음 세션 작업 계획 (구 §7)

### 1.0 시작 시 체크리스트 (5분)

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

> §1.1, §1.2는 2026-05-24 세션에서 **완료**. §2가 분석 트랙 진입 가이드.

### 1.1 Day 4 — Tier 3 sanity 검토 ✅ (commit `d2c0db4`)

**목표**: 현재 Tier 3 `top1=0%` 한 줄 외에 multi-hop 실패 양상을 한 번 들여다보고, Tier 3 데이터의 쓰임새를 명확히 확정.

**작업 단계**:

1. **실패 양상 샘플링** (`data/var_binding.py` 또는 임시 스크립트):
   - Tier 3 첫 10개 sample에 대해 model forward → top-5 prediction 출력
   - chain 길이별(`binding_hop` 2~4)로 정답 token이 몇 위에 위치하는지 분포 확인
   - 결과를 `datasets/_report.md` §4에 "Tier 3 실패 양상" 서브섹션으로 추가

2. **multi-hop logit-diff 정의 검토**:
   - 후보 A — 정답 vs chain 중간값들의 logit-diff (예: `a=3; b=a; c=b; c=?` → 정답 3, 중간값은 prompt에 추가로 안 나타나므로 distractor 없음)
   - 후보 B — root_val(`names[0]`의 값) vs `ANSWER_VALUES`의 다른 모든 값들 (chance 비교)
   - 어느 쪽이 의미 있는지 결정. 의미가 없다면 Tier 3는 **accuracy-only**로 유지하고 그 점을 명확히 기록.

3. **결론 기록**: `datasets/_report.md` §4 표 업데이트, §5(deviation)에 Tier 3 평가 방식 명시.

**완료 기준**: §4에 Tier 3 실패 양상 한 단락 + 평가 정책 한 줄.

### 1.2 Day 5 — 동결 선언 ✅ (commits `a3d4c82` + `aa8d3d4`)

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

### 1.3 보류 결정 — Prompt format secondary variant

분석 트랙에서 "코드 형식만으로 회로 신호가 너무 약하다" 피드백이 오면, 자연어 wrap variant를 `datasets/var_binding_natural.jsonl`로 추가 생성. GPT-2 small이 `"Variables: a is 3, b is 5, c is a. So c is"`에서 ` 3`을 top-2에 두는 걸 확인했음. **이번 세션 시작 시 분석 트랙 결정을 먼저 듣고** 진행 여부 판단.

### 1.4 참고 — 전체 Plan 파일

세션 컨텍스트 복원이 필요하면:
- `/home/yesulmin/.claude/plans/docs-candidates-sonsj-proposal-md-compressed-wigderson.md` (Day 1~5 전체 플랜, Day 1~3은 완료 부분)
- `git log --oneline 5b9a72a..HEAD` (스캐폴딩 이후 데이터셋 트랙이 만든 모든 commit)

---

## 2. 분석 트랙 시작 가이드 (구 §10)

데이터셋은 freeze됨 (`a3d4c82`). 이제 `docs/candidates/sonsj-proposal.md`의 **Step 1~3 분석 트랙**이 시작점. 세션을 새로 열면 "분석 트랙 시작" 또는 "P1 grokking부터" 같은 식으로 진입.

### 2.0 시작 시 부트스트랩 (5분)

```bash
git pull origin main

# 환경 확인
.venv/bin/python -c "import torch, transformers, transformer_lens, einops, plotly; \
  print('cuda:', torch.cuda.is_available())"

# 동결 가드 — 데이터 변경 없는지 확인 (--strict면 hash mismatch 시 실패)
.venv/bin/python -m data._validate --strict

# zero-shot eval regression
.venv/bin/python -m data._zero_shot_eval
```

3개 모두 깨끗히 통과해야 분석 시작점 정상. 하나라도 실패하면 그 원인부터 추적 (특히 hash mismatch는 데이터 재생성된 신호 → §9.2 절차 위반).

### 2.1 트랙 매핑 — proposal Step ↔ 코드/데이터

| 트랙 | proposal | 사용 데이터 | 주요 산출물 |
|---|---|---|---|
| **P1 Grokking 학습** | Step 1 (Week 1) | `datasets/modular_{train,test}.pt` | 2-layer transformer, grokking loss curve |
| **P1 Grokking 회로 분석** | Step 2 (Week 2) | (P1 학습 결과 모델) | Fourier 분해, attention pattern, ablation |
| **P2 Var Binding 회로** | Step 3-2 (Week 3) | `datasets/var_binding_tier{1,2,3}.jsonl` | Tier 1 cf pair activation patching, binding head 식별 |
| **P3 IOI 비교** | Step 3-3 (Week 3) | `data.ioi_loader` (N=500) | Wang et al. 26 head 목록과 P2 결과 overlap |

각 트랙은 [datasets/_report.md](../../datasets/_report.md) §8.3(불변량)에 박제된 `answer_token_id`, `distractor_answer_token_ids`, `source_var_token_pos`를 그대로 사용. 재토큰화나 prompt 가공 불필요.

### 2.2 가장 자연스러운 진입 순서

1. **P1 학습 먼저** (Step 1, Week 1). modular train으로 2-layer transformer를 scratch 학습 + grokking 곡선 확보. 결과는 `results/grokking/run_*/loss_curve.npz` 같은 형태로 저장 권장.
2. **P1 회로 분석** (Step 2, Week 2). 학습된 모델에서 Fourier basis 검증 + 핵심 head ablation.
3. **P2 var binding** (Step 3-2, Week 3). GPT-2 small + Tier 1 cf pair로 activation patching. **첫 결과 보고 §1.3 (자연어 variant) 필요 여부 판단**.
4. **P3 IOI 비교** (Step 3-3, Week 3). P2 결과 + IOI 회로 overlap 계산.

### 2.3 P2 시작 시 체크할 것 (가장 데이터-tight한 트랙)

- Tier 1 clean 첫 5개 sample로 `model.run_with_cache` 호출해 cache shape 확인
- `source_var_token_pos`로 binding line의 source variable 위치 잡고 그 위치에서 patching
- metric은 `logit[answer] - mean(logit[distractor_answer_token_ids])` (Wang et al. IOI 스타일)
- cf pair는 `cf_id`로 묶임, clean→corrupt patching의 단위

### 2.4 §1.3 결정 게이트

P2 첫 activation patching 결과가 나오면:
- **회로 신호가 명확** (특정 head를 patching 했을 때 logit-diff가 의미 있게 회복) → §1.3 skip
- **신호가 약함** (head 단위로 잡히지 않거나 logit-diff 변화가 noise 수준) → §1.3 자연어 wrap variant 생성 (`datasets/var_binding_natural.jsonl`, GPT-2 small이 더 강한 signal 보이는 형식)

### 2.5 데이터 트랙 잔여 작업 — 없음

분석 트랙 피드백 없는 한 데이터 트랙은 추가 작업 없음. [datasets/_report.md](../../datasets/_report.md) §8.2 절차 거치지 않은 데이터 변경 금지.

---

## 3. 진행 상황 체크리스트 (2026-05-24)

### P1 Grokking 트랙

#### Step 1 — 학습 완료 ✅
- [x] Config 설정 (Nanda canonical: 1-layer, d_model=128, wd=1.0)
- [x] 50K epochs full training (313.8s)
- [x] Grokking point 확인 (epoch 10,200)
- [x] 결과 저장 (`results/grokking/run_20260524_012428/`)
- [x] `_report.md` §4.1에 결과 요약 추가

#### Step 2 — 회로 분석 ✅ 완료
- [x] Fourier basis 분해 (freq 85/28, corr 0.8742)
- [x] Attention pattern 분석 (모든 head 50:50 a:b attend)
- [x] Head ablation study (Head 1: -62%, Head 2: -43%)
- [x] 결과 문서화 (`results/reports/02_grokking_training.md`, `03_circuit_analysis.md`)
- [x] `_summary.md` P1 섹션 업데이트 완료

### P2 Var Binding 트랙

- [ ] GPT-2 small 모델 로드 및 cache shape 확인
- [ ] Tier 1 cf pair activation patching
- [ ] Logit-diff metric 계산 (Wang et al. 스타일)
- [ ] 자연어 variant 필요 여부 판단 (§1.3 결정 게이트)
- [ ] Binding head 식별

### P3 IOI 비교 트랙

- [ ] Wang et al. 26 head 목록 확보
- [ ] P2 결과와의 overlap 계산
- [ ] Cross-track circuit universality 분석

---

## 4. 다음 세션 시작 시 작업 순서

1. **부트스트랩** (§2.0): 환경 확인 + 데이터 무결성 검증
2. **P1 Step 2 재개**: 학습된 모델 불러오기 → Fourier 분석 시작
3. **P2 병행 진행**: P1 분석 대기 시간에 Tier 1 activation patching 시작
4. **결과 업데이트**: 각 트랙 결과 나올 때마다 `_summary.md` 업데이트
