# IOI 회로의 코드 도메인 전이 — 실험 보고서

부제: Mechanistic Interpretability를 통한 Circuit Universality의 다축 분해

작성: 손성준 | 과목: Deep Learning | 2026년 봄학기
브랜치: son | 최종 커밋 기준 정리

---

본 문서는 PPTX 장표 작성을 위한 보고서다. 각 절(##)이 1~2장의 슬라이드에 대응하도록
구성했으며, 모든 수치는 `results/` 하위 JSON/PNG/CSV에서 재현 가능하다.

---

## 1. 연구 질문과 핵심 주장

### 연구 질문
Transformer가 코드의 변수 바인딩(`x=5; z=9; a=z; a=?`)과 자연어의 entity 바인딩
(IOI: "Mary and John ... John gave a drink to ?")을 같은 회로로 처리하는가?

이는 Circuit Universality 가설(Olah 2020, Chughtai et al. 2023)의 modality 간 검증이다.

### 핵심 주장 (One-line)
GPT-2 small의 IOI 회로(Wang et al. 2022)는 코드 변수 바인딩 도메인에서 선택적으로
재사용되며, 재사용되는 출력 단계 head는 기능적 부호(positive/negative)까지 보존된다.
Pythia-160M과 GPT-2 medium 재현 결과, universality는 해부학적 수준(layer/head 좌표)이
아니라 기능적 수준(역할/인과 구조)에서 성립한다.

---

## 2. 배경: Variable Binding과 IOI의 구조적 동일성

### 두 task의 대응 관계

| 항목 | 자연어 IOI (Wang 2022) | 코드 Variable Binding (본 연구) |
|---|---|---|
| 입력 예시 | "Mary and John went to the store. John gave a drink to" | `x=5; z=9; a=z; a=` |
| 정답 | Mary | 9 |
| 추론 구조 | 두 entity 중 binding 관계로 정답 선택 | 두 변수 중 binding 관계로 정답 선택 |
| 방해 요소 | 반복 등장하는 entity (John) | 다른 변수의 값 (5) |

표면 형태는 다르지만 "여러 후보 중 어느 것을 가리키는가"라는 추상 구조가 동일하다.
따라서 두 task의 회로 비교는 universality 검증으로서 의미가 있다.

### IOI 회로의 head 분류 (Wang 2022, 총 26개)

| 분류 | 약어 | 역할 | 단계 |
|---|---|---|---|
| Duplicate Token Head | DTH | 중복 토큰 탐지 | 입력 |
| Previous Token Head | PTH | 직전 토큰 복사 | 입력 |
| Induction Head | IH | 패턴 반복 탐지 | 중간 |
| S-Inhibition Head | SIH | 중복 entity 억제 | 구조 |
| Name Mover Head | NMH | 정답을 출력으로 이동 | 출력 |
| Backup Name Mover | BNMH | NMH 보조 | 출력 |
| Negative Name Mover | NNMH | 정답 logit 억제 | 출력 |

---

## 3. 방법론

### 모델
- GPT-2 small (124M, 12 layer x 12 head = 144 heads) — 주 분석 대상, Wang 2022와 동일
- GPT-2 medium (355M, 24 layer x 16 head = 384 heads) — scale 효과 검증
- Pythia-160M (12 layer x 12 head, rotary embedding, parallel attn+MLP) — 아키텍처 효과 검증
- 2-layer scratch transformer — Grokking 재현 (해석 도구 검증용)

### 데이터셋 (합성, 규칙 기반 생성)
- var_binding Tier 1: counterfactual pair 1,500 (clean/corrupt), 단순 binding
- var_binding Tier 2: 1,000, 더 긴 코드와 다수 distractor
- var_binding Tier 3: 500, 멀티홉 (`a=b; b=c; c=5; a=?`)
- modular addition (p=113): 전체 12,769 pair

### 분석 기법
- Activation Patching: corrupt 입력에 clean의 head 출력(hook_z)을 주입,
  logit_diff 회복량 측정. recovery = (patched_ld - corrupt_ld) / (clean_ld - corrupt_ld)
- Logit Difference: logit(정답) - mean(logit(방해답))
- Direct Logit Attribution: head 출력 잔차를 unembed 방향에 투영
- Head Ablation: head를 zero로 만들고 성능 변화 측정
- Fourier 분해: 학습된 임베딩의 주파수 구조 분석

### 실험 인프라
- H100 80GB x 1, 총 GPU 사용 약 3시간
- 평가 단위: 500 counterfactual pair x 전체 head per-head patching

---

## 4. Step 2 — Grokking 재현 (해석 도구 검증)

### 목적
정답 알고리즘이 알려진 task(modular addition)에서 patching/ablation/Fourier 도구가
올바르게 작동하는지 검증한다. 이것이 통과해야 Step 3 결과를 신뢰할 수 있다.

### 설정
- 2-layer transformer, d_model=128, n_heads=4, d_mlp=512, LayerNorm 없음, bias 없음
- 파라미터 422,784개
- AdamW: lr=1e-3, weight_decay=1.0, betas=(0.9, 0.98), full-batch
- 40,000 step, H100에서 101초

### 결과
- Train accuracy 1.000, Test accuracy 0.9928 (최종)
- Memorization → Generalization 전환(grokking)을 명확히 관찰
- 후반부(step 35,600 부근)에 간헐적 불안정(slingshot, test acc 0.932로 일시 하락)
  존재하나 즉시 회복. 원본 Nanda(100%) 대비 약 0.7%p 낮으나 회로 분석에는 충분

### 회로 분석 발견
- Fourier mass: 학습 진행에 따라 임베딩이 소수 주파수 모드(k 약 14, 31, 35, 52)에 집중
- Head/MLP ablation: 특정 컴포넌트 제거 시 test_loss 급증 → 핵심 회로 식별
- 결론: 도구가 알려진 정답 회로(삼각함수 기반 modular addition)를 정확히 재현

관련 그림: results/analysis/grokking_full/{loss_curve.gif, fourier_final.png, attention_evolution.png}
wandb: https://wandb.ai/sonsj97-plateer/grokking-circuits/runs/n4bnqrak

---

## 5. Step 3 — 코드 바인딩 회로 발견 (GPT-2 small)

### Baseline (Tier 1, n=500)

| 지표 | Clean | Corrupt | 차이 |
|---|---|---|---|
| logit_diff 평균 | +0.182 | +0.097 | +0.086 |
| logit_diff 표준편차 | 1.457 | 1.522 | — |
| Top-1 정확도 | 0.8% | 2.0% | — |

Top-1 정확도는 낮지만 logit_diff 신호는 patching에 충분하다.

### Per-head Activation Patching — Top 5

| 순위 | Head | Recovery | IOI 분류 |
|---|---|---|---|
| 1 | L10H7 | +0.491 | NNMH |
| 2 | L10H2 | +0.453 | BNMH |
| 3 | L6H1 | +0.206 | (비IOI) |
| 4 | L3H0 | +0.159 | DTH |
| 5 | L10H10 | +0.109 | BNMH |

Top 5 중 4개가 Wang 2022의 IOI 26 head에 정확히 해당한다.

관련 그림: results/binding_compare/heatmap_classes.png

---

## 6. Step 3 — IOI Universality 정량화

### Overlap
- Top-26 코드 head ∩ IOI 26 head = 10개
- 랜덤 기대값 = 26 x 26 / 144 ≈ 4.7
- Enrichment = 약 2.1배

### 클래스별 평균 recovery

| IOI 분류 | head 수 | 평균 recovery | 최대 | 전이 여부 |
|---|---|---|---|---|
| NNMH | 2 | +0.283 | +0.491 | 강하게 전이 |
| BNMH | 8 | +0.070 | +0.453 | 전이 |
| NMH | 3 | +0.042 | +0.065 | 전이 |
| DTH | 3 | +0.011 | +0.159 | 부분 전이 |
| IH | 4 | -0.012 | +0.098 | 거의 없음 |
| PTH | 2 | -0.014 | +0.023 | 거의 없음 |
| SIH | 4 | -0.049 | +0.032 | 전이 안 됨 (음수) |
| 비IOI | 118 | +0.003 | — | 기준선 |

### 해석
출력 단계 head(NNMH, BNMH, NMH)는 전이되지만 구조적 단계(SIH)는 전이되지 않는다.
즉 IOI 회로 전체가 옮겨진 것이 아니라 선택적 재사용이 일어났다.

(참고: Spearman rho = +0.106, p=0.61. 클래스 단위 평균은 명확하나
head 단위 순위 상관은 약함 — 본 결론은 클래스 단위 평균에 근거한다.)

관련 그림: results/binding_compare/universality_summary.png

---

## 7. 심화 A — SIH 비전이 검증 (위치 무관성)

### 의문
SIH가 음수인 것이 단순히 패치 위치를 잘못 잡았기 때문일 수 있다.

### 방법
SIH 4개 head(L7H3, L7H9, L8H6, L8H10)를 길이-14 프롬프트의 14개 토큰 위치 각각에서
패치 (n=250).

### 결과

| SIH Head | 절대값 최대 위치 | 값 |
|---|---|---|
| L7H3 | pos 13 (final =) | -0.093 |
| L7H9 | pos 13 | -0.019 |
| L8H6 | pos 13 | -0.204 |
| L8H10 | pos 13 | -0.172 |

모든 위치에서 0 또는 음수다. L8H6는 final 위치에서 -0.20으로, 패치하면 오히려
logit_diff가 감소한다.

### 결론
SIH 비전이는 위치 매칭 실패가 아니라 SIH 메커니즘 자체가 코드 도메인에 부재하기
때문이다. 코드 task에는 IOI의 "중복 entity 억제" 구조가 없으므로 SIH의 존재 이유가
없다.

관련 그림: results/binding_position/sih_position_recovery.png

---

## 8. 심화 B — Direct Logit Attribution (기능적 부호 보존)

### 방법
각 head 출력의 마지막 위치 잔차를 정답-방해답 방향(W_U[정답] - mean(W_U[방해답]))에
투영해 정답 logit_diff에 대한 직접 기여를 측정 (n=500).

### 결과

| Head | IOI 역할 | 직접 기여 | z-score | 부호 일치 |
|---|---|---|---|---|
| L10H2 | BNMH (정답 강화) | +1.010 | +3.3 | 일치 |
| L10H7 | NNMH (정답 억제) | -0.305 | -1.9 | 일치 |
| L3H0 | DTH (간접 경로) | -0.018 | -0.5 | 직접 효과 없음 |
| L9H9 | NMH (정답 강화) | +0.303 | — | 일치 |

전체 head 직접 기여 합계 = +2.139

### 핵심
Patching은 "L10H7이 중요하다"만 말하지만, attribution은 "L10H7이 정답을 깎는다"를
정량화한다. 이는 IOI에서 NNMH(Negative Name Mover)의 역할과 정확히 동일하다.
단순한 활성화 재사용이 아니라 기능적 역할(부호 포함)이 그대로 전이되었다는 증거다.

관련 그림: results/binding_logit_attr/logit_attr_heatmap.png

---

## 9. 심화 F — Minimal Circuit (회로 크기 정량화)

### Baseline
- Clean logit_diff = +0.182
- 모든 attention head 제거(empty) = +0.049
- Attention 전체 기여 = +0.133

### Necessity Test (top-K 제거, 나머지 유지)

| 제거 대상 | head 수 | logit_diff 감소 | 랜덤 K 감소 | 배수 |
|---|---|---|---|---|
| top3 | 3 | +0.054 | +0.031 ± 0.070 | 1.7배 |
| top5 | 5 | +0.051 | +0.005 ± 0.038 | 약 10배 |
| top10 | 10 | +0.065 | +0.035 ± 0.043 | 1.9배 |
| ioi26 | 26 | +0.116 | +0.046 ± 0.054 | 2.5배 |

### 핵심
IOI 26개 head(전체의 18%)를 제거하면 attention 전체 기여의 87%(0.116/0.133)가
사라진다. 코드 바인딩 attention 회로는 사실상 IOI 회로의 부분집합이다.

(참고: Sufficiency test, 즉 top-K만 남기고 나머지를 모두 제거하는 방식은 모든 set에서
실패했다. 144개 중 141개를 zero-ablate하는 것이 out-of-distribution intervention이라
잔차 스트림이 깨지기 때문이다. 따라서 necessity test가 이 경우 더 깨끗한 증거다.)

관련 그림: results/binding_minimal_circuit/necessity.png

---

## 10. 심화 D — Pythia-160M 재현 (아키텍처 효과)

### 설정
같은 12 layer x 12 head이지만 rotary embedding, parallel attn+MLP 구조, Pile 학습
데이터인 Pythia-160M에서 동일 실험을 반복했다.

### 비교

| 측면 | GPT-2 small | Pythia-160M |
|---|---|---|
| Clean / Corrupt / Gap | +0.182 / +0.097 / +0.086 | +0.157 / +0.019 / +0.138 |
| Attention 전체 기여 | +0.133 | +0.145 |
| Top head recovery (최대) | +0.491 (L10H7) | +1.015 (L5H9) |
| 상위 head 우세 layer | L10-11 (깊이 83-92%) | L5-6 (깊이 42-50%) |
| Top-10 head 좌표 overlap | — | 1 / 10 (랜덤 기대 0.7) |
| Necessity top-10 감소 | +0.065 (랜덤 대비 1.9배) | +0.073 (랜덤 대비 약 15배) |

### 핵심
기능적 집중도(소수 head 집중)와 인과적 필요성(top-K가 random보다 중요)은 보존되지만,
layer 위치와 head 좌표는 보존되지 않는다. 같은 task를 같은 원리로 풀지만 어디서
푸는지는 모델마다 다르다.

관련 그림: results/binding_pythia/{pythia_heatmap.png, pythia_necessity.png}

---

## 11. 심화 D2 — GPT-2 Medium 재현 (Scale 효과와 Hydra Effect)

### 설정
같은 GPT family의 더 큰 모델 GPT-2 medium (24 layer x 16 head = 384 heads),
n=500, 71분 소요.

### 3-모델 비교

| 측면 | GPT-2 small (12x12) | GPT-2 medium (24x16) | Pythia-160M (12x12) |
|---|---|---|---|
| Clean / Gap | +0.182 / +0.086 | +0.156 / +0.206 | +0.157 / +0.138 |
| Attention 전체 기여 | +0.133 | +0.136 | +0.145 |
| Top head recovery (최대) | +0.491 (L10H7) | +5.778 (L17H12) | +1.015 (L5H9) |
| Layer peak (절대) | L10-11 | L17-18 | L5-6 |
| Layer peak (상대 깊이) | 83-92% | 71-75% | 42-50% |

### Hydra Effect (신규 발견)
GPT-2 medium에서 top-K head를 제거하면 logit_diff가 오히려 증가한다.

| 제거 대상 | logit_diff 변화 | 랜덤 K |
|---|---|---|
| top3 | -0.031 | +0.001 ± 0.016 |
| top5 | -0.079 | -0.007 ± 0.012 |
| top10 | -0.121 | +0.017 ± 0.044 |
| top30 | -0.011 | +0.009 ± 0.058 |

### 해석
Wang 2022가 IOI에서 보고한 "Backup Name Mover hydra effect"(top NMH 제거 시 backup이
작업을 이어받음)가 더 큰 모델에서 과보상까지 일어난다. Top head에 정답 억제형
suppressor(NNMH 계열)가 포함되어, 이를 제거하면 방해답 억제가 풀리며 정답 logit이
상대적으로 커진다.

### 함의
모델이 클수록 회로 redundancy가 증가한다. 144 head 모델에서는 18% 제거로 87% 손실이
났지만, 384 head 모델에서는 2.6% 제거가 오히려 성능을 높인다. 큰 모델 해석에는 단순
zero-ablation이 부족하며 path patching 같은 정교한 도구가 필요하다.

관련 그림: results/binding_gpt2med/{gpt2med_heatmap.png, layer_profile_comparison.png}

---

## 12. 종합 — Universality의 4단계 분해

| 축 | GPT-2 small | GPT-2 medium | Pythia-160M | 보존 여부 |
|---|---|---|---|---|
| 1. 기능적 집중도 (소수 head 집중) | 보존 | 보존 | 보존 | 전부 보존 |
| 2. 인과적 필요성 (top-K > random) | 1.9배 | hydra | 15배 | 전부 보존 |
| 3. Layer 단계 위치 | 83-92% | 71-75% | 42-50% | family 내 보존, family 간 불일치 |
| 4. Head 좌표 (정확한 L,H) | — | (width 다름) | 1/10 | 보존 안 됨 (랜덤 수준) |

### 결론
Universality는 architecture 표면(layer 위치, head 좌표)이 아니라 functional
decomposition 수준(역할, 인과 구조)에서 성립한다. 같은 원리, 다른 해부학적 위치.

---

## 13. 기여 (Contribution)

1. 자연어-코드 cross-domain 회로 재사용을 정량화한 첫 시도
   (Feng & Steinhardt 2023은 binding 메커니즘은 분석했으나 IOI 회로와의 head-level
   매핑은 다루지 않음)
2. Selective circuit reuse 패턴 발견 및 검증 (출력 head 전이, SIH 비전이)
3. 기능적 부호 보존의 정량 증명 (NNMH의 정답 억제 역할이 코드에서도 -0.31, z=-1.9)
4. Head와 position grammar의 분리 (DTH가 IOI와 다른 위치에서 작동)
5. 회로 크기 정량화 (IOI 26 head가 attention 기여의 87% 담당)
6. Cross-model universality의 4단계 분해 (functional 보존, anatomical 비보존)
7. Scale-dependent hydra effect 발견 (GPT-2 medium에서 ablation이 성능 향상)

특히 3, 7번은 본 연구가 처음 정량화한 결과다.

---

## 14. 한계와 향후 과제

- Hydra effect의 메커니즘 미규명: 어떤 backup head가 활성화되는지 path patching 필요
- Tier 3 멀티홉 분석 부분 미완: distractor 정의 불가로 logit_diff 메트릭 적용 어려움
- 더 큰 모델/다른 family (Llama, Mistral)로의 확장 시 framework 일반화 가능
- Causal scrubbing 등 완전한 sufficiency 검증 미수행

### 본 연구가 답할 수 없는 것 (정직한 범위 설정)
- "LLM이 진짜 reasoning을 하는가" 같은 큰 질문은 다루지 않는다
- GPT-4 규모 모델로의 일반화는 compute 한계로 검증하지 못했다
- 답할 수 있는 것: GPT-2/Pythia 규모에서 variable binding이 어디서 어떻게 처리되며
  IOI와 어떤 관계인가, 그리고 그 관계가 모델 간에 어떻게 변하는가

---

## 15. 산출물 요약

| 구분 | 내용 |
|---|---|
| 분석 스크립트 | src/grokking (5), src/binding (12) = 총 17개 |
| 시각화 | PNG 17개 이상, GIF 2개, wandb 차트 5개 |
| 결과 데이터 | JSON 8개, raw npy 5개, CSV 2개 |
| 문서 | docs/results.md (상세), docs/slides.md (Marp), 본 보고서 |
| 학습 로그 | wandb run n4bnqrak (공개) |
| 실험 규모 | 모델 3종, H100 약 3시간, 500 pair 평가 |

### 핵심 그림 (장표 삽입 권장)
- results/analysis/grokking_full/loss_curve.gif — grokking 전환
- results/binding_compare/heatmap_classes.png — patching + IOI 클래스
- results/binding_compare/universality_summary.png — 클래스별 전이
- results/binding_logit_attr/logit_attr_heatmap.png — 기능적 부호
- results/binding_minimal_circuit/necessity.png — 회로 크기
- results/binding_gpt2med/layer_profile_comparison.png — 3-모델 layer 분포
- results/binding_compare/circuit_diagram.png — 회로 schematic

---

## 참고문헌

- Nanda et al. 2023, Progress Measures for Grokking via Mechanistic Interpretability, ICLR
- Wang et al. 2022, Interpretability in the Wild: a Circuit for IOI in GPT-2 small, ICLR
- Chughtai et al. 2023, A Toy Model of Universality, ICML
- Feng & Steinhardt 2023, How do Language Models Bind Entities in Context
- Elhage et al. 2021, A Mathematical Framework for Transformer Circuits, Anthropic
- Bereska & Gavves 2024, Mechanistic Interpretability for Transformer-Based LMs (Review)
