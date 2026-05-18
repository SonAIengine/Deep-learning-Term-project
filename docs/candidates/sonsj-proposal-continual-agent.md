# Term Project 제안: Self-Evolving Personal Agent — 매일 진화하는 작은 LLM, Continual Learning으로 가능한가?

> **분야**: Continual Learning + Personal AI + Tool Calling
> **제안자**: 손성준
> **작성일**: 2026-05-12
> **관련 후보**: 1차 제안 `sonsj-proposal.md` (Grokking + Code Variable Binding)과 narrative 시너지 — 1차는 "학습 mechanism", 본 제안은 "학습 dynamics in production"

---

## 주제

> **"본인이 매일 Claude Code를 사용하며 자동 수집되는 데이터 stream으로, 작은 LLM(1.5B)이 catastrophic forgetting 없이 점점 더 본인 스타일에 specialize되는 self-evolving personal agent를 4주 동안 실제로 작동시킨다."**

본 프로젝트는 단순 fine-tuning이 아닌 **production-level continual learning system**을 구축한다. 본인이 이미 운영 중인 자동 데이터 수집 파이프라인(MinIO + HuggingFace 자동 push, 4대 호스트 sync)을 활용하여, 매일 새로 들어오는 trajectory 데이터로 모델이 **실제로 매일 점진적으로 진화**한다. 4주 동안의 evolution을 측정하고, 4가지 continual learning 전략을 비교하여 **personal domain에 최적인 전략**을 찾는다.

본 분야는 2025-2026의 가장 hot한 sub-area 중 하나로, **EvolveR** (arXiv 2510.16079), **MIT SDFT (Self-Distillation Fine-Tuning)**, **Adaptive Minds** (2025.10), **Letta Continual Learning SDK** (2026.02) 등이 활발히 발표되고 있다. 학계 진단:

> "Catastrophic forgetting이 academic curiosity에서 **production engineering challenge**로 이동.
> Session-scoped chatbot → persistent multi-month service로 agent가 진화 중."
> — Zylos Research, 2026.04

본 프로젝트는 이 학계 흐름의 **학생 수준 실제 production case study**.

### 19개 기존 후보 + 1차 후보와의 차별점

* 19개 후보: 모두 응용 앱 (정적 모델 활용).
* 1차 후보 (MI): 작은 toy transformer scratch 학습 + 회로 분석 (static).
* **본 제안**: **시간 축 도입 — 매일 진화하는 시스템 구축 + 4가지 전략 비교 ablation**.
* 1차와의 narrative 시너지: 1차는 "**학습 mechanism**" (회로), 본 제안은 "**학습 dynamics in production**" (시간에 따른 진화) → 두 후보가 학습의 mechanism + dynamics 양쪽 cover.

---

## 배경: Continual Learning과 Self-Evolving Agent란?

### Continual Learning이란?

**한 줄 정의**: 모델이 새 데이터를 받을 때마다 점진적으로 학습하면서도 이전 능력을 잃지 않는 패러다임.

기존 fine-tuning vs Continual Learning:

```
[기존 Fine-tuning]
한 번에 모든 데이터로 학습 → 끝
            ↓
        Static model

[Continual Learning]
Day 1 데이터 → 학습 → Day 1 model
                          ↓
Day 2 데이터 → 학습 → Day 2 model (Day 1 능력 유지)
                          ↓
Day N 데이터 → 학습 → Day N model
                          ↓
        Evolving model
```

### Catastrophic Forgetting 문제

Continual Learning의 핵심 도전: **새 데이터 학습 시 이전 능력 손실**.

```
[Naive incremental fine-tuning]
Day 0: 코딩 잘함 (HumanEval 60%)
Day 1: 본인 task만 학습
Day 7: 본인 task 잘함 + 코딩 잊음 (HumanEval 30%)
       ↑ Catastrophic forgetting!
```

해결 방법들 (각각 학계 active research):
1. **Memory Replay**: 과거 데이터 sample을 재학습
2. **Elastic Weight Consolidation (EWC)**: 중요 weight 보호
3. **LoRA Stacking (O-LoRA)**: 매일 새 LoRA, orthogonal subspace
4. **Self-Distillation (MIT SDFT)**: 자기 자신 응답으로 distill

### Self-Evolving Agent란?

EvolveR (arXiv 2510.16079) 정의:
> **"Agent가 자율적으로 dynamic task에 적응하고, cross-task 지식을 통합하고, 외부 개입 없이 성능을 sustain하는 능력"**

본 프로젝트는 이 정의에 정확히 부합:
- **자율 적응**: 매일 새 trajectory로 자동 학습
- **Cross-task 통합**: 다양한 본인 task 누적
- **외부 개입 없이**: cron 자동화

---

## 핵심 연구 질문

> **"매일 자동 누적되는 personal data stream에서, 작은 LLM(1.5B)이 catastrophic forgetting 없이 점점 더 본인 스타일에 specialize될 수 있는가? 어떤 continual learning 전략이 personal domain에 최적인가?"**

### 가설들

* **H1 (Continual Improvement)**: 4주 동안 personal task accuracy가 매주 monotonic 증가 (Day 0 50% → Day 28 80%)
* **H2 (Catastrophic Forgetting)**: Naive incremental은 general capability 손실 (HumanEval 60% → 30%), 우리 전략은 보존 (≥55%)
* **H3 (Strategy Comparison)**: 4가지 전략 (Naive / LoRA Stack / Replay / Self-Distill) 중 명확한 winner 존재
* **H4 (Speed-Specialization Trade-off)**: 빠른 specialization과 느린 forgetting 사이 trade-off 정량 측정

### 가능한 결론 시나리오

**Case A — "Continual Learning Works"**
H1~H3 모두 성립. 명확한 winning 전략 발견.
→ "Personal data stream으로 작은 모델의 self-evolution 가능 입증"

**Case B — "Forgetting is Severe"**
H1 부분 성립, H2 실패.
→ "Personal specialization과 general capability의 trade-off 정량화" (학술적 의미)

**Case C — "No Significant Improvement"**
일일 데이터 너무 적어서 학습 효과 미미.
→ "Low-data continual learning의 한계 정의" (negative result의 학술적 가치)

**세 결과 모두 publishable급 발견**. 실패 없는 실험 디자인.

---

## 대략적인 계획

### Step 1: Base Model + 4가지 전략 Implementation (Week 1)

#### 1-1. Base Model 셋업
* **Model**: Qwen2.5-Coder-1.5B-Instruct
* **초기 fine-tuning**: 2026-05-01 이전 본인 데이터 ~2,000 페어로 Day 0 model 생성
* **QLoRA 4-bit**: RTX 4090 단일 GPU

#### 1-2. 4가지 Continual Learning 전략 구현

```python
# Strategy A: Naive Incremental
def update_naive(model, new_data):
    model.train(new_data, epochs=1)
    return model

# Strategy B: LoRA Stacking (O-LoRA style)
def update_lora_stack(base_model, new_data, day):
    new_lora = train_lora(base_model, new_data, orthogonal_to=previous_loras)
    return compose([base_model, *previous_loras, new_lora])

# Strategy C: Memory Replay (SuRe style)
def update_replay(model, new_data, history):
    surprising_history = select_surprising(history, n=50)
    combined = new_data + surprising_history
    model.train(combined, epochs=1)
    return model

# Strategy D: Self-Distillation (MIT SDFT style)
def update_sdft(model, new_data):
    pseudo_labels = model.generate(new_data.inputs)
    model.train(new_data + pseudo_labels, epochs=1)
    return model
```

### Step 2: Daily Update Pipeline 구축 (Week 2)

#### 2-1. Cron 자동화
- 기존: 매일 03:00 데이터 ETL + HF push (이미 작동 중)
- 추가: 매일 03:30 모델 update cron
- 4가지 전략 모델 동시 update + 저장

#### 2-2. 자동 평가 시스템
매일 04:00 평가 cron:
```
[Personal Task Test Set] (불변, 50개)
  → Personal accuracy 측정
[BFCL Subset] (200개)
  → General tool calling 능력
[HumanEval] (164개)
  → 코딩 능력 (forgetting 측정)
[Trajectory Efficiency]
  → 평균 step 수
```

평가 결과는 자동으로 SQLite 또는 parquet에 저장 → 시각화용.

### Step 3: 4주 Evolution 실행 + 분석 (Week 3-4 진행 중)

```
Day 0:  4 전략 모두 동일 base (1차 fine-tuned)
Day 1~28: 매일 30~50 페어 새 데이터로 4 전략 각각 update
          매일 평가
Day 28: 4 전략 비교 + evolution curve 분석
```

### Step 4: 시각화 + Live Demo (Week 4)

- **Evolution curve**: Day별 personal task accuracy 추이 (4 전략 비교)
- **Forgetting curve**: General capability 추이
- **Strategy Pareto frontier**: Personal vs General 균형
- **Live demo**:
  - Day 28 model vs Day 0 model 비교 (동일 task)
  - 새 trajectory 입력 → 즉시 update → 동일 task 재시도 → 개선 확인
- **Evolution timelapse**: Day별 결과 변화 GIF

---

## 실제 활용 데이터셋 + 인프라

### 본인 데이터 수집 인프라 (이미 운영 중)

```
[4대 호스트]
Mac (launchd 10분) ┐
홈서버 (cron 10분) ├─→ MinIO Bronze (원본 JSONL)
회사 서버 ×2      ┘
                      ↓ 매일 03:00 cron
                  MinIO Silver (Parquet, 일별 partition)
                      ↓
              Gold (PII masked + ChatML)
                      ↓
              HuggingFace Push (sft / guppy / dpo / pretrain)
                      ↓
        [신규 추가] 03:30 Model Update Cron
                      ↓
                Day N+1 Personalized Model
```

### 현재 데이터 (2026-05-12)
- **sft**: 2,388 페어 (Day 0 train용)
- **guppy**: 7,352 페어
- **pretrain**: 2,388 페어
- **dpo**: 179 페어
- **HF**: `SuperPowerMz/claude-code-personal-v1` (private)
- **일일 증가량**: 약 30~50 페어 (4주 동안 +800~1,400 페어 예상)

### Test 벤치마크

* **Personal Task Test Set**: 본인 Day 0 이전 데이터에서 50개 hold-out
* **BFCL Subset** (Berkeley Function Calling Leaderboard): general tool calling 200개
* **HumanEval**: 코딩 능력 손상 측정 164개
* **Trajectory Length 측정**: 각 task별 평균 step 수

### 모델

* **Base**: Qwen2.5-Coder-1.5B-Instruct (HuggingFace)
* **PEFT**: QLoRA 4-bit
* **Compute**: RTX 4090 단일 GPU

### 데이터셋 선정 이유

* **자동 production stream**: 분야 #1 challenge ("real-world data scarcity")에 unique 해결책
* **시계열 dynamics**: 시간 축이 자연스럽게 존재하는 데이터
* **PII 마스킹 완료**: 자동으로 안전한 데이터
* **인프라 이미 완비**: 학생 프로젝트가 production 수준 활용

---

## 예상 결과

### 1. 정량적 결과

```text
[Personal Task Accuracy Evolution]
                Day 0   Day 7   Day 14  Day 21  Day 28
Strategy A:     50%     58%     55%     48%     42%   ← Naive, forgetting
Strategy B:     50%     62%     71%     76%     80%   ← LoRA Stack ✓
Strategy C:     50%     60%     68%     72%     75%   ← Replay ✓
Strategy D:     50%     55%     63%     66%     68%   ← SDFT

[General Capability (HumanEval) — Forgetting]
                Day 0   Day 7   Day 14  Day 21  Day 28
Strategy A:     60%     45%     35%     30%     25%   ← Catastrophic
Strategy B:     60%     58%     57%     56%     55%   ← Preserved ✓
Strategy C:     60%     59%     58%     56%     55%   ← Preserved ✓
Strategy D:     60%     58%     57%     55%     54%   ← Preserved ✓

[Trajectory Length (효율성)]
                Day 0   Day 14  Day 28
Strategy A:     5.2     4.8     5.0  (변화 없음)
Strategy B:     5.2     3.8     2.8  (단축 ✓)
Strategy C:     5.2     4.0     3.2
Strategy D:     5.2     4.5     3.8
```

### 2. 시각화 결과물

* **Evolution Curve** (Personal task accuracy over 28 days, 4 전략 overlay)
* **Forgetting Curve** (HumanEval over 28 days)
* **Pareto Frontier**: Personal vs General (전략별 점)
* **Trajectory Length Histogram** Day 0 vs Day 28
* **Live Demo 영상**:
  - Day 0 model vs Day 28 model 동일 task 비교
  - 새 trajectory → 즉시 update → 개선 확인 30초 영상

### 3. 발표 임팩트 포인트

* **"이 모델은 발표 당일에도 더 좋아져 있습니다"** — 실시간 진화
* **Evolution timelapse**: 28일간 성능 변화 GIF
* **Live update demo**: 새 데이터 입력 즉시 학습 영상
* **"Day 0 → Day 28: Personal 50% → 80%, General 60% → 55%"** 한 줄 결론
* **Strategy 비교 Pareto**: "LoRA Stack이 최적, Naive는 catastrophic forgetting"

---

## 이 실험으로 알게 되는 것

### Layer 1: 실험이 직접 답하는 질문

**"매일 자동 누적되는 personal data로 작은 LLM이 catastrophic forgetting 없이 점진적으로 self-improve할 수 있는가?"**

세 시나리오 (A/B/C) 중 어느 것이 성립하는지 명확한 답 + 4 전략 중 winning strategy 정확한 식별.

### Layer 2: 구체적으로 손에 잡히는 발견

1. **Evolution curve**: Day별 personal task 성능 변화 (4 전략 비교)
2. **Forgetting curve**: 전략별 general capability 손실 정도
3. **Strategy 비교**: Personal vs General trade-off에서 최적 전략
4. **Daily data 효과**: 30~50 페어/일 incremental 학습의 실제 효과
5. **Trajectory evolution**: 효율성이 시간에 따라 어떻게 변하는지
6. **Compute cost**: 일일 update의 실제 시간·자원 측정

→ **누구도 갖지 못한 "production stream + 4 전략 비교" 데이터셋**.

### Layer 3: 학계·실용 측면 함의

**학계 측면**:
* EvolveR, MIT SDFT 등 학계 framework의 **personal real-world stream 첫 적용**
* O-LoRA, SuRe, MIT SDFT 등 4 전략의 **head-to-head 비교** (학계에서도 아직 부족)
* "Low-data daily continual learning" sub-area에 데이터 포인트 추가
* Catastrophic forgetting의 **production-level 정량 측정**

**실용 측면**:
* **개인화 코딩 비서의 가능성**: 매일 진화하는 personal assistant
* **Edge device 활용**: 1.5B 모델로 로컬 deployment 가능
* **Privacy**: Personal data가 외부로 안 나감
* **Production-ready 아키텍처**: 다른 사용자도 본인 데이터로 적용 가능한 framework

### Layer 4: 한계 (정직하게)

본 실험이 **답할 수 없는 것**:
* **장기 효과**: 4주는 짧음. 6개월, 1년 진화는 어떨까?
* **다른 사용자 일반화**: 한 사람 데이터 결과가 다른 사람에게도?
* **모델 크기 확장**: 7B, 70B에서도 같은 패턴?
* **다양한 도메인**: 코딩 외 task에서도?

답할 수 있는 것: **"한 학생의 28일간 Claude Code 데이터로 1.5B 모델의 4가지 continual learning 전략 비교"** 까지.

→ 작은 발견이지만 **명확하고 측정 가능하고 production-validated된 결과**.

---

## 강의 개념 연결

| 강의 개념 | 본 프로젝트에서 다루는 방식 |
|----------|---------------------------|
| Transformer architecture | Qwen2.5-Coder 활용 |
| Fine-tuning / Transfer Learning | QLoRA로 효율적 fine-tuning |
| LoRA (Low-Rank Adaptation) | 4가지 전략 중 핵심 도구 |
| Catastrophic Forgetting | Continual learning의 핵심 문제 |
| Memory & Knowledge Retention | Replay, EWC 등 기법 |
| Self-Distillation | MIT SDFT 전략 |
| Sequence / Trajectory modeling | Tool calling trajectory 학습 |
| Production ML / System Design | 자동 update pipeline, monitoring |
| 연구 방법론 | 가설 → ablation → 정량 비교 |

---

## 교수님 평가 기준 대응

| 예상 질문 | 답변 |
|----------|------|
| 왜 이 모델? | Qwen2.5-Coder-1.5B는 학생 컴퓨트로 가능하고 코딩·한국어 친화. 1.5B는 catastrophic forgetting이 충분히 발생할 만한 크기이면서 4 전략을 모두 4주 안에 비교 가능. |
| 왜 이 데이터셋? | **본인 production stream**은 분야 #1 challenge에 unique 해결책. 자동 수집·PII 마스킹 인프라 1년 운영 중. 학생 프로젝트에서 production-level 데이터 활용은 매우 드뭄. |
| 왜 이 평가지표? | Personal accuracy + General capability + Trajectory efficiency 3축 평가. Continual learning의 핵심 trade-off (specialization vs forgetting)를 정량 측정. |
| 왜 4가지 전략? | EvolveR, MIT SDFT, Adaptive Minds 등 2025-2026 핵심 방법론들. **Head-to-head 비교는 학계에서도 아직 부족**. |
| Daily update의 의미? | "Static fine-tuning vs Continual Learning"의 본질적 차이. 4주 evolution이 실제 production agent의 mini case study. |
| 결과 해석 논리? | Day별 measurement → Evolution curve → Strategy 간 ablation → Personal/General trade-off → Optimal strategy 결정. **시계열 데이터의 자연스러운 분석 흐름**. |
| 일반화? | 정직히 단일 사용자 28일 결과 — single-subject case study로 framing. 하지만 **자동화된 framework 자체는 다른 사용자에게도 적용 가능** (future work). |

---

## 위험 요소 + 대응

| 위험 | 가능성 | 대응 |
|------|--------|------|
| 일일 데이터 너무 적음 (30~50개) | 중간 | **그것 자체가 발견** ("low-data continual learning 한계 정의"), back-test로 일년치 데이터 simulate 추가 가능 |
| Catastrophic forgetting 너무 심해서 모든 전략 실패 | 낮음 | 학계 기법들 적용 → 부분 성공 보장 |
| 4 전략 동시 운영 메모리 부담 | 중간 | 4 모델을 시간 분할 학습 (순차 update), 또는 LoRA만 저장 |
| Test set leakage | 중간 | Day 0 이전 hold-out + 외부 benchmark (BFCL, HumanEval) |
| 시간 부족 (Step 2 pipeline 구축 지연) | 중간 | Day 0 model + naive evaluation으로 부분 결과 확보, 4 전략은 일부만 |
| GPU OOM (4 model 동시) | 중간 | 4-bit 양자화 + LoRA만 별도 저장, full model 1개만 메모리 보유 |
| 일일 cron 실패 | 중간 | 실패 시 알림 + 재시도 cron, 누락 일자 일괄 처리 옵션 |

**핵심 안전망**: Step 1~2 (Day 0 base + 자동 pipeline 구축) 만 완성해도 **system 자체가 결과물**. Step 3 (4주 evolution) 결과가 부족해도 framework 데모로 점수 확보.

---

## 1차 후보 (MI)와의 관계 (강화된 narrative)

| 차원 | 1차 (MI) | 본 제안 (Continual Agent) |
|------|---------|--------------------------|
| 모델 | 작은 toy + GPT-2 small | Production-scale (Qwen2.5-Coder-1.5B) |
| 데이터 | 합성 (modular arithmetic, code binding) | **본인 real-world production stream** |
| 시간 축 | Static (한 번 학습) | **Dynamic (28일 evolution)** |
| Method | 회로 발견 + ablation | Continual learning + 4 전략 비교 |
| 결 | 분석·mechanism | 시스템·dynamics |
| 발표 narrative | "Transformer 내부 동작 reverse-engineering" | **"매일 진화하는 personal 비서"** |
| 공통 메시지 | **"학습 과정 분석"** | **"학습 dynamics 분석"** |

**통합 narrative** (2 후보 함께 채택 시):
> "1차는 transformer가 알고리즘을 학습하는 **internal mechanism**을 분석합니다.
> 2차는 LLM이 실제 production 환경에서 시간에 따라 진화하는 **external dynamics**를 측정합니다.
> 두 후보가 함께 **'학습'의 두 측면 (mechanism + dynamics)** 을 cover합니다."

→ Voting 시 강력한 일관성 메시지.

---

## 4주 일정 (구체화)

| Week | 작업 | 산출물 |
|------|------|--------|
| **1** | Base model (Day 0) fine-tuning + 4 전략 구현 + 평가 set 구축 | 4개 baseline model + 자동 평가 cron |
| **2** | Daily update pipeline 구축 (cron 03:30) + 평가 자동화 (cron 04:00) | Production-ready 시스템 |
| **3** | 4주 evolution 시작 (Day 1~14) + 매일 monitoring | Day 14 시점 evolution curve |
| **4** | Day 15~28 + 최종 분석 + 시각화 + Live demo + 발표 자료 | 최종 결과물 + 발표 |

**핵심 시점**:
- Week 2 끝: 시스템 자동 작동 시작
- Week 4 끝: 28일 evolution 완성 (실제로는 system이 시작된 시점부터 28일)

---

## 참조

### 핵심 페이퍼

* [EvolveR: Self-Evolving LLM Agents through Experience-Driven Lifecycle (arXiv 2510.16079, 2025.10)](https://arxiv.org/html/2510.16079v1) — 본 제안 직접 기반
* [Self-Evolving LLMs via Continual Instruction Tuning (arXiv 2509.18133)](https://arxiv.org/html/2509.18133v3)
* [MIT SDFT — Self-Distillation Fine-Tuning](https://venturebeat.com/orchestration/mits-new-fine-tuning-method-lets-llms-learn-new-skills-without-losing-old) — Strategy D 기반
* [Adaptive Minds (2025.10)](https://www.emergentmind.com/topics/lifelong-learning-for-llm-based-agents) — LoRA adapter routing
* [O-LoRA: Orthogonal Continual Learning](https://aclanthology.org/2025.findings-naacl.303/) — Strategy B 기반
* [Continual Learning of LLMs: A Comprehensive Survey (CSUR 2025)](https://dl.acm.org/doi/10.1145/3735633)
* [Continual Learning in LLMs: Methods, Challenges, Opportunities (arXiv 2603.12658)](https://arxiv.org/html/2603.12658v1)
* [AgentFly: Fine-tuning LLM Agents without Fine-tuning LLMs (arXiv 2508.16153)](https://arxiv.org/html/2508.16153v1)

### Tool Calling + Trajectory 관련

* [Amazon — Small LMs Outperform Large in Tool Calling (arXiv 2512.15943)](https://arxiv.org/abs/2512.15943)
* [TRAJECT-Bench (arXiv 2510.04550)](https://arxiv.org/html/2510.04550v1)
* [Beyond the Final Answer (arXiv 2510.02837)](https://arxiv.org/abs/2510.02837)
* [BFCL: Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)

### 라이브러리 + 모델

* [Qwen2.5-Coder-1.5B-Instruct (HuggingFace)](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct)
* [QLoRA / PEFT](https://github.com/huggingface/peft)
* [TRL (Transformer RL, DPO 지원)](https://github.com/huggingface/trl)
* [Letta Continual Learning SDK (2026.02)](https://www.letta.com/blog/continual-learning) — 참고 architecture

### 본인 인프라 (이미 운영 중)

* **데이터셋**: HuggingFace `SuperPowerMz/claude-code-personal-v1` (private, 자동 push 매일 03:00)
* **수집 인프라**:
  * 4대 호스트 (Mac launchd / 홈서버 / 회사 서버 ×2) → MinIO Bronze sync
  * 홈서버 cron 03:00 ETL (Bronze → Silver → Gold)
  * PII 마스킹 + ChatML 변환
  * HuggingFace 자동 push
* **2026-05-12 기준 데이터**: sft 2,388 / guppy 7,352 / pretrain 2,388 / dpo 179, 매일 +30~50 페어 증가

### 분야 위상 (제안 정당화)

* [Continual Learning AI Agents (Zylos Research 2026.04)](https://zylos.ai/research/2026-04-09-continual-learning-catastrophic-forgetting-ai-agents) — "academic curiosity → production challenge" 진단
* [ICLR 2026 Trends: Agentic AI](https://encord.com/iclr-2026/) — "Agent" 키워드 1위
* [Letta Continual Learning SDK 출시 (2026.02)](https://www.letta.com/blog/continual-learning) — Production-level continual learning 등장
