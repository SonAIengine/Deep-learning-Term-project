# Term Project 제안: Transformer Circuit Analysis — Grokking과 Code Variable Binding

> **분야**: Mechanistic Interpretability (MI)
> **제안자**: 손성준
> **작성일**: 2026-05-12

---

## 주제

> **"작은 트랜스포머가 알고리즘을 학습하는 순간을 reverse-engineering하고, 코드와 자연어가 같은 회로로 처리되는지 검증한다"**

본 프로젝트는 두 가지 축으로 진행된다:

1. **축 A — Grokking 회로 분석 (Scratch 학습)**: Modular arithmetic 같은 단순 합성 task를 transformer에 scratch로 학습시키고, **학습 중 어느 순간 모델이 외우기(memorization)에서 일반화(generalization)로 전환되는 'grokking' 현상**의 내부 회로를 분석한다.

2. **축 B — Code Variable Binding 회로 발견 (Pretrained 모델)**: GPT-2 small에서 **코드의 변수 추적(`a=3; c=a; c=?`)을 처리하는 회로를 발견**하고, Wang et al. (2022)이 발견한 자연어 entity 추적(IOI) 회로와 비교한다.

본 분야는 **Mechanistic Interpretability**로, 2026년 1월 MIT Tech Review의 "10 Breakthrough Technologies 2026"에 선정되었으며, Anthropic CEO Dario Amodei가 2025년 4월 "The Urgency of Interpretability" 에세이를 통해 회사의 2027년 핵심 목표로 선언한 영역이다. NeurIPS/ICLR/ICML 2024-2026에 전용 워크샵이 연속 개최되고 있다.


## 배경: Variable Binding이란?

**한 줄 정의**: "변수 이름이 어떤 값을 가리키는지" 추적하는 것.

### 코드에서의 Variable Binding

```python
a = 3      # 변수 a에 값 3을 bind
b = 5      # 변수 b에 값 5를 bind
c = a      # 변수 c에 a의 값(=3)을 bind
print(c)   # 답: 3
```

마지막 줄에서 `c`가 무엇인지 알려면, 모델은 `c = a` 와 `a = 3` 두 줄을 따라가서 정답 `3`을 출력해야 한다. **이 일련의 과정을 attention head들이 협업해서 처리해야 한다.**

### 자연어에서의 동일 구조: IOI (Indirect Object Identification)

Wang et al. (2022)이 GPT-2에서 분석한 task:

```
"Mary and John went to the store. John gave a drink to ___"
                                                          ↑
                                                       정답: Mary
```

여기서도 모델은 두 entity (`Mary`, `John`) 중 binding 관계를 추적하여 정답을 선택해야 한다.

| 자연어 (IOI) | 코드 (Variable Binding) |
|-------------|------------------------|
| Mary, John이라는 entity | a, b, c라는 변수 |
| "John gave a drink to" 다음 entity 예측 | "c =" 다음 값 예측 |
| 여러 entity 중 binding 관계로 정답 선택 | 여러 변수 중 binding 관계로 정답 선택 |

**구조가 동일하다.** 그래서 비교 실험이 의미 있다.

---

## 핵심 연구 질문

> **"Transformer가 코드의 variable binding과 자연어의 entity binding을 같은 회로로 처리하는가, 다른 회로로 처리하는가?"**

이는 최근 학계에서 활발히 논의되는 **Circuit Universality** 가설 (Olah 2020, Chughtai et al. 2023) — 다른 task·modality에서 동일한 회로가 형성되는가 — 의 modality 간 첫 검증이다.

두 가지 가능한 결론:

### Case A — 같은 회로 사용 (Universality 성립)

**발견**: GPT-2의 동일한 attention head들이 자연어 task와 코드 task 양쪽에서 동일한 역할을 수행한다.

**무슨 뜻인가**:
모델이 `"Mary vs John 중 정답 고르기"`와 `"변수 a vs 변수 b 중 정답 고르기"`를 같은 메커니즘으로 처리한다는 의미. 즉, 트랜스포머가 **표면 형태(영어 단어 vs 파이썬 코드)에 관계없이 추상적인 "binding 알고리즘"을 학습**했다는 증거.

**일상 비유**:
사람이 "철수는 학생이다 → 학생"이라는 문장과 "x = 5 → 5"라는 코드를 같은 추론 회로로 처리하는 것과 비슷. 표면 형태는 다르지만 본질적으로 "이것은 무엇을 가리키는가"를 묻는 동일한 작업이기 때문.

**실용적 함의**:
* 코드 fine-tuning이 자연어 능력에도 영향 (회로가 공유되므로).
* Transfer learning이 잘 작동.
* 모델이 단순 패턴 매칭이 아닌 **"진짜로 이해"하는 쪽에 가깝다**는 증거.

---

### Case B — 다른 회로 사용 (Modality-specific)

**발견**: 자연어용 binding head와 코드용 binding head가 명확히 분리되어 있다.

**무슨 뜻인가**:
모델이 자연어 binding 회로와 코드 binding 회로를 **별도로 학습**했다는 의미. 즉, 트랜스포머가 **표면 형태에 따라 specialized된 회로를 task별로 따로 만들어** 처리한다는 증거.

**일상 비유**:
사람이 모국어 문법은 brain의 한 영역에서, 외국어는 다른 영역에서 처리하는 것과 비슷. 같은 "binding" 작업처럼 보여도 뇌가 서로 다른 모듈을 사용.

**실용적 함의**:
* 코드 fine-tuning은 코드 회로만 영향 (자연어 능력 보존됨).
* 각 modality를 따로 점검해야 함 (안전·정렬 검증 측면).
* 모델이 **추상적 이해보다는 task별 specialization**에 가까울 가능성.

---

**두 결과 모두 학술적·실용적 가치 있음**:
* Case A → "트랜스포머의 일반화·추상화 능력 증거"
* Case B → "Modality-specific specialization 발견"

**실패 없는 실험 디자인** — 어느 쪽 결과가 나와도 명확한 발견.

---

## 대략적인 계획

### Step 1: 환경 셋업 + Grokking 베이스라인 재현 (Week 1)

* TransformerLens 라이브러리 셋업.
* Nanda et al. (2023) Grokking 페이퍼 코드 재현.
* 2-layer transformer를 modular addition $(a + b) \mod 113$ 에 학습.
  * 가능한 $(a, b)$ pair 중 30%만 train set으로 사용.
  * 학습 epoch별 train/test loss 추적.
* **베이스라인 grokking 곡선 확보**. 본 단계까지는 공식 코드 활용 → 재현 안전성 보장.

### Step 2: Grokking 회로 분석 (Week 2)

* 학습된 모델의 attention weight + embedding 시각화.
* Embedding과 Output projection에 대한 Fourier 분해 분석.
* 발견된 회로 (sin/cos 기반 modular addition 알고리즘) 검증.
* 학습 중 회로가 형성되는 시점 추적.
* **Ablation 실험**: 특정 attention head·MLP를 끄고 task 성능 변화 측정.

### Step 3: Code Variable Binding 회로 발견 + IOI와 비교 (Week 3)

#### 3-1. 데이터 구성

합성 코드 데이터 자동 생성:
```python
# 예시 1 (단순)
a = 3; b = 5; c = a; c = ?   # 정답: 3

# 예시 2 (변수 4개)
apple = 3; banana = 5; cherry = 7; durian = 11
mango = apple; mango = ?      # 정답: 3
```

* 변수 개수: 2 ~ 6개
* 변수 이름: 다양한 길이·형태
* 약 1000 ~ 5000개 sample

#### 3-2. GPT-2 small에서 회로 발견

* TransformerLens로 attention pattern 분석
* **Activation patching**으로 binding을 담당하는 head 식별
* Ablation으로 인과적 중요도 검증

#### 3-3. IOI 회로와 비교

* Wang et al. (2022)에서 발견된 26개 IOI attention head 목록을 reference로 사용
* 두 task의 회로 overlap 측정
* **Circuit Universality score** 정량화:
  * 같은 head가 양쪽 task에서 활성화되는 비율
  * 각 head의 역할이 일치하는가

### Step 4: 시각화 + 발표 준비 (Week 4)

* Loss curve 애니메이션 (matplotlib + FuncAnimation + ffmpeg).
* Attention pattern 학습 진행 GIF (epoch 별 snapshot).
* Fourier 분해 시각화 (모델이 학습한 trigonometric basis).
* **회로 다이어그램 나란히 비교**: IOI 회로 vs Code Binding 회로.
* **Causal ablation 영상**: "이 head를 끄면 두 task 모두 실패하는가, 한쪽만 실패하는가" 라이브 데모.
* (옵션) Streamlit 인터랙티브 데모 페이지.
* 발표 영상 편집 + 최종 리포트.

---

## 실제 활용 데이터셋

### 주 데이터셋

* **합성 데이터 — Modular arithmetic** (Step 1~2)
  * 직접 코드로 생성, 데이터 확보 문제 없음.
  * 구성: $(a, b) \to (a + b) \mod p$ 형태 pair, $p = 113$ 권장.
  * 전체 $p^2 = 12769$개 pair 중 30%만 train, 나머지 test.

* **합성 데이터 — Code Variable Binding** (Step 3)
  * Python 스타일 변수 할당·참조 코드, 규칙 기반 자동 생성.
  * 변수 개수, 이름 다양성 controlled.
  * 약 1000 ~ 5000개 sample.

* **IOI 데이터셋** (Step 3 비교 baseline)
  * Wang et al. (2022) 공개 데이터, 또는 직접 생성.
  * "Mary and John..." 패턴 약 500개.

### 보조 모델

* **GPT-2 small (124M params)**, HuggingFace 공개
  * Step 3 회로 분석 대상 모델. 너무 크지 않아 단일 GPU로 분석 가능.

### 데이터셋 선정 이유

* **합성 데이터 사용은 의도된 선택**:
  * 정답 알고리즘이 알려진 controlled task → 발견한 회로가 정말 그 알고리즘인지 검증 가능.
  * 데이터 수집·라벨링 부담 0.
  * Reproducibility 확보.

---

## 예상 결과

### 1. 정량적 결과

```text
[Step 1~2: Modular addition task (p=113)]
  - Train loss → 0 at epoch ~1,000
  - Test loss → 0 at epoch ~20,000 (grokking 순간)
  - 발견된 회로: 2개 attention head + MLP가 Fourier basis 활용
  - Ablation: L0H0 제거 시 test accuracy 100% → ~10%

[Step 3: Code Variable Binding (GPT-2 small)]
  - Variable binding task accuracy
  - 발견된 회로의 head 개수·layer 분포
  - Ablation effect: 각 head별 task 성능 감소량

[Step 3: IOI vs Code Binding 비교]
  - Overlap score: 양쪽에서 공통으로 중요한 head 비율
  - Circuit universality score (정량 지표 직접 정의)
  - Layer별 회로 위치 비교
```

### 2. 시각화 결과물

* **Grokking loss curve 애니메이션** (mp4 / GIF)
* **Attention pattern 학습 진행 GIF** (epoch 0 → 50,000)
* **Fourier 분해 시각화** (학습된 sin/cos basis)
* **회로 다이어그램 나란히 비교** (IOI vs Code Binding)
* **Causal ablation 비디오** ("이 head를 끄면 두 task 모두 실패합니다")
* (옵션) **Streamlit 인터랙티브 데모**

### 3. 발표 임팩트 포인트

* "트랜스포머가 외우기에서 알고리즘 깨달음으로 전환하는 순간"의 라이브 시각화.
* "자연어 IOI 회로와 코드 binding 회로가 얼마나 겹치는가" 직관적 비교 시각.
* "특정 attention head 하나를 끄니 양쪽 task가 동시에 망가지는" 인과 데모.

---

## 이 실험으로 알게 되는 것

### Layer 1: 실험이 직접 답하는 질문

**"Transformer가 코드 변수 추적과 자연어 entity 추적을 같은 회로로 처리하는가, 다른 회로로 처리하는가?"**

* **Case A — 같은 회로 사용**: Transformer는 modality에 관계없이 추상적 binding 알고리즘을 학습.
* **Case B — 다른 회로 사용**: Transformer는 코드와 자연어에 서로 다른 binding 메커니즘을 학습.

두 결과 모두 흥미롭고 의미 있음.

### Layer 2: 구체적으로 손에 잡히는 발견

실험 종료 시점에 **확정적으로 알게 되는 것들**:

1. **회로 위치**: GPT-2 small에서 code binding이 정확히 어느 layer·어느 head에서 일어나는지
2. **회로 크기**: 몇 개 head가 협업하는지 (IOI는 26개였음)
3. **인과적 중요도**: head별 ablation effect size 정량 수치
4. **형성 시점**: 학습 중 언제 회로가 형성되는지 (grokking 관점)
5. **Scaling**: 변수 개수 2개 → 5개로 늘어나면 같은 회로가 작동하는지

→ 이는 **재현 불가능한 새로운 데이터**. 누가 똑같은 실험을 안 했다면 본 팀이 처음 측정한 값.

### Layer 3: 학계·실용 측면 함의

**학계 측면**:
* **Circuit Universality 가설**에 새 데이터 포인트 추가.
* Modality 간 (code vs NL) universality 검증은 거의 안 됨 → 빈칸 채움.

**실용 측면**:
* LLM (Copilot, Cursor 등)의 코드 이해 메커니즘 이해의 기초.
* Code-specific fine-tuning이 자연어 능력에 미치는 영향 예측 가능 (회로가 같으면 양쪽 다 영향).
* 코드 모델의 실패 case 진단 (어떤 head 손상이 어떤 실패 유발).

### Layer 4: 한계 (정직하게)

본 실험이 **답할 수 없는 것들**:

* "LLM이 진짜 reasoning을 하는가" — 너무 큰 질문.
* "코드 능력이 인간 프로그래머처럼 작동하는가" — 모델이 너무 작음.
* "이 회로가 큰 모델(GPT-4 등)에서도 같은가" — compute 한계로 검증 못 함.

답할 수 있는 것: **"GPT-2 small 수준에서 variable binding은 어디서 어떻게 처리되는가, IOI와 같은가"** 까지.

→ 작은 발견이지만 명확하고 측정 가능한 발견.

---

## 강의 개념 연결

| 강의 개념 | 본 프로젝트에서 다루는 방식 |
|----------|---------------------------|
| Transformer architecture | Scratch 학습으로 직접 구현하여 깊이 이해 |
| Attention mechanism | 시각화 + ablation으로 실제 동작 검증 |
| Generalization vs memorization | Grokking 현상으로 학습 dynamics 직접 관찰 |
| Loss landscape | Train/test loss 분리 추적 |
| Causal inference | 모델 컴포넌트 ablation으로 인과 관계 검증 |
| Pretrained model 활용 | GPT-2 small에서 회로 발견 (Step 3) |
| 연구 방법론 | 가설 → 실험 → 결과 해석의 논리적 흐름 |

---

## 교수님 평가 기준 대응

회의록에서 정리한 교수님 예상 질문에 대한 답변:

| 예상 질문 | 답변 |
|----------|------|
| 왜 이 모델? | 작은 transformer는 회로가 깔끔하게 발견되는 가장 단순한 setup. GPT-2 small은 IOI 회로(Wang et al. 2022) 비교 기준이 있어 reference 활용 가능. 큰 모델은 분석 자체가 불가능. |
| 왜 이 데이터셋? | 합성 데이터는 **정답 알고리즘이 알려진 controlled task** → 발견한 회로가 정말 그 알고리즘인지 검증 가능. Reproducibility 보장. Code/NL 비교는 modality 간 universality 검증. |
| 왜 이 평가지표? | Accuracy 외에 **Circuit overlap score**, **head ablation effect size**, **Fourier basis fitness** 등 인과·구조 검증 지표 활용. 단순 성능이 아닌 메커니즘 검증. |
| 결과 해석 논리? | 학습 dynamics 관찰 → 회로 형성 가설 → ablation으로 인과 검증 → modality 간 비교로 universality 검증. **각 단계가 다음 단계의 근거**. |

---

## 참조

### 핵심 페이퍼

* [Nanda et al., "Progress Measures for Grokking via Mechanistic Interpretability" (ICLR 2023, arXiv:2301.05217)](https://arxiv.org/abs/2301.05217)
* [Wang et al., "Interpretability in the Wild: a Circuit for IOI in GPT-2 small" (ICLR 2023, arXiv:2211.00593)](https://arxiv.org/abs/2211.00593) ← Step 3 핵심 reference
* [Elhage et al., "A Mathematical Framework for Transformer Circuits" (Anthropic, 2021)](https://transformer-circuits.pub/2021/framework/index.html)
* [Olsson et al., "In-context Learning and Induction Heads" (Anthropic, 2022)](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)
* [Chughtai et al., "A Toy Model of Universality" (ICML 2023)](https://arxiv.org/abs/2302.03025) ← Circuit Universality 핵심
* [Bereska & Gavves, "A Practical Review of Mechanistic Interpretability for Transformer-Based LMs" (2024, arXiv:2407.02646)](https://arxiv.org/abs/2407.02646)

### 라이브러리 + 코드

* [TransformerLens (GitHub)](https://github.com/TransformerLensOrg/TransformerLens)
* [Grokking 공식 코드 (Nanda)](https://github.com/mechanistic-interpretability-grokking/progress-measures-paper)
* [IOI 회로 분석 코드 (Wang et al.)](https://github.com/redwoodresearch/Easy-Transformer)
* [ARENA Curriculum — Chapter 1: Mech Interp](https://www.arena.education/chapter1)
* [Callum McDougall — TransformerLens Intro](https://github.com/callummcdougall/TransformerLens-intro)
* [Neel Nanda — Mechanistic Interpretability Quickstart](https://www.alignmentforum.org/posts/jLAvJt8wuSFySN975/mechanistic-interpretability-quickstart-guide)

### 분야 위상 (제안 정당화)

* [Mechanistic Interpretability — MIT Tech Review 2026 Top 10 Breakthrough Technologies](https://www.technologyreview.com/2026/01/12/1130003/mechanistic-interpretability-ai-research-models-2026-breakthrough-technologies/)
* [Dario Amodei — "The Urgency of Interpretability" (2025)](https://www.darioamodei.com/post/the-urgency-of-interpretability)
* [Mechanistic Interpretability Workshop at ICML 2026](https://mechinterpworkshop.com/)
* [Transformer Circuits Thread (Anthropic)](https://transformer-circuits.pub/)

### 시각화 영감

* [Anthropic — Scaling Monosemanticity (interactive features)](https://transformer-circuits.pub/2024/scaling-monosemanticity/)
* [Anthropic — Golden Gate Claude (feature steering demo)](https://www.anthropic.com/news/golden-gate-claude)
* [3Blue1Brown — Visualizing Attention](https://www.youtube.com/watch?v=eMlx5fFNoYc)
* [Distill — Visualizing Memorization in RNNs](https://distill.pub/2019/memorization-in-rnns/)
