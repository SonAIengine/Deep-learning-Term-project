# 🧠 Deep Learning Term Project: Mechanistic Interpretability

> **목적**: Transformer 모델의 내부 회로(circuit) 분석을 통해 AI가 "생각"하는 방법 이해하기
> 
> **기간**: 2026년 봄학기 | **과목**: Deep Learning (Y1S2)
> 
> **핵심 발견**: Model depth와 task complexity가 circuit architecture를 결정하며, circuit universality는 낮음 (0.12/1.0)

---
## 🎯 프로젝트 개요

이 프로젝트는 **Mechanistic Interpretability** (기계적 해석가능성) 기법을 사용하여 Transformer 모델의 내부 작동 원리를 분석합니다. 

### 연구 질문

1. **Circuit Discovery**: Transformer 모델 내부에서 정보가 어떻게 처리되는가?
2. **Architecture Effects**: 모델 깊이가 회로 형성에 어떤 영향을 미치는가?
3. **Task Complexity**: 작업 복잡도가 회로 특성화에 어떤 영향을 미치는가?
4. **Circuit Universality**: 다른 작업 간에 공통된 회로 패턴이 존재하는가?

---

## 👥 트랙 구성

본 프로젝트는 팀원별 트랙으로 구성됩니다. 팀 공용 분석(P1~P5)은 `tracks/`,
손성준 트랙의 심화 분석(Grokking + Code Variable Binding × IOI universality)은
`src/`에 있습니다.

| 트랙 | 위치 | 진입 문서 |
|---|---|---|
| 팀 공용 분석 (P1~P5) | `tracks/`, `results/{grokking,code,sva,sentiment}` | 본 README 하단 |
| **손성준 심화 트랙** | `src/{grokking,binding}`, `results/binding_*` | [docs/results.md](docs/results.md) · [docs/slides.md](docs/slides.md) |

---

## 🧩 손성준 심화 트랙 — IOI Circuit Universality

> 제안서 ([sonsj-proposal.md](docs/candidates/sonsj-proposal.md)) Step 1~4 + 7개 추가 실험.
> 상세 결과: **[docs/results.md](docs/results.md)** · 발표 슬라이드: **[docs/slides.md](docs/slides.md)** (Marp 23장)

**한 줄 요약**: GPT-2 small의 IOI 회로(Wang 2022)는 코드 변수 바인딩 도메인에서
*선택적으로* 재사용되며, 출력 단계 head는 *기능적 부호(±)까지 보존*된 채 전이된다.
Pythia-160M·GPT-2 medium 재현 결과, universality는 *해부학적 수준*이 아닌
*기능적 수준*에서 성립한다.

**핵심 발견 7개**:

1. **NL↔Code cross-domain 회로 재사용 첫 정량화** (top-26 ∩ IOI-26 = 10, random ×2.1)
2. **Selective reuse**: 출력 head 전이 ✓ / SIH 비전이 ✗
3. **기능적 부호 보존** — NNMH(L10H7)의 negative 역할이 코드에서도 −0.31, z=−1.9
4. **Head ≠ position grammar** — DTH가 예상 외 위치(final `=`)에서 작동
5. **회로 크기**: IOI-26(18%)이 attention 기여의 **87%** 담당
6. **Universality 4단계 분해** (functional ✓ / anatomical ✗) — 3 모델 비교
7. **Scale-dependent hydra effect** — GPT-2 medium ablation이 성능 향상 (−0.121) ⭐ 신규

**핵심 시각화**: [loss_curve.gif](results/analysis/grokking_full/loss_curve.gif) ·
[heatmap_classes.png](results/binding_compare/heatmap_classes.png) ·
[universality_summary.png](results/binding_compare/universality_summary.png) ·
[layer_profile_comparison.png](results/binding_gpt2med/layer_profile_comparison.png) ·
[necessity.png](results/binding_minimal_circuit/necessity.png)

**재현**:

```bash
uv sync   # pyproject.toml 기반

# Grokking 학습 + 분석 (40k steps, ~2분 on H100)
uv run python src/grokking/train.py --tag grokking_full --steps 40000
uv run python src/grokking/analysis.py

# Step 3 핵심 실험
uv run python src/binding/baseline.py
uv run python src/binding/patching.py
uv run python src/binding/compare_ioi.py

# Extras (독립 실행)
uv run python src/binding/sih_position_patching.py   # A: SIH non-transfer
uv run python src/binding/logit_attribution.py       # B: direct logit attribution
uv run python src/binding/minimal_circuit.py         # F: necessity test
uv run python src/binding/pythia_replication.py      # D: Pythia-160M
uv run python src/binding/gpt2med_replication.py     # D2: GPT-2 medium (~70min)

# 시각화
uv run python src/grokking/animate.py
uv run python src/grokking/wandb_export.py
```

---

## 📊 팀 공용 주요 결과

### 3가지 핵심 발견

| 발견 | 내용 | 의미 |
|------|------|------|
| **1️⃣ Model Depth Effects** | 1-layer → 집중된 회로, 12-layer → 분산된 회로 | 모델 깊이가 circuit architecture 결정 |
| **2️⃣ Task Complexity Effects** | Simple task → 집중된 특화, Complex task → 분산된 특화 | 작업 복잡도가 specialization pattern 결정 |
| **3️⃣ Low Circuit Universality** | Cross-track overlap 7.7-20% | 각 작업마다 고유한 회로 사용 |

### 수행한 분석

- ✅ **P1**: Modular Arithmetic Grokking (1-layer, 50K epochs)
- ✅ **P2**: Code Variable Binding (12-layer, 500 pairs)
- ✅ **P3**: IOI Cross-Track Comparison (Wang et al. 2022 vs P2)
- ✅ **P4**: Syntax Task Analysis - Subject-Verb Agreement (12-layer, 100 pairs)
- ✅ **P5**: Semantic Task Analysis - Sentiment Classification (12-layer, 28 pairs)
- ✅ **Cross-Track**: P1 vs P2 Architecture Comparison

---

## 🚀 빠른 시작

### 설치

```bash
# 필수 라이브러리
pip install torch transformers transformer_lens einops matplotlib

# IOI dataset (optional)
pip install git+https://github.com/redwoodresearch/Easy-Transformer.git
```

### 실행 예제

```bash
# P1: Modular Arithmetic Grokking
python -m tracks.grokking.train --smoke  # 100 epoch 테스트
python -m tracks.grokking.train           # 50K epoch 전체 학습

# P2: Code Variable Binding Analysis
python tracks/code/p3_ioi_comparison.py    # IOI 비교 분석

# P1 vs P2 Cross-Track Comparison
python tracks/analysis/p1_p2_cross_track.py

# P4: Syntax Task Analysis (Subject-Verb Agreement)
python tracks/syntax/p4_sva_analysis.py     # SVA 회로 분석

# P5: Semantic Task Analysis (Sentiment Classification)
python tracks/semantic/p5_sentiment_analysis.py  # Sentiment 회로 분석
```

---

## 📁 프로젝트 구조

```
Deep-learning-Term-project/
├── shared/                  # 공유 설정 및 유틸리티
│   └── config.py            # 하이퍼파라미터, 경로, 시드 설정
├── tracks/                  # 분석 트랙 코드
│   ├── grokking/            # P1: Modular arithmetic grokking
│   │   ├── train.py         # 학습 루프
│   │   └── analysis/        # 회로 분석 결과
│   ├── code/                # P2: Code variable binding
│   │   ├── p3_ioi_comparison.py       # IOI 비교
│   │   └── visualize_p3_results.py    # P3 시각화
│   ├── analysis/            # Cross-track 분석
│   │   └── p1_p2_cross_track.py       # P1 vs P2 비교
│   ├── syntax/              # P4: Syntax task analysis
│   │   └── p4_sva_analysis.py         # Subject-verb agreement
│   └── semantic/            # P5: Semantic task analysis
│       └── p5_sentiment_analysis.py   # Sentiment classification
├── datasets/                # 생성된 데이터셋
│   ├── modular_arithmetic/  # P1 학습/테스트 데이터
│   ├── ioi/                 # P2 IOI 스타일 프롬프트
│   ├── sva_pairs.json       # P4 SVA 문장 쌍
│   └── sentiment_pairs.json # P5 Sentiment 문장들
├── results/                 # 분석 결과 및 시각화
│   ├── grokking/            # P1 결과 (50K epoch, 회로 분석)
│   ├── code/                # P2 결과 (500 pairs, stability)
│   ├── comparison/          # Cross-track 비교
│   ├── sva/                 # P4 문법 작업 결과
│   └── sentiment/           # P5 의미 작업 결과
├── docs/                    # 문서화
│   ├── PROJECT_SUMMARY.md   # 프로젝트 요약 (친절한 설명)
│   ├── PROJECT_STRUCTURE.md # 상세 프로젝트 구조
│   └── reports/             # 분석 상세 보고서
└── README.md                # 프로젝트 개요 (이 파일)
```

---

## 🔬 분석 방법

### 사용한 기법

1. **Activation Patching**: 특정 뉴런의 활성화 값을 변경하며 영향 확인
2. **Head Ablation**: Attention head를 하나씩 끄며 성능 변화 측정
3. **Fourier Analysis**: 주파수 영역에서 모델의 학습 패턴 분석

### 분석 대상

| 트랙 | 모델 | 작업 | 주요 특징 |
|------|------|------|-----------|
| **P1** | 1-layer Transformer (d_model=128, 4 heads) | Modular arithmetic (mod 113) | Grokking 현상, Fourier basis |
| **P2** | GPT-2 Small (12-layer, d_model=768) | Code variable binding | Activation patching, stability |
| **P3** | Wang et al. 2022 reference | IOI task | Cross-track comparison |
| **P4** | GPT-2 Small (12-layer, d_model=768) | Subject-verb agreement | Syntax processing circuits |
| **P5** | GPT-2 Small (12-layer, d_model=768) | Sentiment classification | Semantic understanding circuits |

---

## 📈 주요 결과

### P1: Modular Arithmetic Grokking

```
✅ 학습 완료: 50,000 epochs
✅ Grokking point: epoch 10,200
✅ 최종 정확도: 100%

🧠 회로 분석:
- Head 1: CRITICAL (-62% drop when ablated)
- Fourier basis: Freq 85, 28 dominant (r=0.87)
- Attention pattern: Uniform across inputs
```

**발견**: 모델이 단순히 외우는 게 아니라 수학적 구조를 학습함

### P2: Code Variable Binding

```
✅ 500개 코드 쌍 분석 완료
✅ Top heads: L4H5 (1.30), L3H5 (1.30), L4H6 (1.29)
✅ Stability: 5→500 pairs에서 회로 완전히 변경 (Top 5 overlap: 0%)

🔍 Layer Distribution:
- Early-middle layers dominant (L1-L10)
- No single critical head (distributed contribution)
```

**발견**: 충분한 샘플링이 중요 — 적은 샘플로는 잘못된 회로 발견 가능

### P3: IOI Cross-Track Comparison

```
✅ Wang et al. 2022 vs P2 var binding 비교
✅ Cross-track overlap: 20% (Top-10), 10% (Top-20)
✅ Jaccard similarity: 3.23% (매우 낮음)

🎯 Overlapping heads: L10H11, L9H7 (단 2개)
📊 Layer distribution: P2 (early L1-L10) vs Wang et al. (late L8-L11)
```

**발견**: Circuit universality 낮음 — 각 작업은 고유한 전략 사용

### P1 vs P2: Architecture Comparison

```
📐 Architecture Scaling:
- Depth: 1 vs 12 layers (12x difference)
- Heads: 4 vs 144 heads (36x difference)
- Capacity: d_model 128 vs 768 (6x difference)

🧠 Circuit Architecture:
- P1: Concentrated computation (Head 1 critical)
- P2: Distributed processing (no single critical head)
```

**발견**: Model depth가 circuit architecture 결정 — 깊이에 따라 완전히 다른 전략

### P4: Syntax Task (Subject-Verb Agreement)

```
✅ 100개 문장 쌍 분석 완료
✅ Top heads: L0H4, L0H7, L1H1, L1H7, L2H1
✅ 모든 레이어에서 유사한 활성화 패턴 발견

🔍 Layer Distribution:
- 초기 레이어 (L0-L2)에서 높은 중요도
- 문법 처리가 초기 레이어에서 시작됨
- 분산된 문법 표현 (distributed syntax processing)
```

**발견**: 문법 처리가 초기 레이어에서 시작되며 분산된 회로 구조

### P5: Semantic Task (Sentiment Classification)

```
✅ 28개 감정 문장 분석 완료 (positive/negative)
✅ 분류 정확도: 57.1%
✅ Top heads: 다층 분산 표현 (distributed semantic representation)

🔍 Semantic Processing:
- 단어 기반 감정 분석 (keyword-based)
- 의미 정보가 여러 레이어에 분산
- 문법 작업과는 다른 회로 패턴
```

**발견**: 의미 처리가 분산된 회로를 통해 이루어지며, 문법 처리와는 구별되는 패턴

---

## 📊 시각화

### 생성한 플롯 (15개)

**P1 (Grokking)**:
- Loss curve & grokking moment
- Fourier basis scores
- Attention patterns & head ablation

**P2 (Var Binding)**:
- Var binding heatmap
- Recovery distribution
- Stability comparison (5 vs 500 pairs)

**P3 (IOI Comparison)**:
- Layer distribution comparison
- Overlap analysis (4-panel)
- Circuit universality curve

**Cross-Track (P1 vs P2)**:
- Architecture comparison (6-panel)
- Universality analysis (4-panel)

**P4 (Syntax)**:
- SVA sentence pairs analysis
- Head importance distribution
- Syntax processing circuit visualization

**P5 (Semantic)**:
- Sentiment classification results
- Activation pattern analysis
- Semantic circuit comparison

---

## 🌟 주요 기여도

### 이론적 기여

1. **Model Depth → Circuit 관계 규명**
   - 1-layer: 집중된 회로
   - 12-layer: 분산된 회로
   - 처음으로 정량적으로 규명

2. **Task Complexity → Specialization 관계 발견**
   - Simple task: 집중된 특화
   - Complex task: 분산된 특화

3. **Circuit Universality 실증**
   - Cross-task overlap: 7.7-20% (매우 낮음)
   - 각 작업은 고유한 전략 사용

### 방법론적 기여

1. **Stability 분석 프레임워크**
   - 샘플 수 영향 체계적 분석
   - 5→500 pairs에서 회로 완전히 변경 발견

2. **Cross-track 비교 방법론**
   - 서로 다른 작업 간 회로 비교
   - Jaccard similarity 등 정량적 지표 개발

---

## 💡 시사점

### AI 연구자들에게

1. **Circuit Discovery 시 주의사항**
   - ⚠️ 충분한 샘플링 필수
   - ⚠️ 적은 샘플로는 잘못된 회로 발견 가능

2. **Model Selection 가이드**
   - 해석 가능성이 중요하면 → Shallow models
   - 복잡한 작업이 필요하면 → Deep models

3. **Circuit Universality 기대 조정**
   - Universal circuit 찾기보다는
   - Task-specific circuit 이해에 집중

---

## 📚 상세 문서

### 분석 보고서

- `docs/PROJECT_SUMMARY.md` — 프로젝트 전체 요약 (초보자용)
- `docs/reports/02_grokking_training.md` — P1 학습 과정
- `docs/reports/03_circuit_analysis.md` — P1 회로 분석
- `docs/reports/p2_var_binding_implementation.md` — P2 구현
- `docs/reports/04_p3_ioi_comparison.md` — P3 IOI 비교
- `docs/reports/05_p1_p2_cross_track.md` — P1 vs P2 비교

### 결과 데이터

- `results/grokking/` — P1 학습 및 회로 분석 결과
- `results/code/` — P2 var binding 결과 및 시각화
- `results/analysis/` — Cross-track 비교 시각화

---

## 🔧 환경 설정

### Python 버전

- Python 3.8+
- PyTorch 1.10+
- Transformers 4.20+

### 주요 라이브러리

```python
torch                 # Deep learning framework
transformers          # Pre-trained models
transformer_lens      # Mechanistic interpretability
einops               # Tensor operations
matplotlib           # Visualization
numpy                # Numerical computing
```

---

## 📖 참고자료

### 관련 논문

- **Nanda et al. (2023)**: "Progress Measures for Grokking"
- **Wang et al. (2022)**: "Interpretability at Scale: IOI in GPT-2"
- **Olsson et al. (2022)**: "Understanding the Scaling Behavior of Grokking"

### 관련 프로젝트

- [Anthropic Mechanical Interpretability](https://www.anthropic.com/research/mechanistic-interpretability)
- [TransformerLens](https://github.com/neelnanda-io/TransformerLens)
- [Easy-Transformer](https://github.com/redwoodresearch/Easy-Transformer)

---

## 👥 팀

| 트랙 | 담당 |
|------|------|
| P1: Grokking | yesulmin |
| P2: Code Binding | yesulmin |
| P3: IOI Comparison | yesulmin |
| Cross-Track Analysis | yesulmin |

---

## 📝 라이선스

이 프로젝트는 학기 프로젝트로서 진행되었으며, 교육적 목적으로 사용됩니다.

---

## 🎉 결론

이 프로젝트를 통해 Transformer 모델의 내부 회로를 처음으로 체계적으로 분석했습니다. 

**핵심 성과**:
- ✅ Model depth가 circuit architecture 결정
- ✅ Task complexity가 specialization pattern 결정  
- ✅ Circuit universality 낮음 (0.12/1.0)
- ✅ 5개 트랙 (P1-P5) + 2개 cross-track 분석 완료
- ✅ 15+ 시각화 및 5개 상세 보고서
- ✅ 문법/의미 작업 회로 분석 추가 (P4, P5)

**한 학기 동안 Transformer의 "뇌"를 들여다보며, AI가 어떻게 "생각"하는지 이해하는 의미 있는 여정을 했습니다! 🧠✨

---

**Last Updated**: 2026-05-25
**Project Status**: ✅ Complete (5 tracks + 2 cross-track analyses + documentation)
**Repository**: [GitHub](https://github.com/yesulmin-danbaaam/Deep-learning-Term-project)