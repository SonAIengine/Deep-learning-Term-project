# Transformer Circuit Analysis — Grokking & Code Variable Binding

> Mechanistic Interpretability term project. See [docs/candidates/sonsj-proposal.md](docs/candidates/sonsj-proposal.md) for the project proposal and [docs/datasets.md](docs/datasets.md) for the dataset specification.

---

## 🎯 `son` 브랜치 — 손성준 개인 작업

> 본 브랜치는 손성준 제안서 ([sonsj-proposal.md](docs/candidates/sonsj-proposal.md)) Step 1~4 + 7개 추가 실험의 결과물.

### 한 줄 요약
**GPT-2 small의 IOI 회로(Wang 2022)는 코드 변수 바인딩 도메인에서 *선택적으로* 재사용되며, 출력 단계 head는 *기능적 부호(±)까지 보존*된 채 전이된다. Pythia-160M·GPT-2 medium 재현 결과, universality는 *해부학적 수준*이 아닌 *기능적 수준*에서 성립한다.**

### 📑 진입점 (꼭 보세요)

| 문서 | 설명 |
|---|---|
| 📄 [`docs/results.md`](docs/results.md) | **전체 결과 리포트** (수치·그래프·해석) |
| 🎤 [`docs/slides.md`](docs/slides.md) | **발표 슬라이드** (Marp, 23장) |
| 🌐 [wandb run](https://wandb.ai/sonsj97-plateer/grokking-circuits/runs/n4bnqrak) | Grokking 학습 라이브 |

### 🏆 핵심 발견 7개

1. **NL↔Code cross-domain 회로 재사용 첫 정량화** (top-26 ∩ IOI-26 = 10, random ×2.1)
2. **Selective reuse**: 출력 head 전이 ✓ / SIH 비전이 ✗
3. **기능적 부호 보존** — NNMH(L10H7)의 negative 역할이 코드에서도 −0.31, z=−1.9
4. **Head ≠ position grammar** — DTH가 예상 외 위치(final `=`)에서 작동
5. **회로 크기**: IOI-26(18%)이 attention 기여의 **87%** 담당
6. **Universality 4단계 분해** (functional ✓ / anatomical ✗) — 3 모델 비교
7. **Scale-dependent hydra effect** — GPT-2 medium ablation이 성능 향상 (−0.121) ⭐ 신규

### 🎬 핵심 시각화

| 그림 | 무엇 |
|---|---|
| [loss_curve.gif](results/analysis/grokking_full/loss_curve.gif) | Grokking 학습 — memorization → 일반화 |
| [attention_evolution.gif](results/analysis/grokking_full/attention_evolution.gif) | Attention 패턴 학습 진행 |
| [heatmap_classes.png](results/binding_compare/heatmap_classes.png) | Patching recovery + IOI head 클래스별 outline |
| [universality_summary.png](results/binding_compare/universality_summary.png) | IOI 클래스별 전이 패턴 + Top-K enrichment |
| [layer_profile_comparison.png](results/binding_gpt2med/layer_profile_comparison.png) | 3 모델 layer 분포 비교 |
| [necessity.png](results/binding_minimal_circuit/necessity.png) | Minimal circuit necessity test |
| [circuit_diagram.png](results/binding_compare/circuit_diagram.png) | 발견된 회로 schematic |

### 📂 코드 (son 브랜치 추가분)

```
src/
├── grokking/
│   ├── model.py             # 2-layer transformer (Nanda-style, no LN/bias)
│   ├── train.py             # AdamW full-batch, wandb 로깅
│   ├── analysis.py          # Fourier mass + per-head/MLP ablation
│   ├── animate.py           # loss/attention GIF 생성
│   └── wandb_export.py      # wandb API → 차트 PNG
└── binding/
    ├── baseline.py                  # Tier 1 clean/corrupt logit_diff
    ├── patching.py                  # per-head activation patching (12×12)
    ├── compare_ioi.py               # Wang 2022 26 heads vs ours
    ├── tier_generalize.py           # Tier 2/3 head ablation
    ├── position_patching.py         # Top heads × 14 positions
    ├── sih_position_patching.py     # ⭐ Extra A: SIH non-transfer
    ├── logit_attribution.py         # ⭐ Extra B: direct logit attribution
    ├── minimal_circuit.py           # ⭐ Extra F: necessity test
    ├── pythia_replication.py        # ⭐ Extra D: Pythia-160M
    ├── gpt2med_replication.py       # ⭐ Extra D2: GPT-2 medium
    └── figures.py                   # publication-quality figures
```

### 🔁 재현 방법

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
uv run python src/binding/sih_position_patching.py   # A
uv run python src/binding/logit_attribution.py       # B
uv run python src/binding/minimal_circuit.py         # F
uv run python src/binding/pythia_replication.py      # D
uv run python src/binding/gpt2med_replication.py     # D2 (~70min)

# 시각화
uv run python src/grokking/animate.py
uv run python src/grokking/wandb_export.py
```

### 📊 실험 인프라
- H100 80GB × 1, 총 GPU 사용 약 3시간
- 모델: GPT-2 small (124M), GPT-2 medium (355M), Pythia-160M
- 평가: 500 counterfactual pair × per-head patching

---

## 팀 공용 README (main 브랜치 내용)

### Repository Layout (팀 원본)

```
.
├── data/             # Data generation scripts (centralized)
│   ├── _day1_verify.py
│   ├── modular.py
│   ├── var_binding.py
│   └── ioi_loader.py
├── shared/           # Shared config + result schema
├── datasets/         # Generated data
├── tracks/           # Per-track analysis code (P1/P2/P3)
├── results/          # Circuit-analysis JSON outputs
└── docs/             # Proposal, dataset spec, meeting notes
```

### Quick Start (팀 원본)

```bash
# 1. Install
pip install torch transformers transformer_lens einops
pip install git+https://github.com/redwoodresearch/Easy-Transformer.git

# 2. Week 1 Day 1: verify tokenizer pools, update shared/config.py with output
python -m data._day1_verify

# 3. Generate all data
python -m data.modular
python -m data.var_binding
python -m data.ioi_loader   # sanity check
```

## Team Tracks

| Track | Owner | Scope |
|---|---|---|
| Grokking (Step 1~2) | P1 | Modular arithmetic scratch training, Fourier analysis, ablation |
| Code Binding (Step 3-1/3-2) | P2 | GPT-2 + variable binding task, activation patching, ablation |
| IOI + Comparison (Step 3-3) | P3 | IOI replication, Wang et al. 26-head reproduction, universality score |

See [docs/datasets.md](docs/datasets.md) §9 for the Week 1 day-by-day schedule.
