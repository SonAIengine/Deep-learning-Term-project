# 실험 전체 개요

> **실험 ID**: EX-000
> **작성일**: 2026-05-24
> **상태**: 🔄 진행 중
> **담당**: yesul.min

---

## 1. 실험 목적

Transformer의 mechanistic interpretability를 연구하기 위해 modular arithmetic와 variable binding 회로를 분석한다.

**연구 질문**:
1. Grokking: 모델이 memorization에서 generalization으로 전이되는 메커니즘은?
2. Circuit: Modular arithmetic를 푸는 neural circuit의 구조는?
3. Transfer: Code variable binding에서 유사한 회로가 발견되는가?

---

## 2. 실험 구조

```
┌─────────────────────────────────────────────────────────────┐
│                    TRANSFORMER CIRCUIT ANALYSIS              │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  [Phase 1] Dataset Generation   (01_dataset_generation.md)  │
│  ├─ Modular arithmetic (p=113)                              │
│  ├─ Var binding Tier 1/2/3                                  │
│  └─ IOI (Indirect Object Identification)                    │
│                                                               │
│  [Phase 2] P1: Grokking        (02_grokking_training.md)    │
│  ├─ 1-layer Transformer training                            │
│  └─ Grokking phenomenon observation                          │
│                                                               │
│  [Phase 3] P1: Circuit Analysis  (03_circuit_analysis.md)   │
│  ├─ Fourier basis verification                               │
│  ├─ Attention pattern analysis                               │
│  └─ Head ablation study                                      │
│                                                               │
│  [Phase 4] P2: Var Binding    (TODO)                         │
│  └─ Activation patching on GPT-2 small                       │
│                                                               │
│  [Phase 5] P3: IOI Comparison   (TODO)                       │
│  └─ Wang et al. 26 head overlap analysis                     │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. 진행 상황

| 단계 | 문서 | 상태 | 완료일 |
|------|------|------|--------|
| 데이터셋 생성 | `01_dataset_generation.md` | ✅ 완료 | 2026-05-24 |
| Grokking 학습 | `02_grokking_training.md` | ✅ 완료 | 2026-05-24 |
| 회로 분석 | `03_circuit_analysis.md` | ✅ 완료 | 2026-05-24 |
| Var binding 분석 | `04_var_binding_analysis.md` | ⏳ TODO | - |
| IOI 비교 | `05_ioi_comparison.md` | ⏳ TODO | - |
| 최종 보고서 | `99_final_report.md` | ⏳ TODO | - |

---

## 4. 데이터셋 요약

| 데이터셋 | 용도 | 상태 | 파일 위치 |
|----------|------|------|-----------|
| Modular arithmetic | P1 grokking | ✅ Frozen | `datasets/modular_{train,test}.pt` |
| Var binding Tier 1 | P2 circuit analysis | ✅ Frozen | `datasets/var_binding_tier1.jsonl` |
| Var binding Tier 2/3 | P2 extension | ✅ Frozen | `datasets/var_binding_tier{2,3}.jsonl` |
| IOI | P3 comparison | ✅ Ready | `data.ioi_loader` |

**Freeze commit**: `a3d4c82` (2026-05-24)

---

## 5. 결과 파일 위치

```
results/
├── reports/
│   ├── 00_experiment_overview.md      (본 문서)
│   ├── 01_dataset_generation.md
│   ├── 02_grokking_training.md
│   ├── 03_circuit_analysis.md
│   ├── 04_var_binding_analysis.md    (TODO)
│   ├── 05_ioi_comparison.md          (TODO)
│   └── 99_final_report.md            (TODO)
├── grokking/
│   ├── run_20260524_012428/
│   │   ├── loss_curve.npz
│   │   ├── grokking_curve.png
│   │   └── checkpoints/
│   └── analysis/
│       ├── report.txt
│       ├── fourier_scores.npz/.png
│       ├── attention_patterns.npz/.png
│       └── head_ablation.npz/.png
└── (다른 실험 결과...)
```

---

## 6. 관련 Commit

| Commit | 설명 |
|--------|------|
| `a3d4c82` | 데이터셋 동결 (Day 1-3) |
| `d2c0db4` | Tier 3 sanity 검토 |
| `95621c4` | Day 1-3 데이터 생성 + 검증 |

---

## 7. 다음 문서

- **다음**: [`01_dataset_generation.md`](./01_dataset_generation.md) — 데이터셋 생성 상세
- **전체 보고**: [`99_final_report.md`](./99_final_report.md) (완료 시)
