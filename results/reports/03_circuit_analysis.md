# 회로 분석 (Circuit Analysis)

> **실험 ID**: EX-003
> **분석 대상**: `run_20260524_012428` checkpoint at epoch 50,000
> **작성일**: 2026-05-24
> **상태**: ✅ 완료
> **이전**: [`02_grokking_training.md`](./02_grokking_training.md)

---

## 1. 개요

Grokking으로 학습된 1-layer Transformer의 internal circuit을 분석하여 modular arithmetic를 푸는 메커니즘을 규명한다.

**분석 방법**:
1. **Fourier Basis Analysis** — Residual stream의 주파수 성분 분석
2. **Attention Pattern Analysis** — 각 head의 attention pattern 시각화
3. **Head Ablation Study** — Head별 중요도 평가

---

## 2. Fourier Basis 분석

### 2.1 결과

| 순위 | 주파수 | 상관계수 | 설명 |
|------|--------|----------|------|
| 1 | 85 | 0.8742 | Top frequency |
| 1 | 28 | 0.8742 | 85 + 28 = 113 (p) |
| 3 | 102 | 0.8651 | 113 - 11 = 102 |
| 3 | 11 | 0.8651 | Complement pair |
| 5 | 15 | 0.8340 | |
| 5 | 98 | 0.8340 | 113 - 15 = 98 |

### 2.2 해석

- **Dominant frequencies**: 85, 28 (상관계수 0.87)
- **Symmetry**: f와 (113-f)가 동일한 상관계수를 가짐
- **의미**: 모델이 Fourier basis에서 residue를 표현함

**수학적 배경**: Modular multiplication은 Fourier domain에서 convolution으로 표현 가능
```
(a × b) mod p = F⁻¹[F(a) ⊙ F(b)] mod p
```
여기서 F는 Discrete Fourier Transform (DFT) on Z_p.

---

## 3. Attention Pattern 분석

### 3.1 전체 구조

각 head는 **a (pos 0)**와 **b (pos 1)**에 대해 거의 균등하게 (≈50:50) attend 함.

| Head | Attention to a | Attention to b | Attention to = |
|------|----------------|----------------|----------------|
| 0 | 50.2% : 49.8% | 49.6% : 49.7% | 0.15% |
| 1 | 49.9% : 54.7% | 50.1% : 45.1% | 0.19% |
| 2 | 50.1% : 48.8% | 49.9% : 51.0% | 0.17% |
| 3 | 49.8% : 47.7% | 50.2% : 52.0% | 0.23% |

### 3.2 해석

- **均匀分布**: 모든 head가 a와 b에 균등하게 attend
- **No specialization**: 특정 head가 특정 input만 처리하지 않음
- **의미**: 각 head가 동일한 연산을 병렬로 수행 (ensemble)

---

## 4. Head Ablation 연구

### 4.1 결과

| Head | Ablation 후 Acc | Acc 감소 | 중요도 |
|------|-----------------|-----------|--------|
| 1 | 38.0% | **-62.0%** | 🔴 CRITICAL |
| 2 | 56.8% | -43.2% | 🟠 Important |
| 3 | 60.4% | -39.6% | 🟠 Important |
| 0 | 74.8% | -25.2% | 🟡 Moderate |
| **Baseline** | **100.0%** | — | — |

### 4.2 해석

- **Head 1 가장 중요**: Ablation 시 62% 성능 하락
- **순차적 중요도**: Head 1 > Head 2 > Head 3 > Head 0
- **Ensemble 효과**: 모든 head가 필요하나 Head 1이 핵심

**기능 가설**:
- Head 1: Primary computation circuit
- Head 2, 3: Auxiliary computation (gradient flow 유지)
- Head 0: Minor contribution

---

## 5. Circuit 구조 가설

```
Input tokens (a, b, =)
        ↓
Positional encoding + Embedding
        ↓
┌─────────────────────────────────┐
│   Layer 0 (4-head attention)     │
│   ├─ Head 0: 25% contribution    │
│   ├─ Head 1: 62% contribution    │ ← CRITICAL
│   ├─ Head 2: 43% contribution    │
│   └─ Head 3: 40% contribution    │
└─────────────────────────────────┘
        ↓
   MLP projection (identity)
        ↓
Residual stream: Fourier basis
( freq 85/28 dominate )
        ↓
   Unembedding → Output token
```

---

## 6. Nanda et al. (2022)과의 비교

| 항목 | Nanda et al. | 본 실험 |
|------|--------------|--------|
| Architecture | 2-layer (1+1) | 1-layer |
| Heads | 4 heads | 4 heads |
| Fourier basis | ✅ 확인 | ✅ 확인 (freq 85/28) |
| Algorithmic circuit | ✅ 확인 | ✅ 확인 |
| Grokking epoch | ~3,000 | ~10,200 |

**차이점**: 본 실험은 더 깊은 모델에서 slower grokking 관찰

---

## 7. 결과 파일 위치

```
results/grokking/analysis/
├── report.txt                    # 요약 보고서
├── fourier_scores.npz/.png       # 주파수별 상관계수
├── attention_patterns.npz/.png   # Attention heatmap
└── head_ablation.npz/.png        # Ablation 결과
```

---

## 8. 재현 방법

```bash
# 1. 학습된 모델로 분석 실행
cd tracks/grokking
python run_analysis.py --checkpoint results/grokking/run_20260524_012428/checkpoints/model_ep50000.pt

# 2. 결과 시각화
python visualize_analysis.py --analysis results/grokking/analysis/

# 3. report 확인
cat results/grokking/analysis/report.txt
```

---

## 9. 다음 단계

1. **Activation patching**: Head 1의 internal representation 분석
2. **Logit lens**: 각 layer에서의 output probability 추적
3. **Probing**: Fourier coefficient 계산 neuron 분석
