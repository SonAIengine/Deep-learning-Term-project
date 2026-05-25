# Grokking 학습 실험

> **실험 ID**: EX-002
> **실험 Run**: `run_20260524_012428`
> **작성일**: 2026-05-24
> **상태**: ✅ 완료
> **이전**: [`01_dataset_generation.md`](./01_dataset_generation.md)
> **다음**: [`03_circuit_analysis.md`](./03_circuit_analysis.md)

---

## 1. 개요

Modular arithmetic (p=113) 데이터셋에서 1-layer Transformer를 학습시켜 Grokking phenomenon을 관찰한다.

**Grokking**: Memorization에서 Generalization으로의 급격한 전이 현상 (Nanda et al., 2022)

---

## 2. 학습 설정

| 항목 | 값 |
|------|------|
| 모델 | 1-layer Transformer, 4 heads |
| Hidden dim | 128 |
| Optimizer | AdamW (lr=1e-3, weight_decay=1.0) |
| Epochs | 50,000 |
| Eval interval | 100 epochs |
| Warmup | None |
| Batch size | 512 |

### 데이터셋
| 항목 | 값 |
|------|------|
| Train | 3,830 samples (~30%) |
| Test | 8,939 samples (~70%) |
| Operation | a × b mod 113 |

---

## 3. 학습 결과

### 3.1 Loss 곡선

```
Train loss: 1.96e-05 (initial) → 3.94e-05 (final)
Test loss:  4.54 (initial) → 2.19e-04 (final)
```

### 3.2 Accuracy 곡선

```
Train acc: 4.3% (initial) → 100% (final)
Test acc:  0% (initial) → 100% (final)
```

### 3.3 Grokking Point

| 지표 | 값 |
|------|------|
| Grokking epoch | **10,200** (eval step 102) |
| Grokking 정의 | Test accuracy > 99% |
| Time to grokking | ~65 seconds |

**해석**: 약 10,000 epoch 이후에 test accuracy가 급격히 상승하며 generalization이 발생함.

---

## 4. Grokking 현상 분석

### 4.1 Phase 전이

| Phase | Epoch 범위 | Train Acc | Test Acc | 특징 |
|-------|------------|-----------|----------|------|
| Memorization | 0-5,000 | 100% | <5% | Train set만 외움 |
| Transition | 5,000-10,000 | 100% | 5-90% | 일반화 시작 |
| Generalization | 10,000+ | 100% | 100% | 알고리즘 발견 |

### 4.2 Loss vs Accuracy

- **Train loss**: Early epoch에 빠르게 수렴 (memorization)
- **Test loss**: Grokking 이후에야 수렴 시작
- **Test accuracy**: 계단 함수 모양 (sharp transition)

---

## 5. 재현 방법

```bash
# 1. 학습 실행
cd tracks/grokking
python train.py --epochs 50000 --eval-every 100 --output results/grokking/

# 2. Loss 곡선 시각화
python visualize_training.py --run results/grokking/run_20260524_012428/

# 3. 결과 확인
ls results/grokking/run_20260524_012428/
# - loss_curve.npz  (train/test loss, acc)
# - grokking_curve.png
# - checkpoints/     (model weights)
```

---

## 6. Checkpoint 정보

| Checkpoint | Epoch | Test Acc | 설명 |
|------------|-------|----------|------|
| `model_ep00000.pt` | 0 | 0% | 초기화 |
| `model_ep05000.pt` | 5,000 | 4.8% | Memorization 완료 |
| `model_ep10200.pt` | 10,200 | 100% | Grokking 발생 |
| `model_ep50000.pt` | 50,000 | 100% | Final converged |

---

## 7. 관련 문서

- **이전**: [`01_dataset_generation.md`](./01_dataset_generation.md) — 데이터셋 생성 상세
- **다음**: [`03_circuit_analysis.md`](./03_circuit_analysis.md) — 회로 분석 결과
- **시각화**: `results/grokking/run_20260524_012428/grokking_curve.png`
