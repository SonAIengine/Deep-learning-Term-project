# P1 Grokking — 결과 보고서

> **상태**: 스캐폴드 + smoke 통과. **50K full run 미실행** (~8분 예상).
> **최종 업데이트**: 2026-05-24
> **담당**: 분석 트랙 P1 (yesul.min)
> **짝 문서**: [.claude/plans/analysis_tracks.md](../../.claude/plans/analysis_tracks.md) (forward 계획), [.claude/plans/_summary.md](../../.claude/plans/_summary.md) (cross-track 통합).
> **데이터**: [datasets/_report.md](../../datasets/_report.md) freeze commit `a3d4c82`.

본 문서는 P1 트랙(Step 1: Nanda et al. 2023 grokking 재현 + Step 2: 회로 분석)에서 무엇이 구현되었고, 어떤 검증을 통과했는지 기록한다.

---

## 1. 환경

| 항목 | 값 |
|---|---|
| 호스트 | WSL2, Linux 6.6.87.2-microsoft-standard |
| Python | 3.x (`.venv/bin/python`) |
| CUDA | available (smoke 0.9s/100 epoch 기준) |
| 주요 lib | torch, transformer_lens, einops |

데이터셋 동결 가드 + zero-shot eval regression 통과 후 작업 시작 (analysis_tracks §2.0 부트스트랩).

---

## 2. 구현된 파일

`tracks/grokking/` 신규 생성 (6 파일, 모두 untracked):

| 파일 | 역할 | 크기 |
|---|---|---|
| `__init__.py` | 패키지 마커 | 0 |
| `config.py` | Nanda 하이퍼파라미터 (모델 + 학습 + eval) | 1.3K |
| `data.py` | frozen `modular_{train,test}.pt` 로더 (`load_modular()`) | 467B |
| `model.py` | `build_model()` — HookedTransformer config 구성 | 695B |
| `eval.py` | `eval_split()`, `train_loss()` — `=` 위치 전용 CE | 412B |
| `train.py` | full-batch AdamW 학습 루프 (`python -m tracks.grokking.train`) | 3.1K |

진입점:
- `python -m tracks.grokking.train --smoke` — 100 epoch sanity (체크포인트 없음)
- `python -m tracks.grokking.train` — 50K full run, 체크포인트 매 1000 epoch
- `python -m tracks.grokking.train --epochs N` — custom

산출물 경로: `results/grokking/{smoke_,run_}<timestamp>/{loss_curve.npz, checkpoints/}`.

---

## 3. Canonical Nanda config — 출처 + 박제값

리서치 출처: `mechanistic-interpretability-grokking/progress-measures-paper`, `transformers.py` Config 데이터클래스 (lines 28-66). TransformerLens `Grokking_Demo.ipynb`와 일치. arXiv 2301.05217 appendix §A 일치.

### 3.1 모델

| 파라미터 | 값 | 출처 |
|---|---|---|
| `n_layers` | 1 | `transformers.py:46` |
| `d_model` | 128 | `transformers.py:32` |
| `n_heads` | 4 | `transformers.py:51` |
| `d_head` | 32 (= d_model / n_heads) | `transformers.py:65-66` |
| `d_mlp` | 512 (= 4 × d_model) | `transformers.py:50` |
| `n_ctx` | 3 (a, b, "=") | `transformers.py:49` |
| `d_vocab` | 114 (= MODULAR_P + 1) | `transformers.py:48` |
| `act_fn` | `"relu"` | `transformers.py:53` |
| `normalization_type` | **None** (LN call site 전부 주석 처리) | `transformers.py:198,247,267-269,311` |
| pos embedding | learned (`W_pos ~ N(0, 1/d_model)`) | `transformers.py:181-183` |
| attn_only | False (MLP 있음) | `transformers.py:270` |

### 3.2 학습

| 파라미터 | 값 | 출처 |
|---|---|---|
| optimizer | AdamW, betas=(0.9, 0.98) | `transformers.py:470` |
| lr | 1e-3 | `transformers.py:29` |
| weight_decay | **1.0** (grokking 핵심) | `transformers.py:30` |
| LR schedule | linear warmup 10 step → constant | `transformers.py:471` |
| batch | **full-batch** (~3830 train sample/step) | `transformers.py:47` |
| num_epochs | 50,000 | `transformers.py:35` |
| loss | CE on `logits[:, -1, :]` ("=" 위치) | training loop |
| seed | torch+numpy+random 모두 0으로 박제 | `train.py:seed_all()` |

### 3.3 Eval

- log cadence: 매 100 epoch (~500 log points)
- checkpoint cadence: 매 1000 epoch
- "grokked" 임계값: 코드 차원에서는 자동 정지 없음 (`stopping_thresh=-1`). loss curve로 정성 판정.

### 3.4 알려진 reproducibility 갭

- Nanda 원본은 `random.seed()`만 호출 (train/test 셔플용). torch/numpy global RNG는 안 박음 → 가중치 초기화 RNG 미고정. **본 구현에서는** `seed_all(0)`으로 torch/numpy 둘 다 박음 → 원본 대비 더 강한 reproducibility (다른 가중치로 시작할 가능성 있음).
- HookedTransformer의 `init_weights=True` 기본 초기화 스킴이 Nanda의 `N(0, 1/sqrt(d_model))`와 정확히 일치하는지 미확인. drift 발생 시 학습 dynamics가 약간 다를 수 있음 — full run 결과로 검증 필요.

---

## 4. Smoke 결과 (2026-05-24)

```
.venv/bin/python -m tracks.grokking.train --smoke
ep      0  train 4.7428/0.012  test 4.7442/0.009  elapsed 0.3s
ep     99  train 3.2209/0.254  test 6.8955/0.011  elapsed 0.9s
[done] out_dir=.../smoke_20260524_010604  wall=0.9s
```

해석:
- 초기 loss ≈ log(114) = 4.736 → uniform softmax 확인 ✓
- 100 epoch에서 **train loss 4.74 → 3.22, train acc 1.2% → 25.4%** = memorization 시작
- 같은 시점 **test loss 4.74 → 6.90, test acc 0.9% → 1.1%** = test가 *악화* 중 (overfit → grokking 전 단계)
- 이 패턴은 Nanda Figure 1과 일치 (memorization → 정체 → 갑작스러운 generalization)

Wall time: 100 epoch = 0.9s → **50K epoch ≈ 450s (7.5분)** 예상. eval 매 100이라 eval cost 포함된 추정.

---

## 5. 미실행 — 50K full run

스캐폴드 + smoke만 통과. full run launch 안 됨. 실행 명령:

```bash
nohup .venv/bin/python -m tracks.grokking.train \
  > results/grokking/train.log 2>&1 &
```

또는 동기 실행 후 즉시 결과 보기:
```bash
.venv/bin/python -m tracks.grokking.train
```

산출물 (실행 후):
- `results/grokking/run_<timestamp>/loss_curve.npz` — `epoch`, `train_loss`, `test_loss`, `train_acc`, `test_acc` numpy arrays
- `results/grokking/run_<timestamp>/checkpoints/ckpt_NNNNNN.pt` — 50개 (매 1000 epoch)

**다음 보고서 갱신 시 채워야 할 항목**:
- 50K 완료까지 실제 wall time
- train_loss가 0 근처로 도달한 epoch (Nanda 예상: ~1,000)
- test_loss가 0 근처로 도달한 epoch = **grokking 발생 시점** (Nanda 예상: ~20,000)
- 최종 test accuracy
- HookedTransformer init이 Nanda 결과 dynamics를 재현하는지 (§3.4 갭 검증)

---

## 6. 다음 단계 — Step 2 회로 분석

[analysis_tracks.md §2.2](../../.claude/plans/analysis_tracks.md) 참조. full run 완료 후:

1. 후반 체크포인트(ckpt_049000.pt 근처)에서 `W_E`, `W_U` 추출 → DFT 적용 → Fourier basis 검증
2. attention pattern 시각화 (epoch별 snapshot)
3. head ablation: 각 head 끄고 test acc 변화 측정

이 단계는 별도 모듈(`tracks/grokking/analysis.py` 등)로 분리 예정.

---

## 7. 변경 정책

본 문서는 P1 트랙 결과만 기록한다. 분석 트랙의 forward 계획은 [.claude/plans/analysis_tracks.md](../../.claude/plans/analysis_tracks.md), 세 트랙 통합 status는 [.claude/plans/_summary.md](../../.claude/plans/_summary.md). 데이터 freeze 사실은 [datasets/_report.md](../../datasets/_report.md).
