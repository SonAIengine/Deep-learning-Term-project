# 분석 트랙 종합 결과

> **역할**: P1 / P2 / P3 트랙 결과를 한 페이지로 통합. 분석 진행에 따라 갱신.
> **짝 문서**: [datasets/_report.md](../../datasets/_report.md) (데이터 freeze 사실), [analysis_tracks.md](analysis_tracks.md) (forward 계획).

분석 트랙 결과가 나오면 채워짐. 현재는 stub.

---

## P1 — Grokking 회로 분석

- **상태**: ✅ **Step 1 완료**, ✅ **Step 2 완료** — 50K full run 성공, 회로 분석 완료
- **데이터**: `datasets/modular_{train,test}.pt` (frozen `a3d4c82`)
- **구현**: `tracks/grokking/` 8 파일 (Nanda canonical config: 1-layer, d_model=128, no LN, wd=1.0, full-batch)
- **실행**: 50K epochs, 313.8s (~5.2 min)
- **결과**:
  - Final: train/test acc 100%, loss ~0
  - Grokking point: **epoch 10,200** (test acc > 99%)
  - Test loss peak: epoch 1,300 (loss=27.80)
- **출력**: `results/grokking/run_20260524_012428/` (loss_curve.npz, grokking_curve.png, checkpoints/)
- **Step 2 회로 분석**: ✅ 완료
  - **Fourier basis**: Freq 85/28 상위 (상관계수 0.8742)
  - **Attention pattern**: 모든 head가 a,b에 균등하게 (~50:50) attend
  - **Head ablation**: Head 1 CRITICAL (-62% drop), Head 2 Important (-43%)
  - **결과**: `results/grokking/analysis/` (fourier_scores, attention_patterns, head_ablation)
  - **보고서**: `results/reports/02_grokking_training.md`, `03_circuit_analysis.md`

## P2 — Code Variable Binding

- **상태**: ✅ **Step 1 완료** — Activation patching 성공, 회로 신호 확인
- **데이터**: `datasets/var_binding_tier1.jsonl` (frozen `a3d4c82`)
- **구현**: `tracks/code/` 5 파일 (GPT-2 small activation patching)
- **결과**:
  - Top binding heads: **L7H10 (1.21)**, L8H7 (1.20), L9H7 (1.17), L11H0 (1.17)
  - Baseline logit diffs: clean/corrupt 간 명확한 신호 차이 (0.1~2.6)
  - **결정 게이트 통과**: 자연어 variant 불필요 — code-style에서도 회로 신호 명확
- **출력**: `results/code/run_20260524_185842/` (patching_results.json, head_heatmap.pt)
- **다음 단계**: 더 많은 cf pairs로 확장 + cross-validation

## P3 — IOI 비교

- **상태**: pending
- **데이터**: `data.ioi_loader` N=500
- **분석**: Wang et al. 26 head 목록과 P2 결과 overlap 계산

## Cross-track 발견

- TBD: P1 회로 (modular)와 P2 회로 (var binding)의 구조적 공통점
- TBD: Circuit universality score 정의 + 계산
