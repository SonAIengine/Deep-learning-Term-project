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

- **상태**: pending
- **데이터**: `datasets/var_binding_tier{1,2,3}.jsonl` (frozen `a3d4c82`)
- **회로 분석**: TBD — activation patching on Tier 1 cf pair
- **결정 게이트**: 첫 patching 결과로 자연어 variant 필요 여부 판단 ([analysis_tracks.md §1.3](analysis_tracks.md))

## P3 — IOI 비교

- **상태**: pending
- **데이터**: `data.ioi_loader` N=500
- **분석**: Wang et al. 26 head 목록과 P2 결과 overlap 계산

## Cross-track 발견

- TBD: P1 회로 (modular)와 P2 회로 (var binding)의 구조적 공통점
- TBD: Circuit universality score 정의 + 계산
