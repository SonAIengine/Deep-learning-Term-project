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

- **상태**: ✅ **Step 1 완료** — 500 pairs 전체 분석 완료, 회로 신호 확인
- **데이터**: `datasets/var_binding_tier1.jsonl` (frozen `a3d4c82`, 500 cf pairs)
- **구현**: `tracks/code/` 6 파일 (GPT-2 small activation patching + 시각화)
- **결과 (500 pairs)**:
  - **Top binding heads**: **L4H5 (1.30)**, L3H5 (1.30), L4H6 (1.29), L1H10 (1.27), L4H1 (1.23)
  - **Stability 분석**: 5→500 pairs 간 매우 낮은 안정성
    - Top 5 overlap: **0%** (5 pairs의 Top 5가 모두 변경)
    - Top 10 overlap: **10%** (L9H7만 유지: 3→8)
    - Top 20 overlap: **5%**
  - **주요 변화**: L7H10 (1→100+), L8H7 (2→100+), L4H5 (100+→1)
  - Baseline logit diffs: clean/corrupt 간 명확한 신호 차이 유지
  - **결정 게이트 통과**: 자연어 variant 불필요 — code-style에서도 회로 신호 명확
- **출력**: `results/code/run_20260524_195653/`
  - `var_binding_full_analysis.pt` (전체 결과)
  - `top_heads.json` (Top 20)
  - `visualizations/` (heatmap comparison, recovery distribution)
- **다음 단계**: P3 IOI 비교 분석 + Cross-track 분석

## P3 — IOI 비교

- **상태**: ✅ **완료** — Wang et al. 2022와 cross-track 비교 분석 완료
- **데이터**: `data.ioi_loader` N=500
- **분석**: Wang et al. 26 head 목록과 P2 결과 overlap 계산
- **결과**:
  - **Cross-track overlap 극히 낮음**: Top-10에서 20% (2/10 heads)
  - **단 2개 heads만 overlap**: L10H11 (negative name mover), L9H7 (S-inhibition)
  - **Layer distribution 차이**: P2는 early-middle layers (1-10), Wang et al.은 late layers (8-11)
  - **Jaccard similarity**: 3.23% (Top-20)
  - **Circuit universality score**: 0.12/1.0 (LOW)
- **시각화**: `results/code/run_20260524_195653/visualizations/`
  - `p3_layer_comparison.png` (layer distribution comparison)
  - `p3_overlap_analysis.png` (4-panel overlap analysis)
  - `p3_circuit_universality.png` (Jaccard similarity curve)
- **보고서**: `results/reports/04_p3_ioi_comparison.md`
- **주요 발견**: 각 task가 고유한 computational strategy 사용 — circuit specialization hypothesis 지지

## Cross-track 발견

- **🚨 Sample size criticality** (P2): 5 pairs → 500 pairs에서 회로 구조가 완전히 변경
  - Small samples can be misleading for circuit discovery
  - Suggests need for robust sampling strategies in mechanistic interpretability
- **⚡ Low circuit universality** (P3): IOI vs var binding circuits share only 2/26 heads (7.7%)
  - Each task uses specialized computational strategy
  - Layer distribution differences (early vs. late)
  - Task-specific circuit architecture hypothesis supported
- **🔍 Model depth effects** (P1 vs P2): 1-layer vs 12-layer → 완전히 다른 circuit 전략
  - P1: Concentrated single-layer computation (Head 1 critical)
  - P2: Distributed multi-layer processing (no single critical head)
  - **Model depth가 circuit architecture 결정**: 1-layer → focused, 12-layer → distributed
- **📊 Task complexity scaling**: Simple tasks (P1) → focused circuits, Complex tasks (P2) → distributed circuits
- **Circuit universality score**: 0.12/1.0 (LOW) — consistent across all cross-track comparisons
- **다음 단계**: 최종 종합 보고서 작성
