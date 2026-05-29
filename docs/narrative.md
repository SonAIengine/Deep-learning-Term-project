# IOI 회로는 코드에서도 살아있는가 — 연구 내용 정리

손성준 | Deep Learning Term Project | 2026 봄학기

본 문서는 연구의 동기부터 결론까지를 서술형으로 정리한 내용 문서다.
"왜 시작했는가"에서 출발해 무엇을 어떻게 했고 무엇을 발견했으며 그것이 무슨 의미인지를
하나의 흐름으로 읽을 수 있도록 구성했다. 각 절에는 해당 결과를 만든 코드와 그림으로의
링크를 달았고, 문서 하단에 프로젝트 자료 전체를 모아두었다.

관련 문서: [상세 결과 리포트](results.md) · [장표용 보고서](report.md) ·
[발표 슬라이드(Marp)](slides.md) · [발표 노트](speaker_notes.md)

---

## 1. 출발점 — 왜 이 연구를 시작했는가

> 쉽게 말하면: AI가 답은 잘 내놓는데 "머릿속에서 어떻게 푸는지"는 아무도 모른다.
> 그래서 그 머릿속 회로를 직접 들여다보기로 했고, "영어를 풀 때 쓰는 머리 부위와
> 코드를 풀 때 쓰는 머리 부위가 같은가?"를 물어보기로 했다.

오늘날의 언어 모델은 입력을 넣으면 그럴듯한 출력을 내놓지만, 그 사이에서 무슨 계산이
일어나는지는 거의 알려져 있지 않다. 모델은 사실상 블랙박스다. Mechanistic
Interpretability는 이 블랙박스를 열어, 모델 내부의 attention head와 뉴런이 실제로
어떤 알고리즘을 수행하는지를 회로 수준에서 규명하려는 분야다.

이 분야에는 한 가지 핵심 가설이 있다. Circuit Universality, 즉 서로 다른 과제나 서로
다른 양식(modality)에서도 동일한 회로가 형성된다는 가설이다(Olah 2020, Chughtai et al.
2023). 만약 이것이 참이라면, 모델은 표면적인 단어 패턴이 아니라 추상적인 계산 구조를
학습한 것이고, 한 영역의 학습이 다른 영역의 능력에도 영향을 준다는 뜻이 된다.

본 연구가 던진 질문은 이 가설을 modality 사이에서 검증하는 것이다.

> 트랜스포머는 자연어의 개체 추적과 코드의 변수 추적을 같은 회로로 처리하는가,
> 아니면 서로 다른 회로로 처리하는가?

이 질문은 어느 쪽으로 답이 나와도 의미가 있다. 같은 회로라면 모델이 양식을 초월한
추상적 알고리즘을 학습했다는 증거이고, 다른 회로라면 모델이 양식별로 특화된 회로를
따로 만든다는 증거다. 실패가 없는 설계다.

제안서 원문은 [docs/candidates/sonsj-proposal.md](candidates/sonsj-proposal.md)에 있다.

---

## 2. 비교 대상 — 구조가 동일한 두 과제

> 쉽게 말하면: "철수와 영희가 있었는데 철수가 누구에게 줬다 → 영희"라는 영어 문제와,
> "x=5; z=9; a=z; a=? → 9"라는 코드 문제는 겉모습만 다를 뿐 "둘 중 누구를 가리키나"라는
> 똑같은 퀴즈다. 영어 쪽은 이미 정답 회로(어떤 부품이 일하는지)가 밝혀져 있으니,
> 그걸 자로 삼아 코드 쪽과 맞춰본다.

검증을 위해 표면은 다르지만 추론 구조가 동일한 두 과제를 선택했다.

첫째는 자연어의 IOI(Indirect Object Identification) 과제다. "Mary and John went to
the store. John gave a drink to ___"에서 모델은 "Mary"를 예측해야 한다. 두 개체 중
바인딩 관계로 정답을 골라야 하는 문제다. 2022년 Wang 연구진(arXiv:2211.00593)이 GPT-2
small에서 이 회로를 완전히 분석해, 26개의 attention head가 7가지 역할로 협업한다는
것을 밝혔다.

둘째는 코드의 변수 바인딩 과제다. `x=5; z=9; a=z; a=`에서 모델은 9를 예측해야 한다.
두 변수 중 바인딩 관계로 정답값을 추적하는 문제다. 본 연구는 이 과제를 규칙 기반으로
자동 생성했으며, 난이도에 따라 세 단계로 나눴다.

| 데이터셋 | 규모 | 형태 | 용도 |
|---|---|---|---|
| var_binding Tier 1 | 1,500 (clean/corrupt 쌍) | `x=5; z=9; a=z; a=?` | 주 분석 (patching) |
| var_binding Tier 2 | 1,000 | 더 긴 코드, distractor 다수 | 일반화 검증 |
| var_binding Tier 3 | 500 | 멀티홉 `a=b; b=c; c=5; a=?` | 일반화 (정확도) |
| modular addition | 12,769 (113²) | `(a + b) mod 113` | grokking 학습 |

데이터 생성·검증 기준은 [docs/datasets.md](datasets.md)에 정리되어 있다.

표면 형태는 영어 문장과 파이썬 코드로 완전히 다르지만, 본질은 "여러 후보 중 어느
것을 가리키는가"라는 동일한 구조다. 그리고 IOI 회로가 이미 26개 head 단위로 분석되어
있으므로, 이를 기준선으로 삼아 코드 과제의 head와 직접 대응시킬 수 있다.

IOI 회로의 head 분류는 다음과 같다. 본 연구의 비교는 이 7개 클래스를 단위로 한다.

| 분류 | 약어 | 역할 | 단계 |
|---|---|---|---|
| Duplicate Token Head | DTH | 중복 토큰 탐지 | 입력 |
| Previous Token Head | PTH | 직전 토큰 복사 | 입력 |
| Induction Head | IH | 반복 패턴 탐지 | 중간 |
| S-Inhibition Head | SIH | 중복 개체 억제 | 구조 |
| Name Mover Head | NMH | 정답을 출력으로 이동 | 출력 |
| Backup Name Mover | BNMH | NMH 보조 | 출력 |
| Negative Name Mover | NNMH | 정답 logit 억제 | 출력 |

---

## 3. 분석 방법과 실험 설정

> 쉽게 말하면: 회로 부품(attention head)의 역할을 알아내는 방법은 외과수술과 비슷하다.
> 부품 하나만 정상 상태로 바꿔치기해서 답이 살아나면 그 부품이 핵심이고(patching),
> 부품을 아예 꺼서 성능이 떨어지면 그것도 핵심이며(ablation), 부품이 정답을 밀어주는지
> 깎아내리는지 방향까지 재는 것(attribution)이다. 이걸 세 종류 모델에 적용했다.

분석은 세 가지 기법을 중심으로 했다.

Activation Patching은 모델을 정답 입력(clean)과 변형 입력(corrupt) 두 번 실행하고,
특정 head 하나의 출력(`blocks.{L}.attn.hook_z`)만 clean에서 corrupt로 이식해 정답
신호가 얼마나 회복되는지 측정한다. head 하나하나의 인과적 기여를 분리하는 기법이다.
회복량은 다음과 같이 정의한다.

```
recovery = (패치 후 logit_diff − corrupt logit_diff) / (clean logit_diff − corrupt logit_diff)
```

Direct Logit Attribution은 한 단계 더 나아가, 각 head의 마지막 위치 출력 잔차를 정답
방향 W_U[정답] − mean(W_U[방해답])에 투영해, 정답 점수를 올리는지 내리는지 부호까지
측정한다.

Head Ablation은 head를 0으로 만들고 성능 변화를 본다.

평가 지표 logit difference는 정답 토큰 logit에서 방해답 토큰 logit 평균을 뺀 값이다.

분석 대상은 세 모델이다.

| 모델 | 규모 | 특징 | 역할 |
|---|---|---|---|
| GPT-2 small | 124M, 12층 × 12head (144) | Wang 2022 기준선 보유 | 주 분석 |
| GPT-2 medium | 355M, 24층 × 16head (384) | 같은 계열 큰 모델 | scale 효과 |
| Pythia-160M | 12층 × 12head | rotary, parallel attn+MLP, Pile 학습 | 아키텍처 효과 |

전체 실험은 H100 GPU 한 장으로 약 3시간이 걸렸으며, 모든 patching 평가는 500개
counterfactual pair에 대해 전 head를 대상으로 수행했다.

관련 코드: [patching.py](../src/binding/patching.py) ·
[baseline.py](../src/binding/baseline.py) ·
[compare_ioi.py](../src/binding/compare_ioi.py)

---

## 4. 사전 검증 — Grokking으로 도구의 신뢰성 확인

> 쉽게 말하면: 본 게임에 들어가기 전에 "내 측정 도구가 멀쩡한지" 시험한 단계다.
> 정답 푸는 방법이 이미 알려진 쉬운 문제(덧셈)에 도구를 써보고, 알려진 정답 회로를
> 그대로 찾아내는지 확인했다. 도구가 정답을 맞혔으니, 정답을 모르는 진짜 문제(코드)에
> 써도 믿을 수 있다는 뜻이다.

본 분석에 앞서, 사용할 도구가 신뢰할 만한지 검증할 필요가 있었다. 정답 알고리즘이
알려진 과제에서 도구가 그 알고리즘을 정확히 찾아낸다면, 정답을 모르는 코드 과제에도
적용할 수 있다.

이를 위해 2-layer 트랜스포머(d_model=128, n_heads=4, d_mlp=512, LayerNorm 없음, bias
없음, 파라미터 422,784개)를 modular addition(p=113)에 학습시켜 grokking 현상을
재현했다. 최적화는 AdamW(lr=1e-3, weight_decay=1.0, betas=(0.9, 0.98)), full-batch로
40,000 스텝을 돌렸고 H100에서 101초가 걸렸다.

grokking은 모델이 처음에는 암기만 하다가(학습 정확도는 높지만 테스트는 낮음) 어느
순간 갑자기 일반화로 전환되는 현상이다. 본 실험에서 최종 학습 정확도는 1.000, 테스트
정확도는 0.9928이었고, 두 곡선이 시차를 두고 수렴하는 전형적인 grokking 패턴이
관찰됐다. 학습 후반(약 35,600 스텝)에 일시적 불안정(slingshot, 테스트 0.93으로 잠시
하락)이 있었으나 즉시 회복했다.

학습된 모델을 분석한 결과:
- Fourier 분해: 임베딩이 소수 주파수 모드(k ≈ 14, 31, 35, 52)에 집중
- Head/MLP ablation: 특정 컴포넌트 제거 시 테스트 손실 급증 → 핵심 회로 식별

이는 Nanda 2023(arXiv:2301.05217)이 밝힌 삼각함수 기반 modular addition 회로를 그대로
재현한 것이다. 정답을 아는 과제에서 도구가 정답을 맞혔으므로, 이후 코드 과제 분석
결과를 신뢰할 근거를 확보했다.

관련 그림: [학습 곡선 GIF](../results/analysis/grokking_full/loss_curve.gif) ·
[attention 진화 GIF](../results/analysis/grokking_full/attention_evolution.gif) ·
[Fourier 분해](../results/analysis/grokking_full/fourier_final.png) ·
[학습 곡선 정적 그림](../results/analysis/grokking_full/timeseries.png)
관련 코드: [train.py](../src/grokking/train.py) · [model.py](../src/grokking/model.py) ·
[analysis.py](../src/grokking/analysis.py)
학습 로그: [wandb run n4bnqrak](https://wandb.ai/sonsj97-plateer/grokking-circuits/runs/n4bnqrak)

---

## 5. 핵심 결과 1 — 코드 회로의 상당 부분이 IOI 회로였다

> 쉽게 말하면: 코드 문제를 풀 때 가장 열심히 일하는 부품 5개를 뽑았더니, 그중 4개가
> 영어 문제(IOI)에서 일하던 바로 그 부품이었다. 우연히 겹칠 확률보다 2배 넘게 겹쳤다.
> 즉 모델은 코드를 풀려고 새 회로를 만든 게 아니라, 영어용 회로를 그대로 가져다 썼다.

GPT-2 small에 코드 변수 바인딩 문제(Tier 1) 500개를 넣고 144개 head를 전수
patching했다. 기준선 측정 결과는 다음과 같다.

| 지표 | Clean | Corrupt | 차이 |
|---|---|---|---|
| logit difference 평균 | +0.182 | +0.097 | +0.086 |
| 표준편차 | 1.457 | 1.522 | — |
| Top-1 정확도 | 0.8% | 2.0% | — |

Top-1 정확도는 낮지만, logit difference 신호는 patching으로 회로를 분리하기에 충분하다.

전수 patching 결과, 가장 중요한 head 5개 중 4개가 Wang 2022의 IOI 26개 head에 정확히
포함되어 있었다.

| 순위 | Head | 회복량 | IOI 분류 |
|---|---|---|---|
| 1 | L10H7 | +0.491 | NNMH (정답 억제) |
| 2 | L10H2 | +0.453 | BNMH (정답 보조) |
| 3 | L6H1 | +0.206 | 비IOI |
| 4 | L3H0 | +0.159 | DTH (중복 탐지) |
| 5 | L10H10 | +0.109 | BNMH |

상위 26개 코드 head와 IOI 26개 head의 교집합은 10개였다. 무작위 기대값
26 × 26 / 144 ≈ 4.7개의 약 2.1배다. 우연으로 보기 어려운 중첩이다.

관련 그림: [head별 회복 heatmap (IOI 클래스 표시)](../results/binding_compare/heatmap_classes.png) ·
[코드 vs IOI 대조](../results/binding_compare/code_vs_ioi.png) ·
[원본 patching heatmap](../results/binding_patching/head_mean.png)
관련 데이터: [baseline.json](../results/binding_baseline.json) ·
[top_heads.json](../results/binding_patching/top_heads.json)

---

## 6. 핵심 결과 2 — 전부가 아니라 일부만 재사용된다 (선택적 재사용)

> 쉽게 말하면: 영어용 회로 전체가 통째로 복사된 건 아니다. "정답을 출력으로 내보내는"
> 마무리 단계 부품들은 코드에서도 그대로 쓰였지만, "중복된 이름을 눌러주는" 중간 단계
> 부품(SIH)은 안 쓰였다. 골라서 일부만 재사용한 것이라 "선택적 재사용"이라 부른다.

IOI의 7가지 head 역할을 클래스별로 묶어 평균 회복량을 계산하자 뚜렷한 패턴이 나타났다.

| IOI 분류 | head 수 | 평균 회복량 | 최대 | 전이 여부 |
|---|---|---|---|---|
| NNMH | 2 | +0.283 | +0.491 | 강하게 전이 |
| BNMH | 8 | +0.070 | +0.453 | 전이 |
| NMH | 3 | +0.042 | +0.065 | 전이 |
| DTH | 3 | +0.011 | +0.159 | 부분 전이 |
| IH | 4 | −0.012 | +0.098 | 거의 없음 |
| PTH | 2 | −0.014 | +0.023 | 거의 없음 |
| SIH | 4 | −0.049 | +0.032 | 전이 안 됨 |
| 비IOI | 118 | +0.003 | — | 기준선 |

정답을 출력으로 옮기거나 억제하는 출력 단계 head(NNMH, BNMH, NMH)는 전이됐지만,
구조적 단계인 S-Inhibition Head는 전이되지 않았고 오히려 음수였다. IOI 회로가 통째로
복사된 것이 아니라 일부 부품만 선택적으로 재사용된 것이다. 이를 선택적 회로 재사용
(Selective Reuse)이라 부른다.

(클래스 단위 평균은 위와 같이 명확하지만, head 단위 순위 상관은 약하다.
Spearman ρ = +0.106, p = 0.61. 따라서 결론은 클래스 단위 평균에 근거한다.)

관련 그림: [클래스별 전이 + Top-K enrichment](../results/binding_compare/universality_summary.png) ·
[회로 schematic](../results/binding_compare/circuit_diagram.png)
관련 데이터: [universality_report.json](../results/binding_compare/universality_report.json)
관련 코드: [compare_ioi.py](../src/binding/compare_ioi.py) · [figures.py](../src/binding/figures.py)

---

## 7. 검증 — SIH 비전이는 위치 문제가 아니었다

> 쉽게 말하면: "SIH가 안 쓰인 게 혹시 내가 엉뚱한 자리를 봐서 그런 거 아닐까?"를
> 의심해서, 문장의 모든 자리를 다 뒤져봤다. 어느 자리에서도 도움이 안 됐다. 결론은
> 코드 문제엔 "중복된 이름 누르기"라는 할 일 자체가 없어서 SIH가 놀고 있는 것이다.

SIH가 음수로 나온 것이 단지 패치 위치를 잘못 잡았기 때문일 가능성을 배제하기 위해,
SIH 4개 head(L7H3, L7H9, L8H6, L8H10)를 길이-14 프롬프트의 14개 토큰 위치 전부에서
패치했다(n=250).

| SIH Head | 절대값 최대 위치 | 값 |
|---|---|---|
| L7H3 | pos 13 (final =) | −0.093 |
| L7H9 | pos 13 | −0.019 |
| L8H6 | pos 13 | −0.204 |
| L8H10 | pos 13 | −0.172 |

결과는 모든 위치에서 0이거나 음수였고, L8H6는 마지막 위치에서 −0.20까지 떨어졌다.
즉 어디서 패치해도 효과가 없거나 오히려 해롭다.

이로써 SIH 비전이가 위치 문제가 아니라 과제 구조의 문제임이 확인됐다. IOI에서 SIH는
같은 이름이 두 번 등장할 때 첫 번째를 억제하는 역할인데, 코드 과제에는 그런 중복
구조가 없으므로 SIH가 수행할 일 자체가 존재하지 않는다.

관련 그림: [SIH 위치별 회복](../results/binding_position/sih_position_recovery.png) ·
[Top head 위치별 회복](../results/binding_position/position_recovery.png)
관련 코드: [sih_position_patching.py](../src/binding/sih_position_patching.py) ·
[position_patching.py](../src/binding/position_patching.py)

---

## 8. 가장 강한 증거 — 역할의 부호까지 보존된다

> 쉽게 말하면: 어떤 부품은 IOI에서 일부러 정답을 "깎아내리는" 특이한 역할을 한다.
> 놀랍게도 코드에서도 똑같이 정답을 깎아내렸다. 단순히 같은 부품이 켜진 정도가 아니라,
> "정답을 깎는다"는 그 독특한 성격까지 플러스·마이너스 부호 그대로 넘어왔다. 회로가
> 겉모습이 아니라 '하는 일' 수준에서 진짜로 같다는 가장 강력한 증거다.

지금까지의 결과는 "어떤 head가 중요한가"까지였다. 그 head가 정답을 밀어주는지
깎아내리는지를 확인하기 위해, 각 head 출력이 정답 점수에 직접 기여하는 정도를 부호까지
측정했다(Direct Logit Attribution, n=500). 전체 head의 직접 기여 합계는 +2.139였다.

| Head | IOI 역할 | 직접 기여 | z-score | 부호 일치 |
|---|---|---|---|---|
| L10H2 | BNMH (정답 강화) | +1.010 | +3.3 | 일치 |
| L10H7 | NNMH (정답 억제) | −0.305 | −1.9 | 일치 |
| L3H0 | DTH (간접 경로) | −0.018 | −0.5 | 직접 효과 없음 |
| L9H9 | NMH (정답 강화) | +0.303 | — | 일치 |

핵심은 L10H7이다. 이 head는 IOI에서 정답을 의도적으로 억제하는 Negative Name Mover인데,
코드 과제에서도 −0.31로 정답을 똑같이 깎아내렸다. 반대로 L10H2(BNMH)는 +1.01로 정답을
강하게 밀었다. 단순히 같은 head가 재활성화된 것이 아니라, "정답을 억제한다" 혹은
"정답을 민다"는 독특한 기능이 부호까지 그대로 코드로 넘어왔다. 회로가 표면을 넘어
기능 수준에서 보존된다는 가장 직접적인 증거다.

관련 그림: [head별 직접 기여 heatmap](../results/binding_logit_attr/logit_attr_heatmap.png)
관련 데이터: [logit_attr/summary.json](../results/binding_logit_attr/summary.json)
관련 코드: [logit_attribution.py](../src/binding/logit_attribution.py)

---

## 9. 회로의 크기 — 18%의 head가 87%의 일을 한다

> 쉽게 말하면: 부품이 144개나 되지만, 그중 IOI 부품 26개(약 18%)만 빼버려도 코드
> 푸는 능력의 87%가 사라진다. 소수의 핵심 부품이 거의 모든 일을 하고 있다는 뜻이고,
> 코드용 회로는 사실상 영어용 회로의 한 부분일 뿐이다.

이 회로가 모델 전체에서 차지하는 비중을 측정했다. 모든 attention head를 끄면 정답
신호가 0.133 감소한다(attention 전체 기여 = clean 0.182 − empty 0.049). 그런데 IOI
26개 head만 제거하면 — 전체 144개의 18%만 끈 것인데 — 그중 0.116이 사라졌다. 87%다.

| 제거 대상 | head 수 | logit difference 감소 | 무작위 K 감소 | 배수 |
|---|---|---|---|---|
| top3 | 3 | +0.054 | +0.031 ± 0.070 | 1.7배 |
| top5 | 5 | +0.051 | +0.005 ± 0.038 | 약 10배 |
| top10 | 10 | +0.065 | +0.035 ± 0.043 | 1.9배 |
| ioi26 | 26 | +0.116 | +0.046 ± 0.054 | 2.5배 |

코드 변수 바인딩에 쓰이는 attention 회로는 사실상 IOI 회로의 부분집합인 셈이다.

(반대 방향 검증, 즉 상위 head만 남기고 나머지를 모두 끄는 sufficiency 방식은 실패했다.
144개 중 141개를 끄는 것은 모델을 정상 분포 밖으로 밀어내는 과격한 개입이라 잔차
스트림이 깨지기 때문이다. 따라서 제거 기반 necessity 검증이 이 경우 더 신뢰할 수 있다.)

관련 그림: [necessity test](../results/binding_minimal_circuit/necessity.png) ·
[sufficiency (실패 사례)](../results/binding_minimal_circuit/sufficiency.png)
관련 데이터: [minimal_circuit/results.json](../results/binding_minimal_circuit/results.json)
관련 코드: [minimal_circuit.py](../src/binding/minimal_circuit.py)

---

## 10. 일반화 — 다른 모델에서는 어떤가

> 쉽게 말하면: "이거 GPT-2 하나만 그런 거 아냐?"를 막으려고 다른 모델 두 개에 똑같이
> 돌려봤다. 공통점은, 어느 모델이든 소수 부품에 일이 몰리고 그 부품들이 정말 중요하다는
> 것. 차이점은, 그 부품이 몇 번째 층 어디에 있는지는 모델마다 제각각이라는 것이다.

단일 모델의 결과라는 한계를 넘기 위해 두 모델을 추가로 분석했다. Tier 2 데이터로의
일반화도 함께 확인해, L10H2(BNMH)가 노이즈가 많은 Tier 2에서도 견고하게 작동함을
확인했다(ablation 감소량 Tier 1 +0.053 → Tier 2 +0.111).

| 측면 | GPT-2 small | GPT-2 medium | Pythia-160M |
|---|---|---|---|
| 구조 | 12층 × 12head | 24층 × 16head | 12층 × 12head (rotary) |
| Clean / Corrupt / Gap | +0.182 / +0.097 / +0.086 | +0.156 / −0.050 / +0.206 | +0.157 / +0.019 / +0.138 |
| Attention 전체 기여 | +0.133 | +0.136 | +0.145 |
| 최대 회복 head | L10H7 (+0.49) | L17H12 (+5.78) | L5H9 (+1.02) |
| 우세 층 (상대 깊이) | L10-11 (83-92%) | L17-18 (71-75%) | L5-6 (42-50%) |
| small과 head 좌표 중첩 | — | (폭 다름) | 1/10 |

소수 head에 집중되는 성질, 그리고 그 head들이 인과적으로 중요하다는 성질은 세 모델
모두에서 보존됐다. 그러나 그 head들이 몇 번째 층에 있는지, 정확히 어느 좌표인지는
모델마다 달랐다. Pythia는 중간층, GPT-2 계열은 후반층에 몰려 있었고, head 좌표는 거의
겹치지 않았다(무작위 기대 0.7개 대비 1개).

관련 그림: [Pythia heatmap](../results/binding_pythia/pythia_heatmap.png) ·
[Pythia necessity](../results/binding_pythia/pythia_necessity.png) ·
[3-모델 층 분포 비교](../results/binding_gpt2med/layer_profile_comparison.png) ·
[Tier 일반화](../results/binding_tier_generalize/tier_generalize.png)
관련 데이터: [pythia/report.json](../results/binding_pythia/report.json) ·
[gpt2med/report.json](../results/binding_gpt2med/report.json)
관련 코드: [pythia_replication.py](../src/binding/pythia_replication.py) ·
[tier_generalize.py](../src/binding/tier_generalize.py)

---

## 11. 예상치 못한 발견 — 큰 모델의 Hydra Effect

> 쉽게 말하면: 큰 모델에서 핵심 부품을 꺼봤더니 성능이 떨어지기는커녕 오히려 좋아졌다.
> 머리를 자르면 두 개가 자라는 히드라처럼, 큰 모델엔 백업 부품이 많아서 끄면 다른 게
> 대신 일하고, 게다가 '정답을 깎던 부품'까지 같이 꺼지니 점수가 올라간 것이다.
> 교훈: 큰 모델에서 "껐더니 점수 올랐으니 쓸모없는 부품"이라 판단하면 완전히 틀린다.

GPT-2 medium에서 가장 흥미로운 결과가 나왔다. 가장 중요한 head들을 제거하자 성능이
떨어지는 것이 아니라 오히려 올라갔다.

| 제거 대상 | logit difference 변화 | 무작위 K |
|---|---|---|
| 상위 3개 | −0.031 (성능 향상) | +0.001 ± 0.016 |
| 상위 5개 | −0.079 (성능 향상) | −0.007 ± 0.012 |
| 상위 10개 | −0.121 (성능 향상) | +0.017 ± 0.044 |

이는 Wang 2022가 IOI에서 보고한 Backup Name Mover의 hydra effect가 큰 모델에서
과보상 형태로 증폭된 것이다. 상위 head 중 정답을 억제하는 suppressor(NNMH 계열)가
포함되어 있어, 이들을 제거하면 방해답 억제가 풀리면서 정답 점수가 상대적으로 오른다.
동시에 큰 모델에는 백업 head가 풍부해 주요 head를 꺼도 다른 head가 일을 이어받는다.
참고로 무작위 head 제거는 이런 효과가 없어, 통제군 역할을 한다.

이는 실용적으로 중요한 경고다. 큰 모델을 분석할 때 "head를 껐더니 성능이 올랐으니 이
head는 불필요하다"고 해석하면 완전히 틀린다. 모델이 클수록 회로 redundancy가 증가하므로
단순 ablation이 아니라 path patching 같은 정교한 도구가 필요하다.

관련 그림: [GPT-2 medium heatmap](../results/binding_gpt2med/gpt2med_heatmap.png)
관련 코드: [gpt2med_replication.py](../src/binding/gpt2med_replication.py)

---

## 12. 종합 — Universality의 네 축 분해

> 쉽게 말하면: "회로가 전이된다/안 된다"는 식의 단순한 이분법으로는 설명이 안 된다.
> 네 가지로 쪼개 보면, '소수에 집중되나'와 '정말 중요한가'는 어디서나 같지만,
> '몇 번째 층에 있나'와 '정확히 어느 부품이냐'는 모델마다 다르다. 즉 같은 일을 같은
> 원리로 하되, 그 일을 하는 '위치'는 모델마다 다르다.

이 모든 결과를 종합하면, 회로가 모델을 넘어 전이되는지를 "된다/안 된다"로 단순화할 수
없다. 본 연구는 universality를 네 개의 축으로 분해한다.

| 축 | 내용 | 보존 여부 |
|---|---|---|
| 1. 기능적 집중도 | 소수 head에 집중 | 세 모델 모두 보존 |
| 2. 인과적 필요성 | 상위 head가 실제로 중요 | 세 모델 모두 보존 |
| 3. 층 단계 위치 | 작동하는 깊이 | 같은 계열 내 보존, 계열 간 불일치 |
| 4. head 좌표 | 정확한 층/head 번호 | 보존 안 됨 (무작위 수준) |

결론은 명확하다. 회로의 universality는 모델의 겉모습, 즉 몇 번째 층의 몇 번 head인가가
아니라 기능과 역할이라는 추상적 수준에서 성립한다. 같은 원리, 다른 해부학적 위치.
이것이 본 연구의 핵심 주장이다.

---

## 13. 이 연구의 의미

> 쉽게 말하면: 학술적으로는 "전이된다/안 된다"는 거친 그림을 네 축으로 정밀하게
> 다듬었다. 실용적으로는 두 가지가 중요하다 — (1) 코드를 가르치면 영어 능력도 바뀔 수
> 있다(회로를 공유하니까), (2) 큰 모델은 부품 하나 꺼서 판단하면 안 된다(히드라 때문).

학술적으로, 기존의 universality 논의는 "회로가 전이된다/안 된다"는 이분법에 머물렀다.
본 연구는 이를 부품 단위로, 그리고 네 개의 축으로 분해해 가설을 더 정밀한 형태로
다듬었다. 또한 같은 좌표가 아니라 같은 역할이 전이된다는 기능적 universality에 직접적인
데이터 근거를 제공했다.

실용적으로 두 가지 함의가 있다. 첫째, 코드 학습이 자연어 능력에 영향을 줄 수 있다.
출력 단계 회로가 공유되기 때문이다. 따라서 AI 안전성 검증에서 코드와 자연어를 분리해
점검하는 접근이 충분하지 않을 수 있다. 둘째, 큰 모델 해석에 단순 ablation은 위험하다.
hydra effect로 인한 오해석 가능성 때문에 더 정교한 인과 분석 도구가 필요하다.

---

## 14. 선행 연구 대비 차별화

> 쉽게 말하면: 남들은 "영어에서 회로를 찾았다" 또는 "회로가 비슷하더라" 정도였다.
> 우리는 그 회로를 코드로 처음 끌고 와서, 부품별로·모델별로 무엇이 옮겨가고 무엇이
> 안 옮겨가는지를 숫자로 쪼개 보였다. 특히 부호 보존, 선택적 재사용, 히드라 세 가지는
> 우리가 처음 정량화한 것이다.

| 선행 연구 | 한 일 | 본 연구가 추가한 것 |
|---|---|---|
| Wang 2022 (IOI) | 자연어에서 26 head 회로 발견 | 코드에서도 작동하는지 처음 검증 |
| Feng & Steinhardt 2023 | 변수 바인딩 메커니즘 분석 | IOI 회로와 head 단위 직접 매핑 |
| Chughtai 2023 | toy model에서 universality 제안 | 실제 모델/과제/modality 간 정량 검증 |
| 일반 universality 논의 | "전이/비전이" 이분법 | 4축 분해 framework |

본 연구만의 고유한 발견은 세 가지다. 첫째, 정답을 억제하는 역할이 부호까지 보존된다.
둘째, 출력 단계는 전이되지만 구조 단계는 과제 구조의 부재로 전이되지 않는 비대칭성이
존재한다. 셋째, 모델이 클수록 hydra effect가 강해진다.

한 문장으로 정리하면 — 기존 연구가 회로를 "있다/없다"로 봤다면, 본 연구는 회로를
부품, 기능, 인과, 해부학이라는 축으로 쪼개 무엇이 도메인과 모델을 넘어 전이되는지를
처음으로 다차원으로 정량화했다.

---

## 15. 한계와 향후 과제

> 쉽게 말하면: 히드라가 왜 생기는지 더 파야 하고, 멀티홉 문제는 일부만 봤고, 더 큰
> 모델로의 확장은 숙제다. 그리고 "AI가 진짜 생각하느냐" 같은 큰 질문엔 답 안 한다.
> 우리가 답하는 건 "GPT-2 수준에서 변수 추적이 어디서 어떻게 일어나는가"까지다.

hydra effect가 왜 일어나는지, 정확히 어떤 백업 head가 활성화되는지는 path patching을
통한 추가 분석이 필요하다. 멀티홉 변수 바인딩(Tier 3)은 distractor 정의의 어려움으로
일부만 분석했다. Llama, Mistral 등 다른 계열의 더 큰 모델로 확장하면 framework의
일반성을 강화할 수 있다.

본 연구는 "언어 모델이 진정한 추론을 하는가" 같은 큰 질문에는 답하지 않는다. 답할 수
있는 범위는 GPT-2 규모에서 변수 바인딩이 어디서 어떻게 처리되며, 그것이 IOI 회로와
어떤 관계이고, 그 관계가 모델에 따라 어떻게 변하는가까지다. 작지만 명확하고 측정
가능한 발견이다.

---

## 16. 결론

> 쉽게 말하면: GPT-2는 영어 풀던 회로를 코드에서도 다시 썼다. 단, 통째로가 아니라
> 골라서, 그것도 부품의 성격(부호)까지 그대로. 그리고 이 "재사용"은 부품의 정확한
> 위치가 아니라 '하는 일' 수준에서 성립한다. 덤으로 큰 모델일수록 회로가 끈질겨진다는
> 히드라 현상도 발견했다.

본 연구는 GPT-2 small이 자연어 IOI 과제에서 사용하던 회로를 코드 변수 바인딩에서도
재사용한다는 것을 보였다. 그 재사용은 전부가 아니라 선택적이며, 출력 단계 head는
기능적 부호까지 보존된 채 전이되는 반면 구조 단계 head는 전이되지 않는다. 세 모델로의
확장을 통해, universality가 해부학적 위치가 아니라 기능 수준에서 성립함을 보이는 4축
분해 framework를 제시했다. 그 과정에서 scale에 따라 강해지는 hydra effect라는 예상치
못한 현상도 발견했다.

---

## 프로젝트 자료 (전체 링크)

모든 경로는 GitHub의 `son` 브랜치(또는 통합본 `main`) 기준이다.

### 문서
- [docs/results.md](results.md) — 상세 결과 리포트 (수치·해석 전부)
- [docs/report.md](report.md) — 장표용 보고서 (PPTX 변환용)
- [docs/slides.md](slides.md) — Marp 발표 슬라이드
- [docs/speaker_notes.md](speaker_notes.md) — 발표 노트 + 예상 Q&A
- [docs/candidates/sonsj-proposal.md](candidates/sonsj-proposal.md) — 연구 제안서
- [docs/datasets.md](datasets.md) — 데이터셋 명세

### Grokking (Step 2)
- 코드: [train.py](../src/grokking/train.py) · [model.py](../src/grokking/model.py) ·
  [analysis.py](../src/grokking/analysis.py) · [animate.py](../src/grokking/animate.py) ·
  [wandb_export.py](../src/grokking/wandb_export.py)
- 그림: [loss_curve.gif](../results/analysis/grokking_full/loss_curve.gif) ·
  [attention_evolution.gif](../results/analysis/grokking_full/attention_evolution.gif) ·
  [timeseries.png](../results/analysis/grokking_full/timeseries.png) ·
  [fourier_final.png](../results/analysis/grokking_full/fourier_final.png) ·
  [attn_final.png](../results/analysis/grokking_full/attn_final.png)
- wandb: [run n4bnqrak](https://wandb.ai/sonsj97-plateer/grokking-circuits/runs/n4bnqrak)
- wandb 차트(repo 저장): [dashboard](../results/wandb_charts/dashboard.png) ·
  [accuracy](../results/wandb_charts/accuracy.png) ·
  [grokking_gap](../results/wandb_charts/grokking_gap.png) ·
  [loss_log](../results/wandb_charts/loss_log.png)

### Code Variable Binding (Step 3)
- 코드: [baseline.py](../src/binding/baseline.py) · [patching.py](../src/binding/patching.py) ·
  [compare_ioi.py](../src/binding/compare_ioi.py) · [figures.py](../src/binding/figures.py)
- 그림: [heatmap_classes.png](../results/binding_compare/heatmap_classes.png) ·
  [universality_summary.png](../results/binding_compare/universality_summary.png) ·
  [circuit_diagram.png](../results/binding_compare/circuit_diagram.png) ·
  [code_vs_ioi.png](../results/binding_compare/code_vs_ioi.png) ·
  [head_mean.png](../results/binding_patching/head_mean.png)
- 데이터: [baseline.json](../results/binding_baseline.json) ·
  [universality_report.json](../results/binding_compare/universality_report.json) ·
  [top_heads.json](../results/binding_patching/top_heads.json)

### 심화 실험
- A (SIH 비전이): [sih_position_patching.py](../src/binding/sih_position_patching.py) ·
  [sih_position_recovery.png](../results/binding_position/sih_position_recovery.png)
- B (Logit Attribution): [logit_attribution.py](../src/binding/logit_attribution.py) ·
  [logit_attr_heatmap.png](../results/binding_logit_attr/logit_attr_heatmap.png)
- F (Minimal Circuit): [minimal_circuit.py](../src/binding/minimal_circuit.py) ·
  [necessity.png](../results/binding_minimal_circuit/necessity.png)
- D (Pythia): [pythia_replication.py](../src/binding/pythia_replication.py) ·
  [pythia_heatmap.png](../results/binding_pythia/pythia_heatmap.png)
- D2 (GPT-2 medium): [gpt2med_replication.py](../src/binding/gpt2med_replication.py) ·
  [layer_profile_comparison.png](../results/binding_gpt2med/layer_profile_comparison.png) ·
  [gpt2med_heatmap.png](../results/binding_gpt2med/gpt2med_heatmap.png)
- 위치/일반화: [position_patching.py](../src/binding/position_patching.py) ·
  [tier_generalize.py](../src/binding/tier_generalize.py)

### 저장소
- son 브랜치: https://github.com/yesulmin-danbaaam/Deep-learning-Term-project/tree/son
- 통합본 main: https://github.com/yesulmin-danbaaam/Deep-learning-Term-project/tree/main

---

## 참고문헌

- Nanda et al. 2023, Progress Measures for Grokking via Mechanistic Interpretability, ICLR. [arXiv:2301.05217](https://arxiv.org/abs/2301.05217)
- Wang et al. 2022, Interpretability in the Wild: a Circuit for IOI in GPT-2 small, ICLR. [arXiv:2211.00593](https://arxiv.org/abs/2211.00593)
- Chughtai et al. 2023, A Toy Model of Universality, ICML. [arXiv:2302.03025](https://arxiv.org/abs/2302.03025)
- Feng & Steinhardt 2023, How do Language Models Bind Entities in Context. [arXiv:2310.17191](https://arxiv.org/abs/2310.17191)
- Elhage et al. 2021, A Mathematical Framework for Transformer Circuits, Anthropic. [link](https://transformer-circuits.pub/2021/framework/index.html)
- Bereska & Gavves 2024, Mechanistic Interpretability for Transformer-Based LMs (Review). [arXiv:2407.02646](https://arxiv.org/abs/2407.02646)
