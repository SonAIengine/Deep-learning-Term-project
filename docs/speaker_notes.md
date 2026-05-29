# 발표 노트 — Speaker Notes

> `docs/slides.md` (Marp 23장) 동반 노트. 슬라이드별 화자 멘트, 시간 분배, 예상 Q&A.

**총 시간 예산: 15분** (발표 12분 + Q&A 3분 기준)

| 시간 | 슬라이드 | 핵심 메시지 |
|---:|---|---|
| 0:00 | 1. 표지 | 한 줄 소개 + 본인 |
| 0:30 | 2. 한 줄 요약 | **이 발표가 무엇을 보였는지 미리 알려준다** |
| 1:30 | 3. 동기 (MI) | 분야 자체의 중요성 |
| 2:30 | 4. 비교 task | IOI ↔ Code 구조적 동일성 |
| 3:30 | 5. 진행 단계 | 로드맵 |
| 4:30 | 6–7. Step 2 (Grokking) | sanity check 용도임을 강조 |
| 6:00 | 8–9. Step 3 baseline + top heads | **메인 발견** |
| 7:30 | 10–11. IOI universality 비교 | **클래스별 전이 패턴** |
| 8:30 | 12. SIH 비전이 (A) | 차별화 포인트 1 |
| 9:30 | 13. Direct logit attribution (B) | **차별화 포인트 2 (기능적 부호 보존)** |
| 10:30 | 14. Minimal circuit (F) | 회로 크기 정량화 |
| 11:30 | 15–17. Cross-model + hydra | **차별화 포인트 3 (가장 새 발견)** |
| 13:00 | 18. Universality 4축 분해 | **종합 framework** |
| 13:30 | 19–20. Contribution 7개 | 정리 |
| 14:00 | 21–22. 한계 + 산출물 | 정직한 마무리 |
| 14:30 | 23. 감사 | Q&A 진입 |

---

## Slide 1 — 표지

> "안녕하세요, Mechanistic Interpretability를 주제로 텀프로젝트한 손성준입니다. Grokking과 Code Variable Binding 두 축으로 진행했습니다."

(20–30초)

---

## Slide 2 — 한 줄 요약 ⭐

> "결론부터 말씀드리면, **GPT-2 small이 자연어 IOI 회로를 코드 변수 바인딩에 재사용**합니다. 그것도 단순 재사용이 아니라 **기능적 역할의 부호(±)까지 보존**된 형태로요. 추가로 Pythia, GPT-2 medium에서도 재현해서, universality가 *해부학적*이 아닌 *기능적* 수준에서 성립함을 보였습니다."

**Tip**: 결론을 먼저 말하면 청중이 이후 슬라이드의 의미를 자연스럽게 따라옴. (60초)

---

## Slide 3 — 왜 이 주제

> "Mechanistic Interpretability는 2026년 MIT Tech Review에 선정된 분야이고, Anthropic CEO가 회사 핵심 목표로 선언했습니다. 핵심 질문은 — 트랜스포머가 코드 변수와 자연어 entity를 같은 회로로 처리하는가? 이를 Circuit Universality 가설이라 부르고, modality 간 검증은 거의 안 된 상태였습니다."

(60초)

---

## Slide 4 — 비교 task

> "두 task는 표면이 다르지만 구조가 같습니다. IOI는 'John과 Mary 중에 누가 정답인지', 코드 binding은 'x와 z 중에 누가 정답값을 가지는지'. **둘 다 두 후보 중 binding 관계로 정답을 고르는 문제**입니다. 그래서 비교가 의미 있죠."

(60초)

---

## Slide 5 — 진행 단계

> "Step 1, 2는 Nanda 2023 grokking 재현으로 해석 도구를 검증하고, Step 3에서 메인 분석을 했습니다. 제안서 외에 7개 추가 실험을 더했고, 그게 차별화 포인트가 됐습니다."

(60초)

---

## Slide 6–7 — Step 2 Grokking

> "Step 2는 sanity check입니다. 정답이 알려진 modular addition에서 patching·ablation·Fourier 도구가 옳게 동작하는지 확인. **40,000 step을 H100에서 101초**에 끝냈고, test accuracy 99.3%까지 갔습니다. GIF에서 보시면 분명한 grokking 전환이 보입니다."

**Tip**: GIF는 자동 재생되니 멈추지 말고 다음 슬라이드로. (90초)

---

## Slide 8 — Step 3 baseline + top heads ⭐

> "GPT-2 small에 500개 counterfactual pair를 넣어 per-head activation patching을 돌렸습니다. **Top 5개 중 4개가 Wang 2022의 IOI 26 head**에 해당합니다. 랜덤 기대치 대비 2.1배 enrichment."

**핵심**: 청중이 "그래서 어느 head가 어떤 IOI 클래스인지" 알아볼 수 있게 짚어주기. (90초)

---

## Slide 9 — Heatmap

> "이게 patching 결과를 IOI 클래스별로 outline한 그림입니다. 색이 진할수록 recovery가 강해요. 보시면 L10 행에 NNMH(red), BNMH(brown) 같은 출력 단계 head들이 몰려있죠."

(60초)

---

## Slide 10–11 — Universality summary

> "클래스별로 평균을 내봤습니다. **출력 단계인 NNMH +0.28, BNMH +0.07은 전이되지만, 구조적 단계인 SIH는 −0.05**로 오히려 음수입니다. 즉 단순 'IOI 회로가 그대로 옮겨졌다'가 아니라 **선택적 재사용**이 일어난 거죠."

(90초)

---

## Slide 12 — Extra A: SIH 비전이

> "SIH가 음수인 게 위치 매칭 실패 때문일 수도 있죠. 그래서 SIH 4개 head × 14개 토큰 위치를 다 패치해봤습니다. **모든 위치에서 0 또는 음수**. L8H6는 final 위치에서 −0.20까지 떨어집니다. → SIH 메커니즘이 코드 도메인에 아예 없다는 강한 증거."

(60초)

---

## Slide 13 — Extra B: Logit attribution ⭐⭐ 가장 임팩트

> "Patching은 '이 head가 중요하다'만 말합니다. Direct logit attribution을 해보니 — **L10H7는 IOI에서 NNMH로 정답을 깎는 역할인데, 코드에서도 −0.31, z=−1.9로 정답을 깎습니다**. 부호까지 그대로 보존된 거죠. 단순 활성화 재사용이 아니라 기능적 역할 자체가 전이됐다는 결정적 증거입니다."

**Tip**: 천천히, 강조해서. 이게 가장 새 발견. (90초)

---

## Slide 14 — Extra F: Minimal circuit

> "회로 크기를 정량화했습니다. **IOI 26개 head — 144개의 18% — 만 제거해도 attention 전체 기여의 87%가 사라집니다**. 거의 모든 attention 기여가 IOI sub-circuit에 집중되어 있는 거죠."

(60초)

---

## Slide 15–16 — Cross-model

> "단일 모델에 그치지 않게, **Pythia-160M (다른 아키텍처)와 GPT-2 medium (같은 family 큰 모델)**에서 재현했습니다. Pythia에선 functional pattern은 보존되지만 head 좌표는 거의 안 겹쳐요(1/10). 즉 같은 task를 같은 *원리*로 풀지만 *어디서* 푸는지는 모델마다 다릅니다."

(90초)

---

## Slide 17 — ⭐ Hydra effect

> "GPT-2 medium에서 가장 놀라운 발견 — **top-10 head를 ablate하면 logit_diff가 *오히려 증가*** 합니다. Wang 2022의 'backup name mover hydra effect'가 큰 모델에서 *과보상*까지 일어나는 거죠. 큰 모델일수록 회로 redundancy가 강해서 단순 ablation으로는 회로가 안 깨집니다."

**Tip**: 청중이 가장 놀랄 슬라이드. 충분히 멈춰서 강조. (90초)

---

## Slide 18 — Universality 4축 분해

> "이를 종합하면 universality가 4축으로 나뉩니다. 1·2번(functional concentration, causal necessity)은 모든 모델 보존, 3번(layer 위치)은 family 내에서만 보존, 4번(head 좌표)은 random 수준. **결론: universality는 architecture 표면이 아닌 functional level에서 성립합니다**."

(60초)

---

## Slide 19–20 — Contribution 7개

> "정리하면 7개 contribution이 있습니다. 1번 cross-domain 정량화, 2번 selective reuse, 3번 기능적 부호 보존, 4번 head ≠ position grammar, 5번 회로 크기, 6번 4축 분해, 7번 hydra effect at scale. 특히 **3번과 7번은 본 프로젝트가 처음 정량화한 것**입니다."

(60초)

---

## Slide 21 — 한계

> "한계도 정직하게 말씀드리면 — hydra의 *왜?* 는 path patching으로 더 봐야 하고, Tier 3 multi-hop은 분석 못했고, Llama 같은 더 큰 모델로의 일반화는 향후 과제입니다."

(30초)

---

## Slide 22 — 산출물

> "9개 git commit, 13개 분석 스크립트, 17개 PNG + 2 GIF, results 리포트, 본 슬라이드까지 모두 son 브랜치에 push했습니다. wandb run도 공개되어 있습니다."

(30초)

---

## Slide 23 — 감사

> "감사합니다. 질문 받겠습니다."

---

# 예상 Q&A

### Q1. "왜 GPT-2 small을 선택했나? 더 큰 모델은?"
> "Wang 2022의 IOI 회로 분석이 GPT-2 small에서 이루어졌기 때문에 직접 비교 가능한 reference가 있는 모델입니다. 더 큰 모델로는 GPT-2 medium과 Pythia-160M에서 재현했고, hydra effect 같은 scale 효과를 발견했습니다. Llama 등은 GPU·시간 한계로 향후 과제입니다."

### Q2. "발견된 head가 정말 binding 역할을 하는지 어떻게 확신하나?"
> "세 가지 증거를 합쳤습니다. (1) Patching에서 단일 head 패치로 logit_diff의 49%까지 복원, (2) Direct logit attribution으로 정답 logit 방향에 직접 기여(L10H2 +1.01, z=+3.3), (3) Ablation 시 logit_diff 감소. 셋이 일치해야 인과적 결론을 내릴 수 있고, 본 분석에서 일치합니다."

### Q3. "Top-K ablation이 GPT-2 medium에서 오히려 성능을 높인 건 버그 아닌가?"
> "Wang 2022 IOI 논문에서 보고된 'Backup Name Mover hydra effect'와 같은 메커니즘입니다. Top head 중 negative name mover-style suppressor가 포함되어 있어, 이를 제거하면 distractor 억제가 풀리면서 정답 logit이 상대적으로 커집니다. 모델이 크면 backup mechanism이 더 강해서 과보상까지 일어나죠. Random K ablation은 그런 효과가 없는 게 통제군 증거입니다."

### Q4. "Cross-model에서 head 좌표가 안 겹친 건 당연한 거 아닌가?"
> "맞습니다, 예상한 결과입니다. 핵심은 head 좌표는 안 겹치지만 **functional concentration·causal necessity는 보존되었다**는 점입니다. 즉 모델은 '같은 task에 같은 *원리*'를 쓰지만 '*어디서* 푸는지'는 architecture 따라 달라진다 — 이게 universality framework를 4축으로 나눠 본 본 연구의 기여입니다."

### Q5. "Selective reuse를 SIH '비전이'로 결론지었는데, 코드 task에 SIH가 필요 없는 건 아닌가?"
> "그게 정확한 해석입니다. IOI는 'duplicate token이 두 번 등장 → SIH로 첫 번째 entity 억제'라는 구조가 필요한데, 코드 `x=5; z=9; a=z; a=?`에서는 그런 duplicate 구조가 없습니다. **SIH의 존재 이유 자체가 사라진 task**. 그래서 비전이가 정확히는 'task structure 불일치'에서 옴 — 그래서 universality가 '같은 추상 구조에서만' 성립한다는 더 정교한 framing이 가능합니다."

### Q6. "Grokking은 본 분석과 어떤 관련이 있나?"
> "직접 관련은 없습니다. Sanity check 역할입니다. Modular addition은 정답 알고리즘(Fourier basis)이 알려져 있어서, 본 분석에서 사용한 patching·ablation 도구가 옳게 작동하는지 검증할 수 있습니다. 만약 grokking 재현이 실패했다면 도구 자체가 의심됐을 텐데, 잘 재현돼서 Step 3 결과를 신뢰할 수 있게 됐죠."

### Q7. "어떻게 1주일 만에 이만큼 했나?"
> "H100 1장과 효율적인 코드 구성 덕분입니다. Grokking 학습이 101초, patching이 한 모델당 15분, cross-model 추가가 70분. 코드 재사용도 큰 도움이었습니다. 모든 결과는 son 브랜치에서 재현 가능합니다."

### Q8. "결과가 실제로 어떤 의미가 있나? 실용적 함의?"
> "두 가지. (1) **코드 fine-tuning이 자연어 능력에 영향**을 줄 수 있다 — 출력 head 회로가 공유되므로. (2) **큰 모델 해석에는 단순 ablation이 부족**하다 — hydra effect 때문에 path patching 같은 정교한 도구가 필요. Anthropic 등 interpretability 연구가 큰 모델로 갈수록 이게 중요해집니다."

---

## 발표 팁

1. **첫 30초가 결정적** — Slide 2 한 줄 요약을 자신 있게 말하라
2. **GIF는 멈추지 마라** — 자동 재생되니 말로 설명만
3. **Slide 13(NNMH 부호 보존)과 17(hydra)에서 잠시 멈춰라** — 가장 임팩트 큰 두 슬라이드
4. **"7개 contribution"이라는 숫자를 반복** — 청중이 양에 압도되게
5. **Q&A에서 모르면 "그건 향후 과제입니다" 솔직히** — 한계 슬라이드(21)와 일관성 유지

## 발표 직전 체크리스트

- [ ] GIF 자동 재생 동작 확인 (Marp HTML 또는 PDF 변환 후 미리보기)
- [ ] 슬라이드 17(hydra heatmap) 색 대비 프로젝터에서 확인
- [ ] 폰트 깨짐 확인 (한글 + 영문 혼용)
- [ ] 백업: PDF + HTML 둘 다 준비
- [ ] wandb URL 접속 가능 확인
