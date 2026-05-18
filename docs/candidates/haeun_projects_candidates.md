# 딥러닝 프로젝트 계획서

---

## 19. 부동산 탐정

### 주제

> **"동네 후기를 넣으면 광고성/과장 리뷰를 걸러내고, 진짜 살아본 사람 관점의 장단점과 맞춤 매물을 추천해주는 AI"**

상품 플랫폼 리뷰(직방, 다방, 네이버 부동산 등)를 입력하면 광고성/과장성/허위 후기 의심도를 판단하고, 여러 후기에서 공통 장단점을 요약하며, 사용자 조건에 맞는 매물 추천까지 제공하는 앱.

---

### 대략적인 계획

#### 1. 데이터 준비

- 직방·다방·네이버 부동산 크롤링으로 실제 거주 후기 수집 (교통, 소음, 관리비, 주변 환경 등 태그 포함)
- Deceptive Opinion Spam Corpus 기반으로 가짜 후기 탐지 baseline 전이학습
- 공인중개사 홍보성 글 vs 실거주자 후기 이진 라벨링으로 지도학습 데이터 구성
- 국토교통부 실거래가 공공데이터 연계해 가격 이상값 탐지 feature 추가

#### 2. 모델 구성

- **Baseline**: TF-IDF + Logistic Regression으로 광고성 후기 이진 분류
- **딥러닝**: KoBERT fine-tuning으로 가짜/과장 후기 탐지 + 감성 분류 (긍정/부정/중립)
- **리뷰 요약**: KoBART로 다수 후기 → "교통 좋음, 층간소음 심함" 형태의 키워드 요약
- **추천 모델**: 사용자 조건(예산, 교통, 평수, 반려동물 등)을 입력받아 협업 필터링 또는 콘텐츠 기반 필터링으로 매물 추천
- **가짜 의심 근거 feature화**
  - 지나치게 긍정적인 단어 비율 (어휘 다양성 낮음)
  - 입주 직후 단기 계정의 첫 리뷰 여부
  - 구체적 지명·시설 언급 없이 감정 과잉 표현
  - 동일 IP·계정 패턴 (메타데이터 활용 시)
  - 실거래가 대비 가격 묘사 괴리

---

### 참조 논문

| # | 제목 | 저널/출처 | 연도 | 링크 |
|---|------|-----------|------|------|
| 1 | Fake Review Detection on Digital Platforms Using the RoBERTa Model: A Deep Learning and NLP Approach | HighTech and Innovation Journal | 2023 | [ResearchGate](https://www.researchgate.net/publication/394148115_FAKE_REVIEW_DETECTION_ON_DIGITAL_PLATFORMS_USING_THE_ROBERTA_MODEL_A_DEEP_LEARNING_AND_NLP_APPROACH) |
| 2 | Recommender Systems in Real Estate: A Systematic Review | Expert Systems with Applications, Elsevier | 2024 | [Academia.edu](https://www.academia.edu/129368379/Recommender_systems_in_real_estate_a_systematic_review) |

---
---

## 20. 쇼츠 탐정

### 주제

> **"유튜브 쇼츠를 분석하면 AI가 양산한 저품질 콘텐츠를 걸러내고, 진짜 사람이 만든 쇼츠인지 판별해주는 AI"**

쇼츠의 자막·제목·썸네일·댓글 데이터를 입력하면 AI 생성 의심도를 판단하고, 반복 패턴·클릭베이트·감정 조작 요소를 탐지해 콘텐츠 신뢰도를 점수화해주는 시스템.

---

### 대략적인 계획

#### 1. 데이터 준비

- YouTube Data API v3로 쇼츠 메타데이터 수집 (제목, 설명, 댓글, 좋아요 수, 업로드 주기, 채널 개설일 등)
- AI 생성 텍스트 탐지를 위한 공개 데이터셋 활용 — HC3(Human ChatGPT Comparison Corpus), RAID Benchmark 등
- 양산형 쇼츠 특징을 직접 라벨링 — 동일 채널의 업로드 주기 이상값, 자막 반복률, 클릭베이트 키워드 빈도 기반으로 이진 레이블 구성
- 썸네일 이미지는 AI 생성 이미지 탐지 데이터셋(CIFAKE 등) 활용해 멀티모달 feature 추가

#### 2. 모델 구성

- **Baseline**: TF-IDF + Logistic Regression으로 제목/자막 텍스트의 AI 생성 여부 이진 분류
- **딥러닝 (텍스트)**: RoBERTa 또는 KoBERT fine-tuning으로 자막·댓글의 AI 생성 텍스트 탐지 + 클릭베이트 분류
- **딥러닝 (이미지)**: ResNet 또는 EfficientNet으로 썸네일의 AI 생성 이미지 탐지
- **멀티모달 융합**: 텍스트 모델 + 이미지 모델 출력을 Late Fusion으로 결합해 최종 신뢰도 점수 산출
- **AI 생성 의심 근거 feature화**
  - 자막의 어휘 다양성 낮음 (TTR, Type-Token Ratio)
  - 제목의 클릭베이트 패턴 (숫자 어그로, 감탄사 과잉, 과장 형용사)
  - 업로드 주기 비정상적으로 짧음 (하루 5개 이상 등)
  - 채널 개설일 대비 구독자 급등 이상 패턴
  - 댓글 감성 분포가 지나치게 단조롭거나 봇 의심 패턴
  - 썸네일의 AI 생성 아티팩트 (손가락, 텍스트 왜곡 등)

---

### 참조 논문

| # | 제목 | 저널/출처 | 연도 | 링크 |
|---|------|-----------|------|------|
| 1 | AI-Generated Text Detection Using RoBERTa: A Generalizability and Explainability Analysis | arXiv | 2024 | [arXiv](https://arxiv.org/abs/2601.03812) |
| 2 | BaitRadar: A Multi-Model Clickbait Detection Algorithm Using Deep Learning | Monash University / arXiv | 2025 | [arXiv](https://arxiv.org/abs/2505.17448) |
