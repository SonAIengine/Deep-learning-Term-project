# Term Project 후보 아이디어 모음

두 개의 후보 문서(`project_candidates.md` 초안 9개 + yesulmin 생활밀착형 10개)를 하나로 통합한 버전. 19개 아이디어를 동등한 순위로 나열하며, 공통 섹션은 **주제 / 대략적인 계획 / 실제 활용 데이터셋 / 예상 결과 / 참조** 다섯 개로 통일했다. 별점/추천/순위 등 AI 평가 내용은 제외했다.

원본은 `docs/archive/project_candidates.md` 참고.

---

## 한눈에 보기

| #  | 아이디어                         | 한 줄 주제                                              |
| -- | ---------------------------- | -------------------------------------------------- |
| 1  | CommCoach AI                 | 공격적·비효율적 문장을 공손한 표현으로 변환하고 커뮤니케이션 전략 추천            |
| 2  | SLLM Tool Calling Agent      | 자연어 요청을 Tool/API 호출 JSON으로 변환하는 경량 SLLM            |
| 3  | AI 카톡 답장 클론                  | 사용자의 말투를 학습해 개인 스타일 답장 생성                          |
| 4  | 감정 기반 음악 추천 AI               | 얼굴 표정·음성으로 감정을 인식해 맞춤 음악 추천                        |
| 5  | 집중력 예측 AI                    | 얼굴·행동 패턴으로 집중력 저하 상태 예측                            |
| 6  | 기억 왜곡 탐지 AI                  | 동일 질문 답변의 일관성으로 기억 왜곡 가능성 탐지                       |
| 7  | AI 환각 탐지기                    | AI가 생성한 그럴듯한 거짓 정보 탐지                              |
| 8  | 밈 생성 AI                      | 입력 상황에 맞는 밈 이미지 생성                                 |
| 9  | 이상 행동 탐지 AI                  | 영상에서 비정상 행동을 자동으로 탐지                               |
| 10 | 단톡방 놓친 사람용 AI 비서             | 단톡 대화에서 요약/일정/담당자별 할 일 자동 추출                       |
| 11 | 택배·청첩장·부고 스미싱 방패 AI          | 문자의 스미싱 위험도와 의심 근거 판단                              |
| 12 | 영수증 먹보 탐정                    | 영수증 사진으로 소비·식습관 패턴 요약                              |
| 13 | 홈트 자세 잔소리 AI                 | 카메라로 운동 자세 분석해 자세 교정 피드백                           |
| 14 | 오늘 뭐 입지 AI 스타일리스트            | 옷장 사진과 상황 기반 코디 추천                                 |
| 15 | 시각장애인/어르신 생활사진 설명 비서         | 사진 속 물건·글자·상황을 쉬운 말로 설명                            |
| 16 | 냉장고 파먹기 AI                   | 재료로 가능한 레시피와 부족한 재료 추천                             |
| 17 | 화장품 성분 번역기                   | 성분표를 쉬운 말로 번역하고 주의 성분 분류                           |
| 18 | 이어폰 낀 사람용 주변 위험 소리 알림        | 사이렌·경적·유리 깨짐 등 위험 소리 감지 알림                         |
| 19 | 리뷰 낚시 탐정                     | 가짜·과장 리뷰 의심도 판단 + 공통 장단점 요약                        |

---

## 상세 아이디어

### 1. CommCoach AI (상황 인식 기반 커뮤니케이션 코칭 시스템)

#### 주제

공격적이거나 비효율적인 문장을 상황에 맞는 공손하고 효과적인 표현으로 변환하고, 적절한 커뮤니케이션 전략까지 추천하는 AI 시스템

#### 대략적인 계획

* Step 1: 문장 공격성/감정 분류 모델 학습
* Step 2: Transformer 기반 문장 재작성 모델 구축
* Step 3: 상황 기반 전략 분류 모델 추가
* Step 4: 채팅 이미지 입력 시 CNN + OCR 적용 (확장)
* Step 5: 전체 파이프라인 통합 및 데모 제작

#### 실제 활용 데이터셋

* Korean Toxic Comment Dataset (KOTOX)
* Korean Hate Speech Dataset
* Jigsaw Toxic Comment Dataset
* 자체 제작 데이터 (공손 변환 문장 쌍)

#### 예상 결과

* 공격성 분류 정확도: 85% 이상
* 문장 변환 자연스러움 향상 (정성 평가)
* 실제 커뮤니케이션 개선 효과 데모 가능

#### 참조

* [Korean Hate Speech Dataset (kocohub)](https://github.com/kocohub/korean-hate-speech)
* [Korean UnSmile Dataset (smilegate-ai)](https://github.com/smilegate-ai/korean_unsmile_dataset)
* [Jigsaw Toxic Comment Classification Challenge (Kaggle)](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)

---

### 2. SLLM Tool Calling Agent (자연어 → Tool/API 호출 모델)

#### 주제

사용자의 자연어 요청을 적절한 API 또는 Tool 호출(JSON 형태)로 변환하는 경량 SLLM 모델

#### 대략적인 계획

* Step 1: Tool schema 정의 (6~8개)
* Step 2: tool-calling 데이터셋 정리
* Step 3: SLLM (Qwen, Llama 등) QLoRA fine-tuning
* Step 4: JSON parsing + tool executor 구현
* Step 5: UI 기반 실제 실행 데모

#### 실제 활용 데이터셋

* xLAM Function Calling Dataset
* ToolBench
* Berkeley Function Calling Leaderboard Dataset
* delivery-app-function-calling-datasets-korean

#### 예상 결과

* Tool selection accuracy: 80% 이상
* JSON valid output 비율 증가
* 실제 API 호출 성공률 개선

#### 참조

* [xLAM Function Calling 60k (Salesforce, Hugging Face)](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)
* [ToolBench (OpenBMB)](https://github.com/OpenBMB/ToolBench)
* [Berkeley Function Calling Leaderboard (Gorilla)](https://gorilla.cs.berkeley.edu/leaderboard.html)
* [Qwen 모델 (Hugging Face)](https://huggingface.co/Qwen)
* [Llama 모델 (Hugging Face)](https://huggingface.co/meta-llama)

---

### 3. AI 카톡 답장 클론 (개인화 대화 생성 모델)

#### 주제

사용자의 말투를 학습하여 자동으로 개인 스타일의 답장을 생성하는 AI

#### 대략적인 계획

* Step 1: 대화 데이터 전처리 (입력-답장 구조)
* Step 2: Transformer 기반 생성 모델 학습
* Step 3: 개인 데이터로 fine-tuning
* Step 4: 실시간 채팅 데모 제작

#### 실제 활용 데이터셋

* AI Hub 한국어 대화 데이터셋
* DailyDialog Dataset
* 개인 카톡 데이터 (익명화)

#### 예상 결과

* 자연스러운 답장 생성
* 개인 말투 유사도 향상
* 높은 데모 임팩트

#### 참조

* [AI Hub 한국어 대화 데이터](https://aihub.or.kr)
* [DailyDialog Dataset](http://yanran.li/dailydialog)

---

### 4. 감정 기반 음악 추천 AI

#### 주제

사용자의 얼굴 표정 또는 음성을 분석하여 감정 상태에 맞는 음악을 추천하는 시스템

#### 대략적인 계획

* Step 1: 얼굴 감정 인식 CNN 학습
* Step 2: 감정 → 음악 매핑 로직 설계
* Step 3: 실시간 카메라 입력 처리
* Step 4: 추천 결과 UI 구현

#### 실제 활용 데이터셋

* FER2013
* RAVDESS

#### 예상 결과

* 감정 분류 정확도 70~80%
* 실시간 추천 데모 가능
* 사용자 체감 만족도 높음

#### 참조

* [FER2013 (Kaggle)](https://www.kaggle.com/datasets/msambare/fer2013)
* [RAVDESS (Zenodo)](https://zenodo.org/record/1188976)

---

### 5. 집중력 예측 AI

#### 주제

사용자의 얼굴 및 행동 패턴을 기반으로 집중력 저하 상태를 예측하는 모델

#### 대략적인 계획

* Step 1: 영상 데이터 수집
* Step 2: CNN + RNN 모델 학습
* Step 3: 집중도 scoring 시스템 설계
* Step 4: 실시간 예측 데모

#### 실제 활용 데이터셋

* DAiSEE Dataset

#### 예상 결과

* 집중 상태 분류 가능
* 실시간 피드백 시스템 구현
* 교육/업무 활용 가능성

#### 참조

* [DAiSEE Dataset (IIT Hyderabad)](https://people.iith.ac.in/vineethnb/resources/daisee/index.html)

---

### 6. 기억 왜곡 탐지 AI

#### 주제

동일 질문에 대한 답변의 일관성을 분석하여 기억 왜곡 가능성을 탐지

#### 대략적인 계획

* Step 1: 인터뷰 데이터 직접 수집
* Step 2: 텍스트 유사도 분석
* Step 3: 시간 흐름 기반 RNN 모델 적용
* Step 4: consistency score 계산

#### 실제 활용 데이터셋

* Persona-Chat Dataset
* 자체 인터뷰 데이터

#### 예상 결과

* 답변 일관성 수치화
* 새로운 유형의 분석 결과 제시
* 연구형 프로젝트로 적합

#### 참조

* [Persona-Chat (ParlAI)](https://parl.ai/projects/personachat/)
* [Persona-Chat (Hugging Face mirror)](https://huggingface.co/datasets/bavard/personachat_truecased)

---

### 7. AI 환각 탐지기

#### 주제

AI가 생성한 "그럴듯한 거짓 정보"를 탐지하는 시스템

#### 대략적인 계획

* Step 1: 사실/허구 데이터 구축
* Step 2: Transformer 기반 분류 모델 학습
* Step 3: hallucination score 정의
* Step 4: 모델 비교 실험

#### 실제 활용 데이터셋

* FEVER Dataset
* TruthfulQA

#### 예상 결과

* hallucination detection 정확도 향상
* 최신 AI 문제 해결 접근
* 연구 가치 높음

#### 참조

* [FEVER (Fact Extraction and VERification)](https://fever.ai)
* [TruthfulQA (GitHub)](https://github.com/sylinrl/TruthfulQA)

---

### 8. 밈 생성 AI

#### 주제

사용자가 입력한 상황을 기반으로 적절한 밈 이미지를 생성하는 시스템

#### 대략적인 계획

* Step 1: 밈 템플릿 수집
* Step 2: 상황 문장 생성
* Step 3: 텍스트 → 이미지 매핑
* Step 4: UI 기반 출력

#### 실제 활용 데이터셋

* Meme Dataset (Facebook AI)
* 직접 수집 데이터

#### 예상 결과

* 재미 중심 결과
* 높은 발표 임팩트
* 사용자 참여 유도

#### 참조

* [Hateful Memes Dataset (Meta AI)](https://ai.meta.com/datasets/hateful-memes/)

---

### 9. 이상 행동 탐지 AI

#### 주제

영상에서 비정상적인 행동을 자동으로 탐지하는 시스템

#### 대략적인 계획

* Step 1: 영상 데이터 수집
* Step 2: CNN + LSTM 모델 학습
* Step 3: anomaly detection 모델 적용
* Step 4: 영상 데모 제작

#### 실제 활용 데이터셋

* UCF Crime Dataset

#### 예상 결과

* 이상 행동 탐지 가능
* 보안/감시 시스템 응용

#### 참조

* [UCF-Crime Dataset (CRCV, University of Central Florida)](https://www.crcv.ucf.edu/projects/real-world/)

---

### 10. 단톡방 놓친 사람용 AI 비서

#### 주제

> **"읽씹 방지 AI: 단톡방 300개 메시지에서 내 할 일만 뽑아주는 비서"**

카톡/디스코드/팀플 단톡 내용을 입력하면 **핵심 요약, 결정사항, 일정, 담당자별 할 일**을 자동으로 정리해주는 앱.

#### 대략적인 계획

1. **데이터 준비**
   * AI-Hub 한국어 대화 요약 또는 SAMSum 데이터 사용.
   * 대화를 `입력 대화`, `요약문`, `화자`, `시간`, `할 일 표현` 중심으로 전처리.
   * 한국어 데이터가 부족하면 직접 팀플 대화 예시 100~200개 정도를 만들어 테스트셋으로 사용.

2. **모델 구성**
   * Baseline: KoBERT/KLUE-BERT로 문장 중요도 분류.
   * 발전형: KoBART 또는 T5 계열 모델로 대화 요약 fine-tuning.
   * 추가 기능: 일정/마감/담당자 추출은 token classification 또는 rule-based 후처리와 결합.

3. **앱 구현**
   * Streamlit/Gradio로 대화 텍스트 입력창 구현.
   * 출력 탭을 `요약`, `내 할 일`, `중요 일정`, `담당자별 업무`로 분리.
   * "내 이름"을 입력하면 나에게 해당하는 할 일만 필터링.

4. **강의 개념 연결**
   * RNN/LSTM baseline과 Transformer 모델 비교.
   * Attention이 긴 대화에서 중요한 문장에 집중한다는 식으로 설명 가능.
   * Semi-supervised 확장: 라벨 없는 단톡 예시를 모델이 pseudo-summary 생성 후 재학습.

#### 실제 활용 데이터셋

* AI-Hub 한국어 대화 요약 (대화 원문/요약문 35만 건 규모)
* SAMSum (약 16k개 메신저형 대화 요약)
* 자체 제작 팀플 대화 예시 (테스트셋용)

#### 예상 결과

```text
전체 요약:
- 팀은 5/24까지 데이터 전처리를 끝내기로 했다.
- 모델 학습은 지현, 발표자료는 민수, 데모는 내가 담당한다.
- 최종 리허설은 5/30 저녁 8시에 진행한다.

내 할 일:
1. 5/25까지 Streamlit 데모 완성
2. 5/29까지 영상 촬영용 예시 대화 준비

중요 일정:
- 5/24 데이터 전처리 마감
- 5/30 20:00 최종 리허설
```

발표 영상에서는 일부러 긴 단톡 내용을 붙여넣고, AI가 "결정사항/할 일/일정"을 한 번에 정리하는 장면을 보여주면 된다.

#### 참조

* [AI-Hub 한국어 대화 요약](https://aihub.or.kr/aidata/30714)
* [SAMSum Corpus (Hugging Face)](https://huggingface.co/datasets/samsum)
* [KoBART (SKT)](https://github.com/SKT-AI/KoBART)
* [KLUE 벤치마크](https://klue-benchmark.com)

---

### 11. 택배·청첩장·부고 스미싱 방패 AI

#### 주제

> **"문자 링크 누르기 전에 AI가 사기 냄새를 잡아주는 앱"**

문자 내용을 입력하면 **정상/스팸/스미싱 위험도**를 판단하고, 어떤 표현이 위험한지 하이라이트해주는 개인 보안 앱.

#### 대략적인 계획

1. **데이터 준비**
   * UCI SMS Spam Collection으로 영어 baseline 모델 학습.
   * 한국어 스미싱 예시는 직접 수집/제작해서 소량 라벨링.
   * 예시 유형: 택배 주소 오류, 모바일 청첩장, 부고장, 계좌 정지, 카드 결제 오류, 정부지원금 사칭.

2. **모델 구성**
   * Baseline: TF-IDF + Logistic Regression.
   * 딥러닝: LSTM 또는 CNN text classifier.
   * 발전형: KoBERT/KLUE-RoBERTa로 한국어 문자 분류.
   * 위험 근거 표시: attention score, SHAP/LIME, 또는 키워드 기반 하이라이트.

3. **앱 구현**
   * 문자 입력창.
   * 위험도 점수 표시: 0~100%.
   * 의심 근거 표시: 링크 포함, 개인정보 요구, 긴급성 표현, 공식 도메인 불일치 등.
   * "권장 행동" 출력: 링크 클릭 금지, 공식 앱 확인, 고객센터 직접 조회.

4. **강의 개념 연결**
   * Binary classification.
   * Word embedding, RNN/LSTM, Transformer.
   * Semi-supervised: unlabeled 한국어 문자에 pseudo-label 붙여 데이터 확장.

#### 실제 활용 데이터셋

* UCI SMS Spam Collection (공개 SMS 라벨 데이터셋)
* 자체 수집/제작 한국어 스미싱 예시 (소량 라벨링)

#### 예상 결과

```text
위험도: 92%

분류: 스미싱 의심

의심 근거:
- '주소 오류로 배송 보류'라는 긴급성 표현
- 단축 URL 포함
- 개인정보 입력 유도
- 공식 택배사 도메인 아님

권장 행동:
- 링크 클릭 금지
- 택배사 공식 앱에서 송장번호 직접 조회
```

발표 영상에서는 정상 택배 문자와 가짜 택배 문자를 비교해서 넣고, AI가 위험 근거를 색칠해주는 장면을 보여주면 효과적이다.

#### 참조

* [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
* [SHAP (Explainable AI)](https://github.com/shap/shap)
* [LIME (Local Interpretable Model-agnostic Explanations)](https://github.com/marcotcr/lime)

---

### 12. 영수증 먹보 탐정

#### 주제

> **"영수증 사진 한 장으로 소비 습관과 식습관을 요약해주는 AI"**

편의점/마트 영수증 사진을 찍으면 품목, 가격, 카테고리를 추출하고 **식비, 카페인, 간식 비율, 소비 패턴**을 분석해주는 앱.

#### 대략적인 계획

1. **데이터 준비**
   * SROIE로 영수증 OCR/정보 추출 구조 이해.
   * 실제 한국 편의점 영수증은 팀원들이 직접 30~50장 정도 수집해서 데모용으로 사용.
   * 품목명을 `음료`, `간식`, `즉석식품`, `식사`, `생활용품` 등으로 분류.

2. **모델 구성**
   * OCR: EasyOCR, Tesseract, 또는 pretrained OCR 사용.
   * 품목 분류: KoBERT 또는 간단한 text classifier.
   * 소비 패턴 분석: 규칙 기반 + 분류 모델 결합.
   * 발전형: 영수증 layout 정보를 반영하는 document understanding 방식 설명 가능.

3. **앱 구현**
   * 영수증 이미지 업로드.
   * OCR 결과 표시.
   * 총액, 품목별 카테고리, 간식/음료 비율 시각화.
   * "이번 주 소비 코멘트" 생성.

4. **강의 개념 연결**
   * CNN 기반 OCR.
   * Transformer 기반 문서 이해.
   * Text classification.
   * Multimodal: 이미지 + 텍스트 + 표 데이터 결합.

#### 실제 활용 데이터셋

* SROIE (1,000개 스캔 영수증 이미지 + OCR/정보추출 annotation)
* Open Food Facts (식품 ingredients/allergens/nutrition facts DB)
* 자체 수집 한국 편의점 영수증 30~50장 (데모용)

#### 예상 결과

```text
총 결제 금액: 18,700원

품목 분석:
- 음료: 2개 / 5,800원
- 간식: 3개 / 7,200원
- 즉석식품: 1개 / 4,900원
- 기타: 1개 / 800원

AI 코멘트:
이번 영수증은 간식과 음료 비율이 높습니다.
최근 3회 구매 중 카페인 음료가 반복적으로 등장했습니다.
```

발표 영상에서는 영수증 사진을 업로드하고, 앱이 OCR → 품목 분류 → 소비 패턴 코멘트까지 자동으로 만드는 흐름을 보여주면 된다. OCR이 틀릴 수 있으므로 발표용 샘플 영수증은 미리 잘 인식되는 것으로 준비하는 게 좋다.

#### 참조

* [ICDAR 2019 SROIE](https://github.com/zzzDavid/ICDAR-2019-SROIE)
* [Open Food Facts](https://world.openfoodfacts.org)
* [EasyOCR](https://github.com/JaidedAI/EasyOCR)
* [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
* [CORD (Receipt OCR/Parsing)](https://github.com/clovaai/cord)

---

### 13. 홈트 자세 잔소리 AI

#### 주제

> **"스쿼트하다가 자세가 무너지면 바로 잔소리하는 AI 홈트 코치"**

카메라로 운동 자세를 보고 **무릎 각도, 허리 기울기, 좌우 균형**을 분석해 잘못된 자세를 알려주는 앱.

#### 대략적인 계획

1. **데이터 준비**
   * Fit3D 또는 MM-Fit 데이터로 운동 자세 데이터 구조 확인.
   * 시간 부족 시 직접 스쿼트/런지 영상 30~50개 촬영.
   * `정상 자세`, `무릎 안쪽`, `허리 굽음`, `상체 과도한 숙임` 등 라벨링.

2. **모델 구성**
   * MediaPipe/OpenPose로 skeleton keypoint 추출.
   * Baseline: 관절 각도 rule-based 판정.
   * 딥러닝: keypoint sequence를 LSTM/GRU/Transformer에 넣어 자세 분류.
   * 시각화: 관절선 overlay, 위험 각도 빨간색 표시.

3. **앱 구현**
   * 영상 업로드 또는 웹캠 입력.
   * skeleton 표시.
   * 자세 점수 0~100점 출력.
   * 잘못된 부위별 피드백 출력.

4. **강의 개념 연결**
   * CNN 기반 pose estimation.
   * Sequence modeling.
   * LSTM vs Transformer 비교.
   * 시계열 데이터 classification.

#### 실제 활용 데이터셋

* Fit3D (37개 반복 운동 + 3D human shape/motion capture ground truth)
* MM-Fit (스마트폰·워치·이어버드 inertial sensor + RGB-D video + 2D/3D pose estimates)
* 자체 촬영 스쿼트/런지 영상 30~50개

#### 예상 결과

```text
운동: 스쿼트
자세 점수: 72점

피드백:
- 내려갈 때 오른쪽 무릎이 안쪽으로 들어갑니다.
- 허리가 약간 굽었습니다.
- 발 간격은 적절합니다.

개선 제안:
- 무릎 방향을 발끝 방향과 맞추세요.
- 상체를 조금 더 세우세요.
```

발표 영상은 팀원이 일부러 잘못된 자세를 하고, 화면에 skeleton과 피드백이 뜨면 바로 이해된다. 데이터 전처리와 웹캠 데모 안정화가 생각보다 오래 걸릴 수 있다는 점은 주의.

#### 참조

* [Fit3D Dataset](https://fit3d.imar.ro)
* [MM-Fit Dataset](https://mmfit.github.io)
* [MediaPipe Pose (Google)](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker)
* [OpenPose (CMU)](https://github.com/CMU-Perceptual-Computing-Lab/openpose)

---

### 14. 오늘 뭐 입지 AI 스타일리스트

#### 주제

> **"내 옷장 사진과 오늘 날씨를 보고 코디를 추천해주는 AI"**

옷 사진을 업로드하면 옷 종류와 스타일을 인식하고, 사용자가 선택한 상황에 맞춰 코디 조합을 추천해주는 앱.

#### 대략적인 계획

1. **데이터 준비**
   * DeepFashion으로 옷 종류/속성 분류 학습.
   * Polyvore로 상의-하의-신발 조합 호환성 학습.
   * 팀원 옷 사진을 직접 찍어 데모용 "미니 옷장" 구성.

2. **모델 구성**
   * 옷 분류: ResNet/EfficientNet/ViT transfer learning.
   * 속성 분류: 색상, 소매 길이, formal/casual 등 multi-label classification.
   * 코디 추천: CLIP embedding 또는 item compatibility score.
   * 날씨/상황 조건은 rule-based로 결합.

3. **앱 구현**
   * 옷장 사진 업로드.
   * 상황 선택: 발표, 데이트, 학교, 비 오는 날, 추운 날.
   * 추천 코디 3개 출력.
   * 왜 추천했는지 설명: 색상 조화, 계절성, 격식도.

4. **강의 개념 연결**
   * CNN/ViT.
   * Transfer learning.
   * Image embedding.
   * Recommendation model.
   * Multimodal learning.

#### 실제 활용 데이터셋

* DeepFashion (50개 clothing category, 1,000개 attribute, bounding box, clothing landmarks)
* Polyvore (패션 호환성/outfit recommendation 데이터셋)
* 자체 촬영 팀원 옷 사진 (데모용 미니 옷장)

#### 예상 결과

```text
오늘 추천 코디: 발표용 세미포멀

상의: 흰색 셔츠
하의: 검정 슬랙스
아우터: 네이비 자켓
신발: 검정 로퍼

추천 이유:
- 흰색/검정/네이비 조합은 발표 상황에 적합합니다.
- 전체 코디의 formal score가 높습니다.
- 오늘 날씨 기준으로 아우터가 필요합니다.
```

발표 영상에서는 팀원 옷 사진을 여러 장 등록하고, "발표용 코디 추천", "비 오는 날 코디 추천"을 눌러 다른 결과가 나오는 장면을 보여주면 좋다. 성능보다 서비스 기획력이 더 중요해지는 주제다.

#### 참조

* [DeepFashion (MMLab, CUHK)](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html)
* [DeepFashion2](https://github.com/switchablenorms/DeepFashion2)
* [Polyvore Dataset](https://github.com/xthan/polyvore-dataset)
* [CLIP (OpenAI)](https://github.com/openai/CLIP)

---

### 15. 시각장애인/어르신 생활사진 설명 비서

#### 주제

> **"사진 속 물건, 글자, 상황을 쉬운 말로 설명해주는 생활 보조 AI"**

사진을 올리고 "이게 뭐야?", "유통기한 보여?", "약봉투에 뭐라고 적혀 있어?"라고 물으면 AI가 답변해주는 앱.

#### 대략적인 계획

1. **데이터 준비**
   * VizWiz VQA 또는 VizWiz-Captions 사용.
   * 발표용으로 약봉투, 음료병, 안내문, 엘리베이터 버튼, 식품 포장지 사진 직접 촬영.
   * OCR이 필요한 이미지는 별도 라벨링.

2. **모델 구성**
   * 이미지 captioning baseline: CNN encoder + LSTM/Transformer decoder.
   * 발전형: BLIP/ViT-BERT 계열 pretrained 모델 fine-tuning.
   * OCR: 이미지 속 텍스트는 EasyOCR 또는 PaddleOCR로 추출.
   * 답변 생성: 이미지 설명 + OCR 결과 + 질문을 결합.

3. **앱 구현**
   * 이미지 업로드.
   * 질문 입력.
   * 쉬운 문장으로 답변.
   * 확신이 낮을 때는 "정확하지 않을 수 있음" 표시.

4. **강의 개념 연결**
   * CNN/ViT.
   * Image captioning.
   * Transformer decoder.
   * Visual Question Answering.
   * Multimodal learning.

#### 실제 활용 데이터셋

* VizWiz VQA (시각장애인이 찍은 이미지/질문 기반 VQA)
* VizWiz-Captions (39,181개 이미지에 각 5개 caption)
* 자체 촬영 약봉투/음료병/안내문/엘리베이터 버튼/식품 포장지

#### 예상 결과

```text
질문: 이 약봉투에 뭐라고 적혀 있어?

AI 답변:
사진에는 약봉투와 흰색 안내문이 보입니다.
가장 크게 보이는 글자는 '아침 식후 30분'입니다.
작은 글자는 흐려서 정확히 읽기 어렵습니다.

주의:
약 복용과 관련된 최종 판단은 약사나 병원 안내를 확인하세요.
```

발표 영상에서는 사용자가 사진을 올리고 질문을 바꾸면 답변도 바뀌는 모습을 보여주면 좋다. 모델을 처음부터 만들기보다는 pretrained 모델 활용이 현실적이다.

#### 참조

* [VizWiz VQA](https://vizwiz.org/tasks-and-datasets/vqa/)
* [VizWiz-Captions](https://vizwiz.org/tasks-and-datasets/image-captioning/)
* [BLIP (Salesforce)](https://github.com/salesforce/BLIP)
* [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)

---

### 16. 냉장고 파먹기 AI

#### 주제

> **"냉장고에 있는 재료 사진이나 목록을 넣으면 오늘 만들 수 있는 메뉴를 추천하는 AI"**

재료 사진 또는 텍스트 목록을 입력하면 가능한 레시피, 부족한 재료, 유통기한 임박 재료 활용법을 알려주는 앱.

#### 대략적인 계획

1. **데이터 준비**
   * Food-101으로 음식 이미지 분류 baseline 학습.
   * Recipe1M+ 또는 RecipeNLG 계열 데이터로 재료-레시피 매칭 구조 사용.
   * 데모용 재료 목록은 직접 구성: 계란, 두부, 김치, 양파, 밥, 참치 등.

2. **모델 구성**
   * 이미지 입력형: 음식/재료 이미지 분류.
   * 텍스트 입력형: 재료 목록 embedding 후 유사 레시피 검색.
   * 추천 모델: ingredient overlap score + text embedding similarity.
   * 발전형: Transformer로 레시피 제목/요약 생성.

3. **앱 구현**
   * 재료 사진 업로드 또는 재료 텍스트 입력.
   * 가능한 메뉴 TOP 3 추천.
   * 부족한 재료 표시.
   * "10분 안에 가능", "자취생용", "전자레인지 가능" 필터 추가.

4. **강의 개념 연결**
   * CNN/ViT 이미지 분류.
   * Text embedding.
   * Retrieval model.
   * Multimodal learning.
   * Transformer 기반 recipe generation.

#### 실제 활용 데이터셋

* Food-101 (101개 음식 category + 101,000장 이미지)
* Recipe1M+ (100만 개 이상 레시피 + 1,300만 장 음식 이미지)
* 자체 구성 데모용 재료 목록

#### 예상 결과

```text
입력 재료:
계란, 김치, 밥, 양파, 참치

추천 메뉴 1: 김치참치볶음밥
- 사용 가능 재료: 김치, 밥, 양파, 참치
- 추가하면 좋은 재료: 대파, 참기름
- 예상 조리 시간: 12분

추천 메뉴 2: 계란김치덮밥
- 사용 가능 재료: 계란, 김치, 밥
- 난이도: 쉬움
```

발표 영상에서는 냉장고 사진 또는 재료 목록을 넣고, AI가 레시피를 추천하는 장면을 보여주면 된다. 이미지로 개별 재료를 정확히 인식하는 건 어렵기 때문에 **텍스트 입력 + 이미지 보조** 형태가 가장 현실적이다.

#### 참조

* [Food-101 (ETH Zurich)](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)
* [Recipe1M+ (MIT CSAIL)](http://pic2recipe.csail.mit.edu/)
* [RecipeNLG](https://recipenlg.cs.put.poznan.pl/)

---

### 17. 화장품 성분 번역기

#### 주제

> **"어려운 화장품 성분표를 쉬운 말로 번역해주는 AI"**

화장품 성분표 사진이나 텍스트를 입력하면 성분명을 쉽게 설명하고, 민감성 피부 주의 성분, 향료, 알코올, 보습 성분 등을 분류해주는 앱.

#### 대략적인 계획

1. **데이터 준비**
   * Open Beauty Facts에서 화장품 제품명, 성분표 데이터 사용.
   * CosIng에서 INCI 성분명과 기능 정보를 참조.
   * 발표용으로 토너, 선크림, 로션 등 실제 제품 성분표 촬영.

2. **모델 구성**
   * OCR: 성분표 이미지에서 텍스트 추출.
   * NER/token classification: 성분명 단위 분리.
   * 성분 설명: DB lookup + Transformer 기반 쉬운 설명 생성.
   * 사용자 피부 타입별 필터: 민감성, 건성, 지성 등은 rule-based로 처리.

3. **앱 구현**
   * 성분표 이미지 업로드.
   * 성분 자동 분리.
   * 성분별 쉬운 설명.
   * "주의 필요", "보습 관련", "향료 관련", "자외선 차단 관련" 태그 표시.

4. **강의 개념 연결**
   * OCR.
   * Named Entity Recognition.
   * Transformer text classification.
   * Retrieval-augmented explanation.
   * Multimodal: 이미지 OCR + 텍스트 분석.

#### 실제 활용 데이터셋

* Open Beauty Facts (화장품 ingredients/allergens 등 제품 라벨 정보 DB)
* CosIng (European Commission 화장품 성분/물질 정보 데이터베이스)
* 자체 촬영 토너/선크림/로션 등 성분표

#### 예상 결과

```text
분석 제품: 선크림 성분표

주요 성분:
- 정제수: 용매 역할
- 글리세린: 보습 성분
- 나이아신아마이드: 피부 톤 관련 기능성 성분
- 향료: 민감성 피부는 주의 가능

AI 요약:
이 제품은 보습 성분과 기능성 성분이 포함되어 있습니다.
다만 향료가 포함되어 있어 민감성 피부 사용자는 패치 테스트를 권장합니다.
```

발표 영상에서는 화장품 뒷면 성분표를 찍어서 업로드하고, 어려운 성분명이 쉬운 설명으로 바뀌는 장면을 보여주면 된다.

#### 참조

* [Open Beauty Facts (GitHub)](https://github.com/openfoodfacts/openbeautyfacts)
* [Open Beauty Facts 웹](https://world.openbeautyfacts.org)
* [CosIng (European Commission)](https://ec.europa.eu/growth/tools-databases/cosing/)

---

### 18. 이어폰 낀 사람용 주변 위험 소리 알림

#### 주제

> **"이어폰 끼고 걸어도 사이렌, 경적, 유리 깨짐 같은 위험 소리를 알려주는 AI"**

마이크로 주변 소리를 듣고 사이렌, 경적, 개 짖음, 공사 소리, 비명, 유리 깨짐 등을 감지해 알림을 띄우는 앱.

#### 대략적인 계획

1. **데이터 준비**
   * ESC-50, UrbanSound8K, FSD50K 중 하나 선택.
   * 위험 소리 class만 추려서 재구성: siren, car horn, dog bark, glass breaking, drilling 등.
   * 일반 소리와 위험 소리 binary classification도 가능.

2. **모델 구성**
   * 오디오를 mel-spectrogram으로 변환.
   * Baseline: CNN spectrogram classifier.
   * 발전형: CRNN 또는 Audio Spectrogram Transformer 구조 설명.
   * 실시간 데모는 짧은 clip 단위로 예측.

3. **앱 구현**
   * 오디오 파일 업로드 또는 마이크 입력.
   * 감지된 소리 class 표시.
   * 위험도 표시.
   * "주의: 뒤쪽에서 경적 소리 감지" 같은 알림 문구 출력.

4. **강의 개념 연결**
   * CNN.
   * Spectrogram image classification.
   * RNN/CRNN.
   * Transformer.
   * Data augmentation: noise 추가, pitch shift, time stretch.

#### 실제 활용 데이터셋

* ESC-50 (2,000개 환경음 + 50개 class)
* UrbanSound8K (8,732개 도시 소리 excerpt + 10개 class)
* FSD50K (51,197개 audio clip + 200개 sound class)

#### 예상 결과

```text
감지된 소리: Siren
위험도: 높음
확률: 88%

알림:
근처에서 사이렌 소리가 감지되었습니다.
이어폰 볼륨을 낮추고 주변을 확인하세요.
```

발표 영상에서는 이어폰을 낀 팀원이 걷는 상황을 연출하고, 노트북에서 사이렌/경적 소리를 재생했을 때 앱이 알림을 띄우는 식으로 만들면 된다.

#### 참조

* [ESC-50](https://github.com/karolpiczak/ESC-50)
* [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html)
* [FSD50K (Zenodo)](https://zenodo.org/record/4060432)
* [Audio Spectrogram Transformer (AST)](https://github.com/YuanGongND/ast)

---

### 19. 리뷰 낚시 탐정

#### 주제

> **"쇼핑 전에 리뷰를 넣으면 가짜/과장 리뷰를 의심하고 진짜 장단점만 요약해주는 AI"**

상품 리뷰를 입력하면 광고성/과장성/가짜 리뷰 의심도를 판단하고, 여러 리뷰에서 공통 장단점을 요약해주는 앱.

#### 대략적인 계획

1. **데이터 준비**
   * Amazon Reviews 2023으로 리뷰 요약/평점 예측/감성분석 학습.
   * Deceptive Opinion Spam Corpus로 가짜 리뷰 탐지 baseline 학습.
   * 한국어 쇼핑 리뷰 예시는 직접 수집하거나 번역/생성 예시로 테스트셋 구성.

2. **모델 구성**
   * Baseline: TF-IDF + SVM/Logistic Regression.
   * 딥러닝: BiLSTM 또는 Transformer text classifier.
   * 리뷰 요약: KoBART/T5 요약 모델 활용.
   * 가짜 의심 근거: 지나치게 일반적인 표현, 반복 표현, 감정 과잉, 구체성 부족 등 feature화.

3. **앱 구현**
   * 리뷰 여러 개 붙여넣기.
   * 리뷰별 신뢰도 점수 출력.
   * 공통 장점/단점 요약.
   * "낚시 의심 표현" 하이라이트.

4. **강의 개념 연결**
   * Text classification.
   * Sentiment analysis.
   * Transformer.
   * Summarization.
   * Explainable AI.

#### 실제 활용 데이터셋

* Amazon Reviews 2023 (리뷰 텍스트/평점/helpfulness vote/상품 metadata/user-item link)
* Deceptive Opinion Spam Corpus (진실/기만 호텔 리뷰)
* 자체 수집 한국어 쇼핑 리뷰 예시

#### 예상 결과

```text
상품 리뷰 분석 결과

전체 평가:
- 긍정 리뷰가 많지만, 일부 리뷰는 광고성 표현이 강합니다.

공통 장점:
1. 배송이 빠르다
2. 디자인이 깔끔하다
3. 가격 대비 성능이 좋다

공통 단점:
1. 내구성에 대한 불만이 반복됨
2. 실제 색상이 사진과 다르다는 의견 있음

가짜/과장 의심 리뷰:
- 리뷰 #3: 구체적 사용 경험 없이 극찬 표현 반복
- 리뷰 #7: 다른 리뷰와 문장 패턴이 매우 유사
```

발표 영상에서는 같은 상품에 대한 리뷰 10개를 넣고, AI가 "진짜 도움 되는 장단점"과 "의심 리뷰"를 분리하는 장면을 보여주면 된다.

#### 참조

* [Amazon Reviews 2023 (McAuley Lab, Hugging Face)](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)
* [Deceptive Opinion Spam Corpus (Myle Ott)](https://myleott.com/op-spam.html)
