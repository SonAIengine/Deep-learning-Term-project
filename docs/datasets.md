# 데이터셋 명세 v2 — Transformer Circuit Analysis

> **프로젝트**: Transformer Circuit Analysis — Grokking과 Code Variable Binding
> **관련 제안서**: [sonsj-proposal.md](candidates/sonsj-proposal.md)
> **버전**: v2 (2026-05-23)
> **v1 → v2 주요 변경**: counterfactual pair 도입, Tier 1/2/3 구조, 토큰화 사전 검증 단계 추가, 정답 토큰 제약, 1-hop 정책 명시, 저장소 구조 중앙집중화

---

## 0. 변경 이력

| 버전 | 일자 | 주요 변경 |
|---|---|---|
| v1 | 2026-05-23 | 초안. 3개 데이터셋 기본 명세 |
| v2 | 2026-05-23 | **Activation patching이 동작하도록 통제 강화**: counterfactual pair (Tier 1), single-token 변수명 풀 사전 검증, Tier 구조, 정답 1~9 제약 |

---

## 1. 데이터셋 전체 개요

| # | 데이터셋 | 사용 단계 | 규모 | 출처 |
|---|---|---|---|---|
| 1 | Modular Arithmetic | Step 1~2 | 12,769 pair (train 3,800 / test 8,900) | 직접 생성 |
| 2 | Code Variable Binding (Tier 1) | Step 3-2, 3-3 핵심 | 1,000 (500 cf pair) | 직접 생성 |
| 3 | Code Variable Binding (Tier 2) | Step 3 scaling | 500 | 직접 생성 |
| 4 | Code Variable Binding (Tier 3, optional) | Step 3 robustness | 500 | 직접 생성 |
| 5 | IOI (Indirect Object Identification) | Step 3-3 비교 | 500 | `Easy-Transformer` |
| - | GPT-2 small (모델) | Step 3 | 124M params | HuggingFace / TransformerLens |

**총 데이터 규모**: 약 15,000 sample. 데이터 양 자체는 작으며, 학습·분석 시간이 진짜 병목.

---

## 2. 설계 원칙

본 명세가 v1과 가장 크게 달라진 점은 **회로 분석(activation patching·ablation)의 기술적 요구사항**을 데이터 단계에서 미리 반영했다는 것.

### 2.1 왜 Counterfactual Pair인가
Activation patching은 "clean run의 특정 head activation을 corrupted run에 이식했을 때 logit이 얼마나 회복되는가"로 head 역할을 식별한다. 이때 clean/corrupted가 **딱 한 요소만 다른 minimal pair**여야 신호가 깨끗하게 나온다. Wang et al. (2022) IOI가 ABBA/BABA 패턴을 통제한 것과 같은 발상.

### 2.2 왜 Single-Token 변수명인가
Clean·corrupted 두 prompt의 **토큰 position이 1:1로 정렬**되어야 patching이 의미 있다. 변수명이 토크나이저에서 서로 다른 토큰 수로 쪼개지면 position이 어긋나서 분석 자체가 불가능.

### 2.3 왜 정답 값을 1~9로 제한하는가
GPT-2 BPE에서 `" 0"` ~ `" 9"`는 single token이지만 두자리 수는 multi-token으로 쪼개진다. 정답 logit을 단일 토큰에서 직접 읽으려면 single-token이어야 한다.

### 2.4 왜 Tier로 나누는가
- Patching용 데이터는 빡세게 통제 (Tier 1)
- Scaling 검증용은 변수 개수만 다양화 (Tier 2)
- Robustness는 일반 코드처럼 자유롭게 (Tier 3, optional)

용도별 통제 수준이 다르므로 한 덩어리로 묶지 않는다.

### 2.5 왜 1-hop이 정식인가
본 프로젝트 핵심 비교 대상인 IOI가 1-hop이다. Multi-hop을 비교 데이터에 섞으면 "회로가 달라 보이는 게 hop 차이 때문인지 modality 차이 때문인지" 구분 불가. Multi-hop은 Tier 3에서 별도로 다룬다.

---

## 3. 데이터셋 #1 — Modular Arithmetic

### 목적
2-layer transformer를 scratch 학습하여 **grokking 현상**(외우기 → 일반화 전환)을 재현하고 회로를 분석.

### 규모

| 항목 | 값 | 근거 |
|---|---|---|
| 소수 `p` | **113** | Nanda et al. 2023 표준 |
| 전체 pair 수 | **12,769** (= p²) | 가능한 모든 (a, b) 조합 |
| Train 비율 | **30%** (~3,800) | Grokking 곡선이 가장 깔끔하게 나오는 비율 |
| Test 비율 | 70% (~8,900) | 일반화 측정 |

### 형식
- 입력 토큰: `[a, b, '=']` — shape `[3]`
- 타겟: `(a + b) mod p` — 정수
- 어휘 크기: `p + 1` (숫자 0..p-1 + `=` 토큰)

### 생성 코드

```python
# data/modular.py
import torch

def make_modular_data(p=113, train_frac=0.3, seed=0):
    pairs = torch.tensor([(a, b) for a in range(p) for b in range(p)])
    labels = (pairs[:, 0] + pairs[:, 1]) % p
    eq = torch.full((pairs.size(0), 1), p)
    inputs = torch.cat([pairs, eq], dim=1)

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(inputs.size(0), generator=g)
    n_train = int(train_frac * inputs.size(0))
    tr, te = perm[:n_train], perm[n_train:]
    return (inputs[tr], labels[tr]), (inputs[te], labels[te])
```

### 검증 체크리스트
- [ ] `p² = 12,769`개 pair 모두 생성
- [ ] Train/test split이 seed 고정으로 재현 가능
- [ ] 50,000 epoch 학습 시 grokking 곡선(train loss → 0 후 test loss → 0) 관찰

---

## 4. 데이터셋 #2 — Code Variable Binding

### 4.1 Week 1 Day 1 작업 — 토큰화 검증 + 풀 확정 (생성 전 필수)

데이터 생성 코드 작성 **이전에** 반드시 이 단계 먼저.

```python
# data/_day1_verify.py
from transformers import GPT2Tokenizer

tok = GPT2Tokenizer.from_pretrained("gpt2")

# 변수명 후보 (단문자 + 짧은 영단어)
CANDIDATES = [
    "a","b","c","d","e","x","y","z","p","q","r","s",
    "cat","dog","sun","red","blue","book","desk","car",
    "apple","banana","mango","kiwi","lemon","grape",
]

# 앞에 공백 붙여서 single-token으로 인코딩되는 것만 선택
SINGLE_TOKEN_NAMES = [n for n in CANDIDATES
                      if len(tok.encode(" " + n)) == 1]

# 정답 값 후보 — single-token integer만
ANSWER_VALUES = [v for v in range(1, 20)
                 if len(tok.encode(" " + str(v))) == 1]

# 결과를 shared/config.py에 박제하여 모든 트랙이 import
print("SINGLE_TOKEN_NAMES:", SINGLE_TOKEN_NAMES)
print("ANSWER_VALUES:", ANSWER_VALUES)
```

**산출물**: `shared/config.py`에 `SINGLE_TOKEN_NAMES`, `ANSWER_VALUES` 박제.

### 4.2 Tier 구조

| Tier | 규모 | 변수명 풀 | Hop | Counterfactual | 정답값 | 용도 |
|---|---|---|---|---|---|---|
| **Tier 1 core** | 1,000개 (500 cf pair) | single-letter만 (`a,b,c,x,y,z`) | 1-hop | ✅ 필수 | 1~9 | Activation patching, head 식별 |
| **Tier 2 scaling** | 500개 | single-token 영단어 포함 | 1-hop | ❌ | 1~9 | n_vars 2~6 회로 robustness |
| **Tier 3 robustness** | 500개 (optional) | multi-token 허용 | multi-hop 포함 | ❌ | 자유 | Accuracy만 측정, 시간 남으면 |

### 4.3 Tier 1 — Core (Counterfactual Pair, Patching용)

#### 구조
모든 sample은 **clean + corrupted 한 쌍(2개)** 으로 생성. 두 prompt는 **source variable 한 토큰만** 다르고 나머지는 동일.

```
Clean:      "a=3; b=5; c=a; c="   → 정답 3
Corrupted:  "a=3; b=5; c=b; c="   → 정답 5
                       ↑
              source variable만 swap
              (single-letter 보장 → 토큰 길이 동일 → position 정렬 OK)
```

#### 스펙
- 변수 개수 `n_vars`: 2, 3 균등 (각 250 pair)
- 변수명 풀: `["a","b","c","x","y","z"]` (Day 1 검증된 것만)
- 정답값: 1~9 균등 분포
- Binding chain 길이: 1-hop 고정
- 모든 sample이 cf pair에 속함

#### 생성 로직 (의사코드)

```python
def gen_tier1_pair(n_vars, rng, pair_id):
    names = rng.sample(SINGLE_LETTER_POOL, n_vars)
    values = {v: rng.choice(ANSWER_VALUES) for v in names}
    while len(set(values.values())) < 2:   # clean/cf 답 같으면 안 됨
        values = {v: rng.choice(ANSWER_VALUES) for v in names}

    target = names[-1]                      # 마지막 변수가 target
    src_clean, src_corrupt = rng.sample([v for v in names if v != target], 2)

    prefix = "; ".join(f"{v}={values[v]}" for v in names if v != target)
    clean  = f"{prefix}; {target}={src_clean}; {target}="
    corrupt= f"{prefix}; {target}={src_corrupt}; {target}="

    return {
        "clean":   make_record(pair_id, "clean",   clean,  values[src_clean],  ...),
        "corrupt": make_record(pair_id, "corrupt", corrupt, values[src_corrupt], ...),
    }
```

### 4.4 Tier 2 — Scaling (1-hop, counterfactual 없음)

#### 목적
"발견한 회로가 n_vars 2 → 6으로 늘어나도 작동하는가" 검증.

#### 스펙
- `n_vars`: 2, 3, 4, 5, 6 각 100개씩 (균등)
- 변수명 풀: `SINGLE_TOKEN_NAMES` 전체 (단문자 + 영단어)
- Counterfactual 없음 (단일 sample)
- Binding 1-hop 고정

### 4.5 Tier 3 — Robustness (Optional)

#### 목적
"코드답게" 자연스러운 케이스에서 accuracy 측정. Patching은 안 함.

#### 스펙
- multi-token 변수명 허용 (`my_var`, `result_1`, `data_x`)
- multi-hop binding 포함 (`c=a; d=c; e=d; e=?`)
- 의도적으로 길고 다양

### 4.6 메타데이터 스키마 (JSONL 형식)

모든 sample은 아래 필드를 포함한 JSONL 한 줄로 저장.

```json
{
  "id": "vb_0001",
  "cf_id": "vb_0001_pair",
  "role": "clean",
  "tier": 1,
  "prompt": "a=3; b=5; c=a; c=",
  "answer": 3,
  "answer_token_id": 513,
  "n_vars": 3,
  "target_var": "c",
  "source_var": "a",
  "distractor_vars": ["b"],
  "binding_hop": 1,
  "source_var_token_pos": 10,
  "answer_token_pos": 13,
  "var_name_token_lens": {"a":1,"b":1,"c":1},
  "tokenization_check": "passed"
}
```

**핵심 필드**:
- `cf_id`: counterfactual pair 식별자 (Tier 1만). 분석 코드는 이걸로 join.
- `role`: `"clean"` / `"corrupt"` / `"single"` (Tier 2/3)
- `source_var_token_pos`, `answer_token_pos`: patching 코드가 직접 사용
- `tokenization_check`: 매 sample마다 자동 검증 결과 (`"passed"` 아니면 폐기)

### 4.7 검증 체크리스트 (생성 후 필수)

1. [ ] **GPT-2 small zero-shot accuracy ≥ 70%** (Tier 1 기준) — 못 풀면 회로 발견할 게 없음
2. [ ] 모든 sample의 `tokenization_check == "passed"`
3. [ ] Tier 1의 모든 cf pair에서 clean·corrupt prompt 토큰 수 **정확히 일치**
4. [ ] Tier 1 cf pair의 `source_var_token_pos`가 clean·corrupt에서 동일
5. [ ] 정답값(1~9) 분포가 균등 (특정 값 편향 시 logit 분석 왜곡)
6. [ ] 변수명 사용 빈도가 풀 내 균등

---

## 5. 데이터셋 #3 — IOI (Indirect Object Identification)

### 목적
Wang et al. (2022)이 GPT-2 small에서 발견한 **자연어 entity 추적 회로**를 본 환경에서 재현하고, Code Variable Binding 회로와 비교.

### 규모

| 용도 | 양 | 근거 |
|---|---|---|
| 회로 재현 | 500 | 제안서 명시, Wang et al.도 비슷한 규모 |
| 비교 (overlap score) | 500 | Code Binding Tier 1과 동일 N으로 통일 |

### 형식

- 템플릿: `"{A} and {B} went to the {place}. {B} gave a {obj} to ___"`
- 정답: 첫 등장 entity `{A}` (indirect object)

### 로딩 (권장: 공식 로더)

```bash
pip install git+https://github.com/redwoodresearch/Easy-Transformer.git
```

```python
from easy_transformer.ioi_dataset import IOIDataset
ds = IOIDataset(prompt_type="mixed", N=500, tokenizer=tokenizer)
```

**왜 공식 로더?** Wang et al. 26개 head 결과와 직접 비교 가능. 토크나이저·prompt 형식이 reference와 정확히 일치.

### 검증 체크리스트
- [ ] GPT-2 small에서 IOI accuracy ≥ 90% (Wang et al. 보고 수준)
- [ ] Wang et al. 26개 head 중 재현된 head 수 ≥ 20

---

## 6. 보조 — GPT-2 small

| 항목 | 값 |
|---|---|
| Params | 124M |
| Layers | 12 |
| Heads | 12 per layer (총 144 head) |
| d_model | 768 |

### 로드 (TransformerLens 권장)

```python
from transformer_lens import HookedTransformer
model = HookedTransformer.from_pretrained("gpt2")
```

**왜 TransformerLens?** Attention pattern·activation을 hook으로 추출 가능. HuggingFace `GPT2LMHeadModel`보다 회로 분석에 훨씬 편리.

---

## 7. 공통 설정 — `shared/config.py`

```python
# 세 트랙 모두 이 파일을 import. 직접 하드코딩 금지.
import torch

SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Modular
MODULAR_P = 113
MODULAR_TRAIN_FRAC = 0.3

# Code Variable Binding
VB_TIER1_N_PAIRS = 500      # → 1,000 sample
VB_TIER2_N = 500
VB_TIER3_N = 500

# IOI
IOI_N = 500

# Model
MODEL_NAME = "gpt2"

# Day 1 토큰화 검증 결과 박제 (Day 1 작업 후 실제 값으로 채움)
SINGLE_LETTER_POOL = ["a","b","c","x","y","z"]     # Tier 1 전용
SINGLE_TOKEN_NAMES = [...]                          # Tier 2 (Day 1 후 확정)
ANSWER_VALUES = [1,2,3,4,5,6,7,8,9]                # Day 1 검증 후
```

---

## 8. 저장소 구조

```
repo/
├── data/                       ← 데이터 생성 코드 (중앙집중)
│   ├── _day1_verify.py         (Week 1 Day 1 실행, 결과를 config에 박제)
│   ├── modular.py
│   ├── var_binding.py          (Tier 1/2/3 생성)
│   └── ioi_loader.py
├── shared/
│   ├── config.py               (seed, N, 풀, 정답값)
│   └── circuit_format.py       (회로 결과 JSON 스키마)
├── datasets/                   ← 생성된 데이터 (gitignore 또는 Git LFS)
│   ├── modular_train.pt
│   ├── modular_test.pt
│   ├── var_binding_tier1.jsonl
│   ├── var_binding_tier2.jsonl
│   └── var_binding_tier3.jsonl
├── tracks/                     ← 분석 코드 (트랙별)
│   ├── grokking/
│   ├── code/
│   └── ioi/
└── results/                    ← 회로 분석 결과 JSON
    ├── modular_circuit.json
    ├── code_circuit.json
    └── ioi_circuit.json
```

---

## 9. Week 1 일정 (데이터 관련)

| 시점 | 작업 | 결과물 |
|---|---|---|
| Day 1 오전 | `data/_day1_verify.py` 실행 → 변수명·정답값 풀 확정 | `shared/config.py` 커밋 |
| Day 1 오후 | Modular, IOI 데이터 생성 (즉시 완료) | `datasets/modular_*.pt`, IOI 로드 검증 |
| Day 2 | Code Binding Tier 1 (counterfactual pair) 생성 + 검증 체크리스트 통과 | `datasets/var_binding_tier1.jsonl` |
| Day 3 | Code Binding Tier 2 생성 + GPT-2 zero-shot accuracy 측정 | `datasets/var_binding_tier2.jsonl` + accuracy 보고 |
| Day 4 | (옵션) Tier 3 생성, 또는 다른 트랙 지원 | `datasets/var_binding_tier3.jsonl` |
| Day 5 | 데이터 동결 선언, 팀 전체 sync | 이후 변경 금지 원칙 명시 |

**원칙**: 데이터 동결 이후 변경하려면 팀 전체 합의 필요. 분석 시작 후 데이터 바뀌면 모든 결과 재실행해야 함.

---

## 10. 데이터 양이 병목이 아닌 이유

| 작업 | 예상 시간 (단일 GPU) |
|---|---|
| Modular 데이터 생성 | < 1초 |
| Variable binding Tier 1~3 전체 생성 | < 10초 |
| IOI 500개 로드 | 즉시 |
| **2-layer transformer scratch 학습 50k epoch** | **3 ~ 6시간** ← 최대 병목 |
| GPT-2 small에 1,000개 forward | < 1분 |
| Activation patching (500 cf pair × 144 head) | 30 ~ 60분 |

→ 데이터 양보다 **grokking 학습 시간**과 **activation patching 시간**이 일정의 critical path.

---

## 부록 A — v2가 v1과 다른 점 요약

| 항목 | v1 | v2 |
|---|---|---|
| Code Binding 구조 | flat 1,500~2,000 | Tier 1/2/3 분리 |
| Counterfactual pair | ❌ | ✅ Tier 1 필수 |
| 변수명 토큰화 검증 | ❌ (코드가 섞어 씀) | ✅ Day 1 사전 검증 |
| 정답 값 제약 | 1~20 임의 | 1~9 single-token만 |
| 1-hop vs multi-hop | 미명시 | 1-hop 정식, multi-hop은 Tier 3 |
| 메타데이터 | 최소 (prompt, answer) | 풍부 (cf_id, token_pos, check 등) |
| 저장소 구조 | 트랙별 분산 | 중앙 `data/` 폴더 |
| Week 1 Day 1 작업 | 데이터 생성만 | 토큰화 검증 먼저 → 그 다음 생성 |
