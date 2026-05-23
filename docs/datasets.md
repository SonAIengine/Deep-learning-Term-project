# 데이터셋 명세 — Transformer Circuit Analysis

> **프로젝트**: Transformer Circuit Analysis — Grokking과 Code Variable Binding
> **관련 제안서**: [sonsj-proposal.md](candidates/sonsj-proposal.md)
> **작성일**: 2026-05-23

본 문서는 프로젝트에서 사용할 데이터셋의 종류·규모·생성 방법·검증 기준을 정리한다. Week 1 첫날에 아래 데이터를 모두 생성·고정하여 세 트랙(P1·P2·P3)이 같은 데이터로 평행 작업할 수 있게 한다.

---

## 1. 데이터셋 전체 개요

| # | 데이터셋 | 사용 단계 | 규모 | 출처 | 담당 트랙 |
|---|---|---|---|---|---|
| 1 | Modular Arithmetic | Step 1~2 | 12,769 pair (train 3,800 / test 8,900) | 직접 생성 | P1 |
| 2 | Code Variable Binding | Step 3 | 1,500 ~ 2,000 sample | 직접 생성 | P2 |
| 3 | IOI (Indirect Object Identification) | Step 3 비교 | 500 sample | `Easy-Transformer` 또는 직접 생성 | P3 |
| - | GPT-2 small (모델) | Step 3 | 124M params | HuggingFace / TransformerLens | P2·P3 공통 |

**총 데이터 규모**: 약 15,000 sample. 데이터 양 자체는 작으며, 학습·분석 시간이 진짜 병목이다.

---

## 2. 데이터셋 #1 — Modular Arithmetic

### 목적
2-layer transformer를 scratch 학습하여 **grokking 현상**(외우기 → 일반화 전환)을 재현하고 회로를 분석한다.

### 규모

| 항목 | 값 | 근거 |
|---|---|---|
| 소수 `p` | **113** | Nanda et al. 2023 표준 설정 |
| 전체 pair 수 | **12,769개** (= p²) | 가능한 모든 (a, b) 조합 |
| Train 비율 | **30%** (약 3,800개) | Grokking 곡선이 가장 깔끔하게 나오는 비율. 100%면 그냥 memorize, 10%면 학습 자체가 안 됨 |
| Test 비율 | 70% (약 8,900개) | 일반화 측정 |

### 형식
- 입력 토큰: `[a, b, '=']` — shape `[3]`
- 타겟: `(a + b) mod p` — 정수 한 개
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

### 검증 기준
- [ ] `p² = 12,769` 개 pair 모두 생성되었는가
- [ ] Train/test split이 seed 고정으로 재현 가능한가
- [ ] 50,000 epoch 학습 시 grokking 곡선(train loss → 0 후 test loss → 0) 관찰되는가

---

## 3. 데이터셋 #2 — Code Variable Binding

### 목적
GPT-2 small에서 **코드의 변수 추적 회로**를 발견한다. Activation patching·ablation의 평가 데이터로 사용.

### 규모

| 용도 | 양 | 근거 |
|---|---|---|
| Task accuracy 측정 | 500 ~ 1,000개 | ±2% 신뢰구간 확보 |
| Activation patching | 100 ~ 200개 | 패치는 sample당 비쌈 (head·layer 수만큼 forward pass) |
| 회로 검증 (ablation) | 500개 | head 끄고 재측정 |
| **총 권장** | **1,500 ~ 2,000개** | 제안서 "1,000 ~ 5,000" 범위 하단 |

**양보다 분포 다양성이 중요.** GPT-2를 fine-tuning하지 않으므로 학습 데이터가 아닌 평가용. IOI 페이퍼도 분석에 수백~수천 샘플만 사용.

### 분포 설계 (1,500개 기준)

| 차원 | 분포 |
|---|---|
| 변수 개수 | 2개, 3개, 4개, 5개, 6개 각 300개씩 (균등) |
| 변수 이름 형태 | 단문자(`a`, `x`) / 단어(`apple`, `mango`) / 임의 토큰(`xqz`) — 균등 |
| Binding chain 길이 | 1단계(`c = a`) 기본, 2단계(`c = a; d = c`) 20% |

### 형식

- 입력: 변수 할당·재할당이 포함된 Python 스타일 코드 한 줄 (세미콜론 구분)
- 타겟: 마지막 `?` 위치의 정답 정수

예시:
```
"apple = 3; banana = 5; cherry = 7; mango = apple; mango ="  → 3
"a = 2; b = 8; c = a; c ="                                   → 2
"x = 1; y = 4; z = 9; w = y; w ="                            → 4
```

### 생성 코드

```python
# data/var_binding.py
import random

NAMES_POOL = {
    "short":  ["a","b","c","x","y","z","p","q"],
    "word":   ["apple","banana","cherry","mango","kiwi","grape","lemon"],
    "random": ["xqz","plk","mvb","tre","wsd","kjh","nbg"],
}

def gen_sample(n_vars, name_style, rng):
    names = rng.sample(NAMES_POOL[name_style], n_vars)
    values = {v: rng.randint(1, 20) for v in names}
    lines = [f"{v} = {values[v]}" for v in names]
    src, dst = rng.sample(names, 2)
    lines.append(f"{dst} = {src}")
    prompt = "; ".join(lines) + f"; {dst} ="
    return prompt, values[src]

def make_dataset(n=1500, seed=0):
    rng = random.Random(seed)
    styles = ["short", "word", "random"]
    var_counts = [2, 3, 4, 5, 6]
    data = []
    for i in range(n):
        n_vars = var_counts[i % len(var_counts)]
        style = styles[i % len(styles)]
        data.append(gen_sample(n_vars, style, rng))
    return data
```

### 검증 기준
- [ ] 변수 개수별 샘플 수가 균등한가 (각 300개 ±10%)
- [ ] 변수 이름 스타일별 샘플 수가 균등한가
- [ ] 모든 샘플의 정답이 1~20 범위 정수인가
- [ ] GPT-2 small의 baseline accuracy ≥ 70% (회로 분석이 의미 있으려면)

---

## 4. 데이터셋 #3 — IOI (Indirect Object Identification)

### 목적
Wang et al. (2022)이 GPT-2 small에서 발견한 **자연어 entity 추적 회로**를 본 환경에서 재현하고, Code Variable Binding 회로와 비교한다.

### 규모

| 용도 | 양 | 근거 |
|---|---|---|
| 회로 재현 | 500개 | 제안서 명시, Wang et al.도 비슷한 규모 |
| 비교 (overlap score) | 500개 | Code Binding과 동일 N으로 통일 |

**왜 500개로 충분?** IOI는 템플릿 기반이라 본질적으로 같은 패턴 반복. 1,000개로 늘려도 회로 분석 결과 거의 안 바뀜.

### 형식

- 템플릿: `"{A} and {B} went to the {place}. {B} gave a {obj} to ___"`
- 정답: 첫 등장 entity `{A}` (indirect object)

예시:
```
"Mary and John went to the store. John gave a drink to"  → Mary
"Alice and Bob went to the park. Bob gave a book to"     → Alice
```

### 로딩 방법 (권장)

공식 `Easy-Transformer` 저장소의 `IOIDataset` 사용:

```bash
pip install git+https://github.com/redwoodresearch/Easy-Transformer.git
```

```python
from easy_transformer.ioi_dataset import IOIDataset
ds = IOIDataset(prompt_type="mixed", N=500, tokenizer=tokenizer)
```

**왜 직접 생성보다 공식 로더?** Wang et al.이 발견한 26개 head 결과와 직접 비교 가능. 토크나이저 호환·prompt 형식이 reference와 정확히 일치.

### 검증 기준
- [ ] GPT-2 small에서 IOI accuracy ≥ 90% (Wang et al. 보고 수준)
- [ ] Wang et al. 26개 head 중 본 환경에서 재현된 head 수 ≥ 20

---

## 5. 보조 — GPT-2 small

### 사양
- Params: 124M
- Layers: 12
- Heads: 12 per layer (총 144 head)
- d_model: 768

### 로드 (TransformerLens 권장)

```python
from transformer_lens import HookedTransformer
model = HookedTransformer.from_pretrained("gpt2")
```

**왜 TransformerLens?** Attention pattern·activation을 hook으로 쉽게 추출 가능. HuggingFace `GPT2LMHeadModel`보다 회로 분석에 훨씬 편리.

---

## 6. 공통 설정 (모든 트랙이 import)

```python
# shared/config.py
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODULAR_P = 113
MODULAR_TRAIN_FRAC = 0.3

VAR_BINDING_N = 1500
IOI_N = 500

MODEL_NAME = "gpt2"  # GPT-2 small
```

**규칙**: 세 트랙 모두 seed·디바이스·N을 이 파일에서 import. 직접 하드코딩 금지 → 결과 호환 보장.

---

## 7. 데이터 생성 일정

| 시점 | 작업 | 담당 |
|---|---|---|
| Week 1 Day 1 | 위 3개 데이터셋 모두 생성·디스크 저장 (각 트랙 폴더에) | P1·P2·P3 각자 |
| Week 1 Day 1 | `shared/config.py` 커밋 | P3 |
| Week 1 Day 2~ | 각 트랙별 독립 작업 시작 | 전원 |

**원칙**: 데이터 생성은 모두 < 1분이면 끝남. Week 1 첫날에 고정하고 이후 변경 금지.

---

## 8. 저장소 구조 (제안)

```
repo/
├── shared/
│   ├── config.py              # 시드·N·모델명 등 공통 상수
│   └── circuit_format.py      # 회로 결과 JSON 스키마
├── tracks/
│   ├── grokking/              # P1 전담
│   │   └── data/modular.py
│   ├── code/                  # P2 전담
│   │   └── data/var_binding.py
│   └── ioi/                   # P3 전담
│       └── data/ioi_loader.py
└── results/                   # 회로 분석 결과 JSON
    ├── modular_circuit.json
    ├── code_circuit.json
    └── ioi_circuit.json
```

---

## 9. 참고 — 데이터 양이 병목이 아닌 이유

| 작업 | 예상 시간 (단일 GPU) |
|---|---|
| Modular 데이터 생성 | < 1초 |
| Variable binding 1,500개 생성 | < 1초 |
| IOI 500개 로드 | 즉시 |
| **2-layer transformer scratch 학습 50k epoch** | **3 ~ 6시간** ← 최대 병목 |
| GPT-2 small에 1,000개 forward | < 1분 |
| Activation patching (100 sample × 144 head) | 10 ~ 30분 |

→ 데이터 양보다 **P1의 grokking 학습 시간**과 **P2의 activation patching 시간**이 일정의 critical path.
