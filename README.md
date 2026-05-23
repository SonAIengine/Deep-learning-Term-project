# Transformer Circuit Analysis — Grokking & Code Variable Binding

> Mechanistic Interpretability term project. See [docs/candidates/sonsj-proposal.md](docs/candidates/sonsj-proposal.md) for the project proposal and [docs/datasets.md](docs/datasets.md) for the dataset specification.

## Repository Layout

```
.
├── data/             # Data generation scripts (centralized)
│   ├── _day1_verify.py    # Week 1 Day 1: tokenizer verification → update shared/config.py
│   ├── modular.py         # Modular arithmetic (Step 1~2)
│   ├── var_binding.py     # Code Variable Binding Tier 1/2/3 (Step 3)
│   └── ioi_loader.py      # IOI dataset wrapper (Step 3 비교)
├── shared/           # Shared config + result schema (import everywhere)
│   ├── config.py
│   └── circuit_format.py
├── datasets/         # Generated data (gitignored — regenerate from data/)
├── tracks/           # Per-track analysis code
│   ├── grokking/     # P1: Step 1~2
│   ├── code/         # P2: Step 3 code-binding circuit discovery
│   └── ioi/          # P3: Step 3 IOI replication + comparison
├── results/          # Circuit-analysis JSON outputs (gitignored)
└── docs/             # Proposal, dataset spec, meeting notes
```

## Quick Start

```bash
# 1. Install
pip install torch transformers transformer_lens einops
pip install git+https://github.com/redwoodresearch/Easy-Transformer.git

# 2. Week 1 Day 1: verify tokenizer pools, update shared/config.py with output
python -m data._day1_verify

# 3. Generate all data
python -m data.modular
python -m data.var_binding
python -m data.ioi_loader   # sanity check
```

## Team Tracks

| Track | Owner | Scope |
|---|---|---|
| Grokking (Step 1~2) | P1 | Modular arithmetic scratch training, Fourier analysis, ablation |
| Code Binding (Step 3-1/3-2) | P2 | GPT-2 + variable binding task, activation patching, ablation |
| IOI + Comparison (Step 3-3) | P3 | IOI replication, Wang et al. 26-head reproduction, universality score |

See [docs/datasets.md](docs/datasets.md) §9 for the Week 1 day-by-day schedule.
