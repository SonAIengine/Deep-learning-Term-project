# Project Structure

This document provides an overview of the Deep Learning Term Project on Mechanistic Interpretability of Transformer models.

## 📁 Repository Organization

```
Deep-learning-Term-project/
├── README.md                      # Project overview and quick start guide
├── PROJECT_STRUCTURE.md           # This file - detailed project structure
│
├── shared/                        # Shared utilities and configuration
│   └── config.py                  # Centralized configuration for all tracks
│
├── tracks/                        # Analysis tracks (different tasks/models)
│   ├── grokking/                  # P1: Modular arithmetic with 1-layer model
│   │   ├── train.py               # Grokking training script
│   │   ├── analysis.py            # Circuit analysis for modular arithmetic
│   │   ├── config.py              # Track-specific configuration
│   │   ├── data.py                # Data generation for modular arithmetic
│   │   ├── model.py               # 1-layer attention model
│   │   └── visualize_analysis.py  # Visualization for grokking results
│   │
│   ├── code/                      # P2: Code variable binding with GPT-2
│   │   ├── run_analysis.py        # Main analysis script for var binding
│   │   ├── analysis.py            # Circuit analysis functions
│   │   ├── config.py              # Track-specific configuration
│   │   ├── data.py                # IOI-style data generation
│   │   └── visualize_results.py   # Visualization for code results
│   │
│   ├── analysis/                  # P3: Cross-track comparison
│   │   └── p1_p2_cross_track.py   # Cross-track circuit comparison
│   │
│   ├── syntax/                    # P4: Syntax task analysis (SVA)
│   │   └── p4_sva_analysis.py     # Subject-verb agreement analysis
│   │
│   └── semantic/                  # P5: Semantic task analysis (Sentiment)
│       └── p5_sentiment_analysis.py # Sentiment classification analysis
│
├── datasets/                      # Generated datasets for all tasks
│   ├── modular_arithmetic/        # P1 training/test data
│   ├── ioi/                       # P2 IOI-style prompts
│   ├── sva_pairs.json             # P4 SVA sentence pairs
│   └── sentiment_pairs.json       # P5 sentiment sentences
│
├── results/                       # Analysis results and visualizations
│   ├── grokking/                  # P1 results
│   ├── code/                      # P2 results
│   ├── comparison/                # P3 cross-track comparison
│   ├── sva/                       # P4 syntax results
│   └── sentiment/                 # P5 semantic results
│
└── requirements.txt               # Python dependencies

```

## 🎯 Analysis Tracks

### P1: Modular Arithmetic (Grokking)
- **Model**: 1-layer attention-only transformer
- **Task**: Modular addition (mod 113)
- **Goal**: Study circuit formation during grokking phenomenon
- **Key Insights**:
  - Emergence of algorithmic solutions after overfitting
  - Clear circuit structure with periodic activations
  - Fourier analysis reveals frequency-based computation

### P2: Code Variable Binding
- **Model**: GPT-2 small (12-layer, 144 heads)
- **Task**: Indirect Object Identification (IOI)
- **Goal**: Understand how language models track variable bindings
- **Key Insights**:
  - Name mover heads in layers 9-11
  - Backup name mover heads for robustness
  - Position and semantic information flow

### P3: Cross-Track Comparison
- **Goal**: Compare circuits across different model scales and tasks
- **Analysis**: P1 vs P2 circuit architecture
- **Key Insights**:
  - Model depth determines circuit complexity
  - Task complexity affects specialization patterns
  - Low circuit universality (0.12/1.0)

### P4: Syntax Task (SVA)
- **Model**: GPT-2 small
- **Task**: Subject-verb agreement
- **Goal**: Identify circuits for grammatical processing
- **Key Findings**:
  - Distributed syntax processing across layers
  - Similar activation patterns to P2
  - Early layers (0-2) show high importance

### P5: Semantic Task (Sentiment)
- **Model**: GPT-2 small
- **Task**: Sentiment classification
- **Goal**: Identify circuits for semantic understanding
- **Key Findings**:
  - 57.1% sentiment classification accuracy
  - Distributed semantic representation
  - Different circuit patterns than syntax tasks

## 🔧 Key Components

### Configuration Management
All tracks use `shared/config.py` for:
- Reproducibility (SEED, DEVICE)
- Model specifications (MODEL_NAME = "gpt2")
- Dataset sizes and parameters
- Path management

### Analysis Techniques
1. **Activation Patching**: Identify important components
2. **Head Ablation**: Test individual head contributions
3. **Fourier Analysis**: Analyze periodic patterns (P1)
4. **Cross-track Comparison**: Compare circuit architectures

## 📊 Results Summary

### Key Findings
1. **Model Depth Impact**: 
   - 1-layer models (P1) show direct, interpretable circuits
   - 12-layer models (P2, P4, P5) show distributed, complex circuits

2. **Task Complexity**:
   - Simple tasks (P1): Highly specialized circuits
   - Complex tasks (P2, P5): Distributed processing across multiple heads

3. **Circuit Universality**:
   - Low overlap between different tasks
   - Task-specific circuit formations
   - Limited generalization across task types

## 🚀 Running the Analysis

### Individual Track Analysis
```bash
# P1: Grokking analysis
python tracks/grokking/train.py

# P2: Variable binding analysis  
python tracks/code/run_analysis.py

# P3: Cross-track comparison
python tracks/analysis/p1_p2_cross_track.py

# P4: Syntax task analysis
python tracks/syntax/p4_sva_analysis.py

# P5: Semantic task analysis
python tracks/semantic/p5_sentiment_analysis.py
```

### Full Pipeline
Run tracks sequentially for complete analysis:
1. Train P1 model and analyze circuits
2. Run P2 analysis on GPT-2
3. Compare P1 and P2 architectures
4. Analyze P4 syntax task
5. Analyze P5 semantic task

## 📝 Notes

- All results are saved in `results/` directory with JSON format
- Visualizations are generated as PNG files
- Each track is self-contained with its own configuration
- Shared utilities are in `shared/config.py`
