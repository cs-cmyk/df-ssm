# DF-SSM: Density Field State Space Models

**1-Bit Distillation, Efficient Inference, and Knowledge Organization in Mamba-2**

## Overview

DF-SSM compresses Mamba-2 1.3B from 2,688 MB to **278 MB** (9.7× smaller) with **21.4× faster** GPU inference (batch=1, vs mamba-ssm reference implementation), trained in **6 hours on 1 GPU** using only **32M distillation tokens** (presupposes a pretrained FP16 teacher).

The speedup is primarily memory-bandwidth-driven: 1-bit packed scaffold weights require 8× less HBM transfer, and cuBLAS INT8 tensor cores provide ~2× compute throughput over FP16. Custom CUDA kernels handle only the stateful SSM and convolution operations (~5% of per-layer compute).

| Metric | FP16 Teacher | DF-SSM |
|---|---|---|
| Size | 2,688 MB | 278 MB (9.7×) |
| GPU batch=1 | 14 tok/s | 299 tok/s (21.4×) |
| GPU batch=32 | 482 tok/s | 1,963 tok/s (4.1×) |
| CPU | 12 tok/s | 22 tok/s (1.8×) |
| BoolQ | — | 60.8% |
| PIQA | — | 67.1% |
| Training | 300B tokens | 32M tokens |

## Quick Start

### Requirements

```bash
pip install torch transformers einops datasets mamba-ssm lm-eval
```

### Training

**Step 1: DFW distillation (~2.5 hours)**
```bash
python training/df_ssm_mamba2_distill_dfw.py \
    --teacher state-spaces/mamba2-1.3b \
    --tokens 12M --Ks 23 --Kw 16
```

**Step 2: LoRA correction (~3.4 hours)**
```bash
python training/df_ssm_dfw_lora.py \
    --scaffold dfssm_dfw_step1501.pt \
    --lora-layers all --lora-rank 16 \
    --tokens 20M
```

**Step 3: Export**
```bash
python training/export_dfssm.py \
    --scaffold dfssm_dfw_step1501.pt \
    --lora dfw_lora_all_r16_final.pt \
    --output model.dfssm
```

### Inference

**GPU:**
```bash
python inference/gpu/df_ssm_inference.py model.dfssm --n_tokens 128
```

**CPU:**
```bash
gcc -O3 -march=native -fopenmp -shared -fPIC \
    -o libcpu_inference.so inference/cpu/cpu_inference.c -lm
OMP_NUM_THREADS=4 python inference/cpu/cpu_inference_driver.py model.dfssm
```

### Evaluation

```bash
python eval/downstream_eval.py \
    --scaffold dfssm_dfw_step1501.pt \
    --lora dfw_lora_all_r16_final.pt \
    --tasks boolq,piqa,hellaswag,winogrande,arc_easy
```

### Interpretability

```bash
# Knowledge atlas (445 prompts × 19 categories × 4 layers)
python interpretability/atlas_large.py

# Render 4-panel figure
python interpretability/render_atlas.py atlas_large.json

# Knowledge localization (causal intervention)
python interpretability/knowledge_systematic.py

# Three-phase analysis
python interpretability/knowledge_crystallization.py

# Logit lens
python interpretability/logit_lens.py

# Category space analysis
python interpretability/category_space.py
```

### Paper Figures

```bash
python paper/fig1_benchmarks.py
python interpretability/render_atlas.py atlas_large.json -o paper/figures/fig2_atlas.png
python paper/fig3_rank_trajectories.py
python paper/fig4_three_phases.py
python paper/fig5_separation_heatmap.py
```

## Architecture

```
Input token
    │
    ▼
Embedding (int8, 103 MB)
    │
    ▼ ×48 layers
┌──────────────────────────────────────────────┐
│  RMSNorm                          [PyTorch]  │
│  in_proj:  scaffold(1-bit) matmul [cuBLAS]   │
│            + LoRA(int8)           [PyTorch]   │
│  Conv1d shift register       [custom CUDA]   │
│  SSM step (state cached)     [custom CUDA]   │
│  Gate + RMSNorm                   [PyTorch]  │
│  out_proj: scaffold(1-bit) matmul [cuBLAS]   │
│            + LoRA(int8)           [PyTorch]   │
│  Residual add                                │
└──────────────────────────────────────────────┘
    │
    ▼
Final RMSNorm → LM Head (tied with embedding)
```

The scaffold matmul uses `torch._int_mm` (cuBLAS INT8 tensor cores).
Custom CUDA kernels are only for stateful ops (conv step, SSM step).
The speedup is primarily memory-bandwidth-driven (8× less weight data from HBM).

## Model Composition

| Component | Precision | Size | % |
|---|---|---|---|
| Scaffold | 1-bit packed | 155 MB | 56% |
| Embedding | int8 | 103 MB | 37% |
| LoRA | int8 | 12 MB | 4% |
| Other | FP16/FP32 | 8 MB | 3% |
| **Total** | | **278 MB** | |

## Key Findings

### Compression
- Int8 LoRA quantization is lossless (PPL 49.2 → 49.1)
- Int8 embedding quantization is lossless (PPL 49.2 → 49.2)
- LoRA is NOT a rounding table (zero correlation with quantization residual)

### Interpretability
- **Three-phase processing:** Categorize (L0-3) → Recall (L25-35) → Format (L36-47)
- **Knowledge localization:** 45/45 capital pairs flip at L32-L36
- **Intent space:** 94% classification accuracy at L3, zero vocabulary alignment
- **Template-first classification:** Uniform-template categories cluster instantly; diverse-template categories never fully separate
- **Structure precedes strength:** Knowledge organization develops before factual recall

## Citation

```bibtex
@article{dfssm2026,
  title={Density Field State Space Models: 1-Bit Distillation, Efficient Inference, and Knowledge Organization in Mamba-2},
  author={},
  year={2026}
}
```

## License

[TBD]
