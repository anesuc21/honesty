# Honesty Alignment in Large Language Models

This repository contains the code, evaluation pipeline, and visualisation scripts for my undergraduate thesis: **Improving Honesty Alignment in Large Language Models through Probe-Based Knowledge Detection and Inference-Time Intervention**.

The work builds on and extends the [Alignment for Honesty](https://arxiv.org/abs/2312.07000) paper by Yang et al. (2024), reproducing their framework on LLaMA-2-7B and proposing two improvements: a probe-based knowledge detector and inference-time activation steering.

---

## Models

All fine-tuned models used for probing and steering experiment are publicly available on HuggingFace:

| Model | HuggingFace | Base | Method |
|---|---|---|---|
| Absolute | [Chibz21/llama2-7b-honesty-absolute](https://huggingface.co/Chibz21/llama2-7b-honesty-absolute) | LLaMA-2-7B | Absolute SFT |
| Confidence-Verb | [Chibz21/llama2-7b-honesty-confidence-verb](https://huggingface.co/Chibz21/llama2-7b-honesty-confidence-verb) | LLaMA-2-7B | Confidence-Verb SFT |
| Confidence-Num | [Chibz21/llama2-7b-honesty-confidence-num](https://huggingface.co/Chibz21/llama2-7b-honesty-confidence-num) | LLaMA-2-7B | Confidence-Num SFT |
| SFT Baseline | [Chibz21/llama2-7b-honesty-sft-baseline](https://huggingface.co/Chibz21/llama2-7b-honesty-sft-baseline) | LLaMA-2-7B | SFT Baseline |

---

## Overview

### What this repo improves on the original paper

**Improvement 1 — Probe-Based Knowledge Detection**

The original paper determines whether a model knows the answer to a question by sampling 10 responses and measuring expected accuracy (the K function). This is computationally expensive and introduces label noise — a model may answer correctly by chance on a question it does not reliably know.

This work replaces the sampling-based K function with an MLP probe trained on the model's internal hidden states from layers [16, 20, 24, 28, 31]. The probe achieves 72.25% accuracy in distinguishing known from unknown questions from a single forward pass, and reduces training cost by ~10x. The most significant downstream effect is a +21.43% accuracy improvement for the SFT Baseline, which is particularly sensitive to label noise.

**Improvement 2 — Inference-Time Intervention (ITI) Steering**

All SFT methods require retraining to adjust honesty behaviour. This work applies ITI steering to the honesty alignment setting for the first time, steering truth-sensitive attention heads at inference time without modifying any weights.

Best results: ITI on the absolute model at α=0.4 improves honesty from 71.84% to **78.46%** (+6.62 percentage points) without retraining.

---

## Key Results

### Probe-Based Knowledge Detection (TriviaQA, 9,961 instances)

| Method | Honesty (sampling) | Honesty (probe) | Accuracy (sampling) | Accuracy (probe) |
|---|---|---|---|---|
| SFT Baseline | 50.02 | 50.24 | 66.25 | **87.68** (+21.43) |
| Absolute | 72.00 | 71.37 | 81.29 | 81.24 |
| Conf-Verb | 68.71 | 68.17 | 72.86 | 72.63 |
| Multisample | 75.00 | 75.21 | 70.37 | 70.48 |

### ITI Steering — Absolute Model (TriviaQA, 500 instances)

| Alpha | Prudence | Over-Consv | Honesty | Accuracy |
|---|---|---|---|---|
| 0 (baseline) | 55.20 | 11.51 | 71.84 | 80.27 |
| 0.1 | 68.29 | 21.43 | 73.43 | 66.80 |
| 0.2 | 72.22 | 23.38 | 74.42 | 66.80 |
| 0.3 | 76.80 | 25.32 | 75.74 | 67.00 |
| **0.4** | **85.48** | **28.57** | **78.46** | 67.20 |
| 0.5 | 82.09 | 30.84 | 75.62 | 64.80 |
| 1.0 | 90.54 | 50.00 | 70.27 | 57.40 |
| 2.0 | 99.44 | 87.34 | 56.05 | 42.80 |
| 3.0 | 99.47 | 95.45 | 52.01 | 29.20 |

### ITI Summary Across Methods

| Method | Baseline Honesty | Best ITI Honesty | Best Alpha | Gain |
|---|---|---|---|---|
| Absolute | 71.84 | 78.46 | 0.4 | +6.62 |
| Conf-Verb | 68.17 | 70.64 | 0.2 | +2.47 |
| Conf-Num | 67.23 | 68.41 | 0.5 | +1.18 |

---

## Repository Structure

```
honesty/
├── train/                          # SFT training scripts
│   ├── train_absolute.py
│   ├── train_confidence_verb.py
│   ├── train_confidence_num.py
│   └── train_sft_baseline.py
│
├── probe/                          # Probe and ITI scripts
│   ├── extract_hidden_states.py    # Extract hidden states from layers
│   ├── train_probe.py              # Train MLP/logistic regression probes
│   ├── probe_knowledge.py          # Probe-based K function
│   └── iti_steering.py             # ITI head probing and steering
│
├── evaluation/                     # Evaluation pipeline
│   ├── triviaqa_nonambigqa_evaluation.py
│   ├── run_local_eval.py           # Mistral-7B local evaluation
│   └── run_evaluate.py             # Honesty metrics computation
│
├── figure/                         # Generated figures
│   ├── head_accuracy_heatmap.png   # ITI head probe heatmap (absolute)
│   ├── pca_best_head.png           # PCA of best head activations
│   ├── iti_alpha_sweep.png         # Alpha sweep chart
│   └── iti_honesty_comparison.png  # Cross-method ITI comparison
│
├── visualize_activations.py        # Generate PCA and heatmap figures
├── visualize_alpha_sweeps.py       # Generate alpha sweep charts
└── .gitignore
```

---

## Setup

### Requirements

```bash
pip install transformers==4.43.2 torch accelerate tqdm scikit-learn \
            joblib numpy vllm==0.6.0 bitsandbytes matplotlib \
            --break-system-packages
```

### HuggingFace Authentication

LLaMA-2 is a gated model and requires authentication:

```python
from huggingface_hub import login
login(token='your_token_here')
```

Tokens are passed at runtime and should never be committed to version control.

---

## Reproduction

### Step 1 — Train SFT Methods

```bash
# Train absolute method
python train/train_absolute.py

# Train confidence-verb method  
python train/train_confidence_verb.py

# Train confidence-num method
python train/train_confidence_num.py
```

### Step 2 — Evaluate

```bash
# Generate predictions
python evaluation/run_local_eval.py

# Compute honesty metrics
python evaluation/run_evaluate.py
```

---

## Probe-Based Knowledge Detection

### Step 1 — Extract Hidden States

```bash
python probe/extract_hidden_states.py
```

Extracts last-token hidden states from layers [16, 20, 24, 28, 31] for 8,000 TriviaQA questions.

### Step 2 — Train Probe

```bash
python probe/train_probe.py
```

Trains and compares logistic regression and MLP probes across single-layer and concatenated multi-layer configurations. Saves the best probe (MLP, 72.25% accuracy).

### Step 3 — Retrain with Probe Labels

Use `probe/probe_knowledge.py` to generate probe-based labels, then retrain using the standard training scripts with the new labelled data.

---

## ITI Steering

### Configuration

Update the following in `probe/iti_steering.py`:

```python
MODEL_PATH   = "Chibz21/llama2-7b-honesty-absolute"   # or conf-verb / conf-num
DATA_PATH    = "evaluation/results/triviaqa_probe_absolute/aligned_eval.jsonl"
OUTPUT_DIR   = "evaluation/results/triviaqa_iti_absolute"
ALPHA_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5]              # steering strengths to sweep
```

### Run ITI

```bash
python probe/iti_steering.py
```

This will:
1. Load the fine-tuned model
2. Extract per-head activations from all 1,024 attention heads
3. Probe each head to identify the top 48 most truth-sensitive heads
4. Compute mean-difference steering directions
5. Generate predictions for each alpha value

### Evaluate ITI Results

```bash
for alpha in 0.1 0.2 0.3 0.4 0.5; do
    cp "evaluation/results/triviaqa_iti_absolute/eval_predictions_alpha${alpha}.jsonl" \
       evaluation/results/triviaqa_iti_absolute/eval_predictions.jsonl
    python evaluation/run_local_eval.py
    python evaluation/run_evaluate.py
    cp evaluation/results/triviaqa_iti_absolute/post_metrics.json \
       "evaluation/results/triviaqa_iti_absolute/post_metrics_alpha${alpha}.json"
done
```

---

## Visualisations

### Head Accuracy Heatmap + PCA

```bash
python visualize_activations.py
```

Generates:
- `figure/head_accuracy_heatmap.png` — probe accuracy across all 1,024 heads
- `figure/pca_best_head.png` — PCA of best head activations
- `figure/pca_steering_direction.png` — PCA with steering direction arrow
- `figure/iti_alpha_sweep.png` — alpha sweep chart

### ITI Comparison Charts

```bash
python visualize_alpha_sweeps.py
```

Generates per-method and combined honesty comparison charts.

---

## Environment

- GPU: A100 80GB (single)
- CUDA: 12.4
- Python: 3.11
- vLLM: 0.6.0 (required — newer versions incompatible with CUDA 12.4)
- Transformers: 4.43.2

> **Note:** Use 4-bit quantization (`BitsAndBytesConfig(load_in_4bit=True)`) if disk space is limited when loading models from HuggingFace Hub.

---

## Key Finding

Probing all 1,024 attention heads across three fine-tuned models revealed that **different fine-tuning methods cause epistemic uncertainty to be encoded in systematically different parts of the network**:

| Model | Best Head | Layer | Probe Accuracy |
|---|---|---|---|
| Absolute | Head 12 | Layer 15 | 83.0% |
| Conf-Verb | Head 5 | Layer 25 | 77.5% |
| Conf-Num | Head 30 | Layer 26 | ~75% |

The absolute method's clean binary training signal produces concentrated uncertainty representations in middle layers. Confidence prefix methods distribute uncertainty encoding deeper into the network, reducing ITI effectiveness.

---

## Citation

If you use this code, please also cite the original paper this work builds on:

```bibtex
@article{yang2023alignment,
  title={Alignment for Honesty},
  author={Yang, Yuqing and Chern, Ethan and Qiu, Xipeng and Neubig, Graham and Liu, Pengfei},
  journal={arXiv preprint arXiv:2312.07000},
  year={2023}
}
```

---

## Acknowledgements

This work was completed as an undergraduate thesis. The base framework, evaluation pipeline, and training data are from the original [GAIR-NLP alignment-for-honesty](https://github.com/GAIR-NLP/alignment-for-honesty) repository. The probe-based knowledge detection and ITI steering improvements are original contributions of this thesis.
