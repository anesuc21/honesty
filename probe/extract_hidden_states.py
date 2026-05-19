# probe/extract_hidden_states.py
# Step 1: Extract hidden states from LLaMA-2-7B on TriviaQA questions
# Saves both last token and mean pooled hidden states for all probe layers

import os
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATH  = "meta-llama/Llama-2-7b-chat-hf"
DATA_PATH   = "/workspace/honesty/data/training_data/triviaqa_7b.jsonl"
OUTPUT_PATH = "/workspace/honesty/probe/hidden_states.npz"

PROBE_LAYERS = [16, 20, 24, 28, 31]

PROMPT_TEMPLATE = "Q: {question}\nA:"

# ============================================================================
# LOAD MODEL
# ============================================================================

print(f"Loading model: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    output_hidden_states=True,
)
model.eval()
print("✓ Model loaded")

# ============================================================================
# LOAD DATA
# ============================================================================

print(f"\nLoading data: {DATA_PATH}")
data = [json.loads(l) for l in open(DATA_PATH)]
print(f"✓ Loaded {len(data)} samples")

# ============================================================================
# EXTRACT HIDDEN STATES
# ============================================================================

def get_label(instance):
    return 1 if instance['greedy_label'] == 'known' else 0

def get_soft_label(instance):
    return instance['sampling_knowns'] / 10.0

print(f"\nExtracting hidden states from layers: {PROBE_LAYERS}")
print(f"Saving both last token and mean pooled representations")

all_hidden_states_last = {layer: [] for layer in PROBE_LAYERS}
all_hidden_states_mean = {layer: [] for layer in PROBE_LAYERS}
all_labels      = []
all_soft_labels = []

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

with torch.no_grad():
    for instance in tqdm(data, desc="Extracting"):
        question = instance['question']
        prompt   = PROMPT_TEMPLATE.format(question=question)

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(model.device)

        outputs = model(**inputs, output_hidden_states=True)

        for layer in PROBE_LAYERS:
            hidden = outputs.hidden_states[layer]  # (1, seq_len, hidden_dim)

            # Last token hidden state
            last_token = hidden[0, -1, :].float().cpu().numpy()
            all_hidden_states_last[layer].append(last_token)

            # Mean pooling across all tokens
            attention_mask = inputs['attention_mask'][0]
            masked_hidden  = hidden[0] * attention_mask.unsqueeze(-1).float()
            mean_token = (masked_hidden.sum(dim=0) / attention_mask.sum()).float().cpu().numpy()
            all_hidden_states_mean[layer].append(mean_token)

        all_labels.append(get_label(instance))
        all_soft_labels.append(get_soft_label(instance))

# Convert to numpy
hidden_arrays = {}
for layer in PROBE_LAYERS:
    hidden_arrays[f"layer_{layer}_last"] = np.array(all_hidden_states_last[layer])
    hidden_arrays[f"layer_{layer}_mean"] = np.array(all_hidden_states_mean[layer])

labels      = np.array(all_labels)
soft_labels = np.array(all_soft_labels)

print(f"\n✓ Extracted hidden states")
print(f"  Shape per layer: {hidden_arrays[f'layer_{PROBE_LAYERS[0]}_last'].shape}")
print(f"  Labels: {labels.sum()} known, {(1-labels).sum()} unknown")

np.savez(OUTPUT_PATH, **hidden_arrays, labels=labels, soft_labels=soft_labels)
print(f"\n✓ Saved to {OUTPUT_PATH}")