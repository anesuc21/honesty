# probe/probe_knowledge.py
# Step 3: Replace the sampling-based K function with the trained probe
# Generates new training data using probe-based known/unknown labels
 
import os
import json
import torch
import numpy as np
import joblib
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
 
# ============================================================================
# CONFIGURATION
# ============================================================================
 
MODEL_PATH      = "meta-llama/Llama-2-7b-chat-hf"
DATA_PATH       = "/workspace/honesty/data/training_data/triviaqa_7b.jsonl"
PROBE_DIR       = "/workspace/honesty/probe/"
OUTPUT_PATH     = "/workspace/honesty/data/training_data/triviaqa_7b_probe.jsonl"
 
PROMPT_TEMPLATE = "Q: {question}\nA:"
 
# ============================================================================
# LOAD PROBE
# ============================================================================
 
print("Loading probe...")
probe = joblib.load(os.path.join(PROBE_DIR, 'probe.pkl'))
scaler = joblib.load(os.path.join(PROBE_DIR, 'scaler.pkl'))
 
with open(os.path.join(PROBE_DIR, 'probe_config.json')) as f:
    config = json.load(f)
 
best_layer = config['best_layer']
print(f"✓ Loaded probe (best layer: {best_layer}, accuracy: {config['best_accuracy']:.4f})")
 
# ============================================================================
# LOAD MODEL
# ============================================================================
 
print(f"\nLoading model: {MODEL_PATH}")
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
# PROBE-BASED K FUNCTION
# ============================================================================
 
def probe_knows(question):
    """
    Returns (label, confidence) where:
    - label: 'known' or 'unknown'
    - confidence: probability of 'known' (0-1)
    Single forward pass — replaces 10-sample K function.
    """
    prompt = PROMPT_TEMPLATE.format(question=question)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(model.device)
 
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
 
    # Extract hidden state at best layer, last token
    hidden = outputs.hidden_states[best_layer]
    last_token = hidden[0, -1, :].float().cpu().numpy().reshape(1, -1)
 
    # Scale and predict
    last_token_scaled = scaler.transform(last_token)
    prob_known = probe.predict_proba(last_token_scaled)[0][1]
    label = 'known' if prob_known >= 0.5 else 'unknown'
 
    return label, float(prob_known)
 
# ============================================================================
# GENERATE PROBE-BASED LABELS
# ============================================================================
 
print("\nGenerating probe-based K labels...")
 
agree = 0
disagree = 0
results = []
 
for instance in tqdm(data, desc="Probing"):
    probe_label, probe_conf = probe_knows(instance['question'])
    original_label = instance['greedy_label']
 
    if probe_label == original_label:
        agree += 1
    else:
        disagree += 1
 
    # Add probe labels to instance
    new_instance = dict(instance)
    new_instance['probe_label'] = probe_label
    new_instance['probe_confidence'] = probe_conf
    new_instance['original_greedy_label'] = original_label
    results.append(new_instance)
 
print(f"\n✓ Probing complete")
print(f"  Agreement with greedy label: {agree}/{len(data)} ({agree/len(data)*100:.1f}%)")
print(f"  Disagreement: {disagree}/{len(data)} ({disagree/len(data)*100:.1f}%)")
 
# Label distribution
probe_known = sum(1 for r in results if r['probe_label'] == 'known')
probe_unknown = sum(1 for r in results if r['probe_label'] == 'unknown')
orig_known = sum(1 for r in results if r['original_greedy_label'] == 'known')
orig_unknown = sum(1 for r in results if r['original_greedy_label'] == 'unknown')
 
print(f"\n  Original:  {orig_known} known, {orig_unknown} unknown")
print(f"  Probe:     {probe_known} known, {probe_unknown} unknown")
 
# ============================================================================
# SAVE
# ============================================================================
 
with open(OUTPUT_PATH, 'w') as f:
    for instance in results:
        f.write(json.dumps(instance) + '\n')
 
print(f"\n✓ Saved probe-labelled data to {OUTPUT_PATH}")
print("\nNext step: use probe_label instead of greedy_label in process_training_data.py")
 