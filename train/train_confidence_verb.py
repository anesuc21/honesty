# train_confidence_verb.py
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import json
from datetime import datetime
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_MODEL = "meta-llama/Llama-2-7b-chat-hf"
DATA_PATH  = "/workspace/honesty/data/triviaqa/confidence-verb_p3/train.pt"
OUTPUT_DIR = "outputs/probe-confidence-verb"

print("="*80)
print("TRAINING CONFIDENCE VERB (Probe-based K)")
print("="*80)
print(f"Base model: {BASE_MODEL}")
print(f"Data:       {DATA_PATH}")
print(f"Output:     {OUTPUT_DIR}")
print(f"GPUs:       {torch.cuda.device_count()}")

# ============================================================================
# LOAD MODEL AND TOKENIZER
# ============================================================================

print("\nLoading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# No device_map — torchrun handles multi-GPU placement
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
)
model.gradient_checkpointing_enable()

print("✓ Model loaded")

# ============================================================================
# LOAD AND PROCESS DATA
# ============================================================================

print("\nLoading dataset...")
data = torch.load(DATA_PATH, weights_only=False)
print(f"✓ Loaded {len(data)} samples")

if len(data) > 0:
    print(f"\nSample format:")
    print(f"  Input:  {data[0]['input'][:100]}...")
    print(f"  Output: {data[0]['output'][:100]}...")

def process_sample(entry):
    prompt = entry["input"].strip()
    answer = entry["output"].strip()
    full_text = f"{prompt}{answer}{tokenizer.eos_token}"

    full_enc = tokenizer(
        full_text,
        truncation=True,
        max_length=1024,
        padding="max_length",
        return_tensors="pt"
    )

    input_ids      = full_enc["input_ids"].squeeze()
    attention_mask = full_enc["attention_mask"].squeeze()
    labels         = input_ids.clone()

    labels = labels.clamp(min=-100)
    labels[labels >= tokenizer.vocab_size] = -100

    prompt_enc = tokenizer(
        prompt,
        add_special_tokens=True,
        truncation=True,
        return_tensors="pt"
    )
    prompt_len = (prompt_enc["input_ids"] != tokenizer.pad_token_id).sum().item()
    if prompt_len >= len(labels):
        prompt_len = len(labels) - 1

    labels[:prompt_len] = -100
    labels[attention_mask == 0] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

print("\nProcessing samples...")
train_dataset = Dataset.from_list([process_sample(d) for d in data])
print(f"✓ Processed {len(train_dataset)} samples")

# ============================================================================
# SETUP TRAINING
# ============================================================================

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    # Training settings (from paper)
    num_train_epochs=2,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=1e-6,
    weight_decay=0.1,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",

    # Optimization
    bf16=True,
    gradient_checkpointing=True,
    optim="adamw_torch",

    # No intermediate checkpoints — avoids network fs errors
    save_strategy="no",

    # Logging
    logging_steps=50,
    logging_first_step=True,
    report_to="none",
    dataloader_num_workers=4,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)

# ============================================================================
# TRAIN
# ============================================================================

print("\n Starting training...")
print(f"   GPUs: {torch.cuda.device_count()}")
print(f"   Per-device batch size: 8")
print(f"   Effective batch size: {8 * torch.cuda.device_count()}")
print(f"   Steps per epoch: {len(train_dataset) // (8 * max(torch.cuda.device_count(), 1))}")
print(f"   Total steps: {(len(train_dataset) // (8 * max(torch.cuda.device_count(), 1))) * 2}")

trainer.train()

# ============================================================================
# SAVE
# ============================================================================

final_path = f"{OUTPUT_DIR}/final"
os.makedirs(final_path, exist_ok=True)
print(f"\n Saving final model to {final_path}")
trainer.save_model(final_path)
tokenizer.save_pretrained(final_path)

info = {
    "method": "confidence-verb-probe",
    "base_model": BASE_MODEL,
    "data_path": DATA_PATH,
    "num_samples": len(train_dataset),
    "epochs": 2,
    "effective_batch_size": 8 * torch.cuda.device_count(),
    "learning_rate": 1e-6,
    "knowledge_detection": "probe-based",
    "gpus": torch.cuda.device_count(),
    "timestamp": datetime.now().isoformat(),
}

with open(os.path.join(final_path, "training_info.json"), "w") as f:
    json.dump(info, f, indent=2)

print(f"\n Done! Saved to {final_path}")
print("\n" + "="*80)
print("TRAINING COMPLETE")
print("="*80)