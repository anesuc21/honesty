# train_multisample.py
import torch
import os
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
DATA_PATH = "/workspace/my_honesty/data/triviaqa/multisample_p1/train.pt"
OUTPUT_DIR = "outputs/paper-reproduction-multisample"

print("="*80)
print("TRAINING MULTISAMPLE")
print("="*80)
print(f"Base model: {BASE_MODEL}")
print(f"Data: {DATA_PATH}")
print(f"Output: {OUTPUT_DIR}")

# ============================================================================
# LOAD MODEL AND TOKENIZER
# ============================================================================

print("\nLoading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.gradient_checkpointing_enable()

print("✓ Model loaded")

# ============================================================================
# LOAD AND PROCESS DATA
# ============================================================================

print("\nLoading dataset...")
data = torch.load(DATA_PATH)
print(f"✓ Loaded {len(data)} samples")

if len(data) > 0:
    print(f"\nSample format:")
    print(f"  Input:  {data[0]['input'][:100]}...")
    print(f"  Output: {data[0]['output'][:100]}...")

def process_sample(entry):
    """Process a single training sample."""
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

    input_ids = full_enc["input_ids"].squeeze()
    attention_mask = full_enc["attention_mask"].squeeze()
    labels = input_ids.clone()

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

    # Training settings (from paper) — 1 epoch for multisample
    num_train_epochs=1,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    learning_rate=1e-6,
    weight_decay=0.1,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",

    # Optimization
    bf16=True,
    gradient_checkpointing=True,
    optim="adamw_torch",

    # Logging and saving
    logging_steps=50,
    logging_first_step=True,
    save_steps=500,
    save_total_limit=2,
    save_strategy="steps",

    # Other
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

checkpoint_path = None
if os.path.exists(OUTPUT_DIR):
    checkpoints = [
        os.path.join(OUTPUT_DIR, d)
        for d in os.listdir(OUTPUT_DIR)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(OUTPUT_DIR, d))
    ]
    if checkpoints:
        checkpoint_path = max(checkpoints, key=os.path.getctime)
        print(f"\n Found checkpoint: {checkpoint_path}")

print("\n Starting training...")
print(f"   Effective batch size: 8")
print(f"   Steps per epoch: {len(train_dataset) // 8}")
print(f"   Total steps: {len(train_dataset) // 8}")
print(f"   Estimated time: ~8 hours (80,000 samples)")

if checkpoint_path:
    print(f"\n   Resuming from: {checkpoint_path}")
    trainer.train(resume_from_checkpoint=checkpoint_path)
else:
    trainer.train()

# ============================================================================
# SAVE
# ============================================================================

final_path = f"{OUTPUT_DIR}/final"
print(f"\n Saving final model to {final_path}")
trainer.save_model(final_path)
tokenizer.save_pretrained(final_path)

import json
from datetime import datetime

info = {
    "method": "multisample",
    "base_model": BASE_MODEL,
    "data_path": DATA_PATH,
    "num_samples": len(train_dataset),
    "epochs": 1,
    "effective_batch_size": 8,
    "learning_rate": 1e-6,
    "timestamp": datetime.now().isoformat(),
}

with open(os.path.join(final_path, "training_info.json"), "w") as f:
    json.dump(info, f, indent=2)

print(f"\n Done! Saved to {final_path}")

print("\n" + "="*80)
print("TRAINING COMPLETE")
print("="*80)
print("\nNext steps:")
print("  1. Evaluate: python evaluation/generate_predictions.py")
print("  2. Metrics: python evaluation/run_evaluate.py")