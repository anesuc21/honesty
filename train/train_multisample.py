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

BASE_MODEL = "meta-llama/Llama-2-7b-chat-hf"
DATA_PATH = "/workspace/honesty/data/triviaqa_official/multisample_p1/train.pt"
OUTPUT_DIR = "outputs/paper-reproduction-multisample"

print("="*80)
print("TRAINING MULTISAMPLE (80,000 samples - 10x data!)")
print("="*80)
print(f"  WARNING: This will take ~8-10 hours!")
print(f"Base model: {BASE_MODEL}")
print(f"Data: {DATA_PATH}")
print(f"Output: {OUTPUT_DIR}")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.gradient_checkpointing_enable()

print("\nLoading dataset...")
data = torch.load(DATA_PATH)
print(f"✓ Loaded {len(data)} samples (10x normal dataset)")

def process_sample(entry):
    prompt = entry["input"].strip()
    answer = entry["output"].strip()
    full_text = f"{prompt}{answer}{tokenizer.eos_token}"
    
    full_enc = tokenizer(full_text, truncation=True, max_length=1024, padding="max_length", return_tensors="pt")
    prompt_enc = tokenizer(prompt, add_special_tokens=True, truncation=True, return_tensors="pt")
    
    input_ids = full_enc["input_ids"].squeeze()
    attention_mask = full_enc["attention_mask"].squeeze()
    labels = input_ids.clone()
    
    prompt_len = (prompt_enc["input_ids"] != tokenizer.pad_token_id).sum().item()
    if prompt_len >= len(labels):
        prompt_len = len(labels) - 1
    labels[:prompt_len] = -100
    labels[attention_mask == 0] = -100
    
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

train_dataset = Dataset.from_list([process_sample(d) for d in data])

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,  # Only 1 epoch! (10x more data)
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    learning_rate=1e-6,
    weight_decay=0.1,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    bf16=True,
    gradient_checkpointing=True,
    optim="adamw_8bit",
    logging_steps=100,
    logging_first_step=True,
    save_steps=1000,
    save_total_limit=2,
    report_to="none",
    dataloader_num_workers=4,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8),
)

print("\n🚀 Starting training...")
print(f"   Steps: {len(train_dataset) // 8}")
print(f"   Estimated time: 8-10 hours")
print(f"   (Go get some sleep! )")

trainer.train()

final_path = f"{OUTPUT_DIR}/final"
trainer.save_model(final_path)
tokenizer.save_pretrained(final_path)

print(f" Done! Saved to {final_path}")
