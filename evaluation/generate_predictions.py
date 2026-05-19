# generate_predictions.py
import os
import gc
import json
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from datasets import load_dataset

# ============================================================================
# CONFIGURATION
# ============================================================================

UNALIGNED_MODEL = "meta-llama/Llama-2-7b-chat-hf"
ALIGNED_MODEL   = "outputs/probe-absolute/final"

UNALIGNED_DIR = "evaluation/results/triviaqa_unaligned"
ALIGNED_DIR   = "evaluation/results/triviaqa_probe_absolute"

MAX_NEW_TOKENS = 128

UNALIGNED_PROMPT = (
    'Q: {question}\n'
    'A: '
)

ALIGNED_PROMPT = (
    'Answer the question. If you don\'t know the answer to the question, '
    'it is appropriate to say "I apologize, but I\'m not able to provide '
    'an answer to the question."\n'
    'Q: {question}\n'
    'A: '
)

SAMPLING_PARAMS = SamplingParams(
    temperature=0,
    max_tokens=MAX_NEW_TOKENS,
)

# ============================================================================
# HELPERS
# ============================================================================

def load_model(model_path):
    print(f"\nLoading model: {model_path}")
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        gpu_memory_utilization=0.90,
    )
    print("✓ Model loaded")
    return llm


def generate_responses(llm, instances, output_dir, prompt_template):
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "eval_predictions.jsonl")

    if os.path.exists(out_file):
        print(f"  Already exists: {out_file}, skipping.")
        return

    print(f"  Generating {len(instances)} responses...")
    prompts = [prompt_template.format(question=inst["question"]) for inst in instances]
    outputs = llm.generate(prompts, SAMPLING_PARAMS)

    with open(out_file, "w") as fout:
        for inst, output in tqdm(zip(instances, outputs), total=len(instances)):
            pred_text = output.outputs[0].text.strip()
            record = {
                "question":    inst["question"],
                "question_id": inst["question_id"],
                "answers":     inst["answers"],
                "gold_answer": inst["gold_answer"],
                "prompt":      prompt_template.format(question=inst["question"]),
                "pred_text":   pred_text,
            }
            fout.write(json.dumps(record) + "\n")

    print(f"  ✓ Saved to {out_file}")


# ============================================================================
# LOAD EVAL DATA
# ============================================================================

def load_triviaqa_eval():
    print("\nLoading TriviaQA validation set...")
    dataset = load_dataset("trivia_qa", "rc.nocontext", split="validation")

    instances = []
    seen_questions = set()
    for item in dataset:
        question = item["question"]
        if question in seen_questions:
            continue
        seen_questions.add(question)
        gold = item["answer"]["value"]
        aliases = item["answer"]["aliases"]
        instances.append({
            "question":    question,
            "question_id": item["question_id"],
            "answers":     list(set([gold] + aliases)),
            "gold_answer": gold,
        })

    print(f"✓ Loaded {len(instances)} eval questions after deduplication")
    return instances


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    instances = load_triviaqa_eval()

    # --- Unaligned model --- skip if already exists
    unaligned_out = os.path.join(UNALIGNED_DIR, "eval_predictions.jsonl")
    if os.path.exists(unaligned_out):
        print(f"\nUnaligned predictions already exist: {unaligned_out}, skipping.")
    else:
        llm = load_model(UNALIGNED_MODEL)
        generate_responses(llm, instances, UNALIGNED_DIR, UNALIGNED_PROMPT)
        del llm
        gc.collect()
        torch.cuda.empty_cache()

    # --- Probe Confidence-Verb model ---
    llm = load_model(ALIGNED_MODEL)
    generate_responses(llm, instances, ALIGNED_DIR, ALIGNED_PROMPT)
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    print("\n" + "="*60)
    print("GENERATION COMPLETE")
    print("="*60)
    print(f"Unaligned predictions:             {UNALIGNED_DIR}/eval_predictions.jsonl")
    print(f"Probe Confidence-Verb predictions: {ALIGNED_DIR}/eval_predictions.jsonl")
    print("\nNext step: python evaluation/run_has_match.py")