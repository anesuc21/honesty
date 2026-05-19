# run_chatgpt_eval.py
import os
import sys
import json
from tqdm import tqdm
from openai import OpenAI
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from triviaqa_nonambigqa_chatgpt import process_data1, process_data2

# ============================================================================
# CONFIGURATION
# ============================================================================

OPENAI_API_KEY = "xxx"  # replace with your key
MODEL          = "gpt-3.5-turbo-0125"

UNALIGNED_DIR = "evaluation/results/triviaqa_unaligned"
ALIGNED_DIR   = "evaluation/results/triviaqa_confidence_num"

client = OpenAI(api_key=OPENAI_API_KEY)

# ============================================================================
# HELPERS
# ============================================================================

def call_chatgpt(message, retries=3):
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=message,
                temperature=0,
                max_tokens=50,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  API error (attempt {attempt+1}): {e}")
    return "no answer"


def run_chatgpt_eval(data_dir):
    print(f"\nRunning ChatGPT evaluation on {data_dir}")

    # Read the labelled file from run_has_match.py
    labelled_file = os.path.join(data_dir, "aligned_eval.jsonl")
    data = [json.loads(line) for line in open(labelled_file)]

    # Only run ChatGPT on wrong instances (has_match=0 and pred=wrong)
    needs_chatgpt = [
        inst for inst in data
        if inst.get('has_match', 0) == 0 and inst.get('pred') == 'wrong'
    ]
    print(f"  {len(needs_chatgpt)} instances need ChatGPT evaluation")

    if len(needs_chatgpt) == 0:
        print("  Nothing to evaluate, skipping.")
        return

    needs_ids = {inst['question_id'] for inst in needs_chatgpt}

    # ---- Step 1: Extract short answers ----
    print("\n  Step 1: Extracting short answers...")
    step1_data = process_data1(data_dir, "aligned_eval.jsonl")
    step1_data = [inst for inst in step1_data if inst['question_id'] in needs_ids]

    for inst in tqdm(step1_data, desc="Extracting answers"):
        pred = call_chatgpt(inst['chatgpt_message'])
        inst['chatgpt_pred'] = pred

    # Save step1 results to temp file so process_data2 can read chatgpt_pred
    temp_file = os.path.join(data_dir, "step1_temp.jsonl")
    with open(temp_file, "w") as f:
        for inst in step1_data:
            f.write(json.dumps(inst) + "\n")

    # ---- Step 2: Compare to gold answers ----
    print("\n  Step 2: Comparing to gold answers...")
    step2_data = process_data2(data_dir, "step1_temp.jsonl")

    for inst in tqdm(step2_data, desc="Scoring answers"):
        score = call_chatgpt(inst['chatgpt_message'])
        inst['chatgpt_score'] = score

    # ---- Save chatgpt_evaluation.jsonl ----
    out_file = os.path.join(data_dir, "chatgpt_evaluation.jsonl")
    with open(out_file, "w") as f:
        for inst in step2_data:
            f.write(json.dumps(inst) + "\n")

    # Clean up temp file
    os.remove(temp_file)
    print(f"  ✓ Saved to {out_file}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("CHATGPT EVALUATION")
    print("="*60)

    print("\n--- Unaligned model ---")
    run_chatgpt_eval(UNALIGNED_DIR)

    print("\n--- Aligned model ---")
    run_chatgpt_eval(ALIGNED_DIR)

    print("\n" + "="*60)
    print("DONE — next step: python evaluation/run_evaluate.py")
    print("="*60)