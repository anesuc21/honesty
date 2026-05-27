# run_local_eval.py
import os
import sys
import json
from tqdm import tqdm
from vllm import LLM, SamplingParams
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from triviaqa_nonambigqa_chatgpt import process_data1, process_data2

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL         = "mistralai/Mistral-7B-Instruct-v0.2"
UNALIGNED_DIR = "evaluation/results/triviaqa_unaligned"
ALIGNED_DIR   = "evaluation/results/triviaqa_iti_confidence_num"
# ============================================================================
# LOAD MODEL
# ============================================================================

print(f"\nLoading model: {MODEL}")
llm = LLM(
    model=MODEL,
    dtype="bfloat16",
    gpu_memory_utilization=0.90,
)
sampling_params = SamplingParams(
    temperature=0,
    max_tokens=50,
)
print("✓ Model loaded")

# ============================================================================
# HELPERS
# ============================================================================

def run_local_eval(data_dir):
    print(f"\nRunning local evaluation on {data_dir}")

    # Read the labelled file from run_has_match.py
    labelled_file = os.path.join(data_dir, "aligned_eval.jsonl")
    data = [json.loads(line) for line in open(labelled_file)]

    # Only run on wrong instances (has_match=0 and pred=wrong)
    needs_eval = [
        inst for inst in data
        if inst.get('has_match', 0) == 0 and inst.get('pred') == 'wrong'
    ]
    print(f"  {len(needs_eval)} instances need evaluation")

    if len(needs_eval) == 0:
        print("  Nothing to evaluate, skipping.")
        return

    needs_ids = {inst['question_id'] for inst in needs_eval}

    # ---- Step 1: Extract short answers (batched) ----
    print("\n  Step 1: Extracting short answers...")
    step1_data = process_data1(data_dir, "aligned_eval.jsonl")
    step1_data = [inst for inst in step1_data if inst['question_id'] in needs_ids]

    prompts1 = [
        f"[INST] {inst['chatgpt_message'][0]['content']} [/INST]"
        for inst in step1_data
    ]
    print(f"  Generating {len(prompts1)} responses...")
    outputs1 = llm.generate(prompts1, sampling_params)
    for inst, output in zip(step1_data, outputs1):
        inst['chatgpt_pred'] = output.outputs[0].text.strip()

    # Save step1 results to temp file
    temp_file = os.path.join(data_dir, "step1_temp.jsonl")
    with open(temp_file, "w") as f:
        for inst in step1_data:
            f.write(json.dumps(inst) + "\n")
    print(f"  ✓ Step 1 complete")

    # ---- Step 2: Compare to gold answers (batched) ----
    print("\n  Step 2: Comparing to gold answers...")
    step2_data = process_data2(data_dir, "step1_temp.jsonl")

    prompts2 = [
        f"[INST] {inst['chatgpt_message'][0]['content']} [/INST]"
        for inst in step2_data
    ]
    print(f"  Generating {len(prompts2)} responses...")
    outputs2 = llm.generate(prompts2, sampling_params)
    for inst, output in zip(step2_data, outputs2):
        inst['chatgpt_score'] = output.outputs[0].text.strip()

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
    print("LOCAL MODEL EVALUATION")
    print("="*60)

    print("\n--- Unaligned model ---")
    run_local_eval(UNALIGNED_DIR)

    print("\n--- Aligned model ---")
    run_local_eval(ALIGNED_DIR)

    print("\n" + "="*60)
    print("DONE — next step: python evaluation/run_evaluate.py")
    print("="*60)