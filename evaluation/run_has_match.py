# run_has_match.py
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from triviaqa_nonambigqa_evaluation import compute_has_match, evaluate

# ============================================================================
# CONFIGURATION
# ============================================================================

UNALIGNED_DIR = "evaluation/results/triviaqa_unaligned"
ALIGNED_DIR = "evaluation/results/triviaqa_iti_absolute"
# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("COMPUTING HAS MATCH — SFT BASELINE")
    print("="*60)

    unaligned_raw  = os.path.join(UNALIGNED_DIR, "eval_predictions.jsonl")
    aligned_raw    = os.path.join(ALIGNED_DIR,   "eval_predictions.jsonl")
    unaligned_eval = os.path.join(UNALIGNED_DIR, "aligned_eval.jsonl")
    aligned_eval   = os.path.join(ALIGNED_DIR,   "aligned_eval.jsonl")

    print("\n--- Unaligned model ---")
    compute_has_match(unaligned_raw, unaligned_eval)

    print("\n--- SFT Baseline model ---")
    compute_has_match(aligned_raw, aligned_eval)

    print("\n" + "="*60)
    print("HAS MATCH COMPLETE")
    print("="*60)
    print(f"Unaligned eval:    {unaligned_eval}")
    print(f"SFT Baseline eval: {aligned_eval}")
    print("\nNext step: python evaluation/run_local_eval.py")