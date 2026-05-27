import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from triviaqa_nonambigqa_evaluation import evaluate

UNALIGNED_DIR = "evaluation/results/triviaqa_unaligned"
ALIGNED_DIR   = "evaluation/results/triviaqa_iti_confidence_num"

if __name__ == "__main__":
    print("="*60)
    print("FINAL EVALUATION")
    print("="*60)
    reference_path = os.path.join(UNALIGNED_DIR, "aligned_eval.jsonl")
    print("\n--- Computing metrics for SFT BASELINE ---")
    evaluate(data_dir=ALIGNED_DIR, reference_path=reference_path)
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"\nMetrics saved to: {ALIGNED_DIR}/post_metrics.json")
