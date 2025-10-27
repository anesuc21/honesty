# evaluate_all_nonambigqa_with_honesty.py
import os
import subprocess
import json

print("="*70)
print("NON-AMBIGQA EVALUATION WITH HONESTY METRICS")
print("="*70)

# Step 1: First evaluate baseline to create reference
print("\n[1/4] Evaluating baseline (unaligned reference)...")
baseline_pred_file = "/workspace/honesty/eval_results_table4/non-ambigqa/sft-baseline/predictions.jsonl"
baseline_output_dir = "/workspace/honesty/eval_results_table4/non-ambigqa/baseline"

if not os.path.exists(baseline_pred_file):
    print("❌ Baseline predictions not found!")
    print(f"   Expected: {baseline_pred_file}")
    print("\n   Run: python generate_all_ood_predictions.py first")
    exit(1)

# Evaluate baseline (no reference needed)
result = subprocess.run([
    'python', 
    'evaluate_nonambigqa_with_baseline.py', 
    'baseline'
], capture_output=False)

if result.returncode != 0:
    print("❌ Failed to evaluate baseline")
    exit(1)

# Step 2-4: Evaluate aligned models using baseline as reference
aligned_models = ['sft-baseline', 'confidence-verb', 'multisample']

for i, model in enumerate(aligned_models, start=2):
    print(f"\n[{i}/4] Evaluating {model} (with honesty metrics)...")
    
    result = subprocess.run([
        'python',
        'evaluate_nonambigqa_with_baseline.py',
        model
    ], capture_output=False)
    
    if result.returncode != 0:
        print(f"❌ Failed to evaluate {model}")

# Step 5: Print summary
print("\n" + "="*70)
print("COMPLETE RESULTS: NON-AMBIGQA")
print("="*70)

all_models = ['baseline', 'sft-baseline', 'confidence-verb', 'multisample']

for model in all_models:
    metrics_file = f"/workspace/honesty/eval_results_table4/non-ambigqa/{model}/post_metrics.json"
    
    if os.path.exists(metrics_file):
        with open(metrics_file) as f:
            metrics = json.load(f)
        
        print(f"\n{model.upper()}:")
        print(f"  Accuracy:          {metrics['answer_accuracy']*100:6.2f}%")
        print(f"  Correct:           {metrics['correct']*100:6.2f}%")
        print(f"  IDK:               {metrics['idk']*100:6.2f}%")
        print(f"  Wrong:             {metrics['wrong']*100:6.2f}%")
        
        if metrics.get('honesty', 0) > 0:
            print(f"  Over-Conservative: {metrics['over-consv']*100:6.2f}%")
            print(f"  Prudence:          {metrics['prudence']*100:6.2f}%")
            print(f"  Honesty:           {metrics['honesty']*100:6.2f}%")
    else:
        print(f"\n{model.upper()}: ❌ Metrics not found")

print("\n" + "="*70)

# Create comparison table
print("\nCOMPARISON TABLE (vs Unaligned Baseline)")
print("="*70)
print(f"{'Model':<20} {'Accuracy':>10} {'IDK':>10} {'Over-Consv':>12} {'Prudence':>10} {'Honesty':>10}")
print("-"*70)

for model in all_models:
    metrics_file = f"/workspace/honesty/eval_results_table4/non-ambigqa/{model}/post_metrics.json"
    
    if os.path.exists(metrics_file):
        with open(metrics_file) as f:
            metrics = json.load(f)
        
        acc = f"{metrics['answer_accuracy']*100:5.2f}%"
        idk = f"{metrics['idk']*100:5.2f}%"
        
        if metrics.get('honesty', 0) > 0:
            over = f"{metrics['over-consv']*100:5.2f}%"
            prud = f"{metrics['prudence']*100:5.2f}%"
            hon = f"{metrics['honesty']*100:5.2f}%"
        else:
            over = "N/A"
            prud = "N/A"
            hon = "N/A"
        
        print(f"{model:<20} {acc:>10} {idk:>10} {over:>12} {prud:>10} {hon:>10}")

print("="*70)
print("\n✅ COMPLETE NON-AMBIGQA EVALUATION FINISHED!")
print("="*70)