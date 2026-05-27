# visualize_alpha_sweeps.py
# Generate alpha sweep charts for all three ITI methods

import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "/workspace/honesty/figure/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# DATA
# ============================================================================

data = {
    'Absolute': {
        'alphas':    [0,     0.1,   0.2,   0.3,   0.4,   0.5,   1.0,   2.0,   3.0],
        'honesty':   [71.84, 73.43, 74.42, 75.74, 78.46, 75.62, 70.27, 56.05, 52.01],
        'prudence':  [55.20, 68.29, 72.22, 76.80, 85.48, 82.09, 90.54, 99.44, 99.47],
        'overcons':  [11.51, 21.43, 23.38, 25.32, 28.57, 30.84, 50.00, 87.34, 95.45],
        'accuracy':  [80.27, 66.80, 66.80, 67.00, 67.20, 64.80, 57.40, 42.80, 29.20],
        'best_alpha': 0.4,
        'baseline_honesty': 71.84,
    },
    'Confidence-Verb': {
        'alphas':    [0,     0.1,   0.2,   0.3,   0.4,   0.5],
        'honesty':   [68.17, 70.63, 70.64, 70.53, 68.69, 67.08],
        'prudence':  [69.94, 51.32, 53.93, 60.87, 62.38, 67.92],
        'overcons':  [33.60, 10.06, 12.66, 19.81, 25.00, 33.77],
        'accuracy':  [72.63, 82.40, 79.20, 78.20, 76.80, 74.80],
        'best_alpha': 0.2,
        'baseline_honesty': 68.17,
    },
    'Confidence-Num': {
        'alphas':    [0,     0.1,   0.2,   0.3,   0.4,   0.5,   0.6,   0.7],
        'honesty':   [67.23, 62.86, 64.74, 64.88, 67.74, 68.41, 64.57, 64.18],
        'prudence':  [58.58, 40.00, 48.31, 54.44, 66.33, 73.83, 74.59, 85.19],
        'overcons':  [24.11, 14.29, 18.83, 24.68, 30.84, 37.01, 45.45, 56.82],
        'accuracy':  [69.79, 77.60, 72.60, 68.00, 63.60, 58.00, 50.40, 41.20],
        'best_alpha': 0.5,
        'baseline_honesty': 67.23,
    },
}

# ============================================================================
# PLOT INDIVIDUAL CHARTS
# ============================================================================

for method, d in data.items():
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(d['alphas'], d['honesty'],  'o-',  color='#2c3e50', linewidth=2.5, markersize=7, label='Honesty')
    ax.plot(d['alphas'], d['prudence'], 's--', color='#27ae60', linewidth=2,   markersize=6, label='Prudence')
    ax.plot(d['alphas'], d['overcons'], '^--', color='#e74c3c', linewidth=2,   markersize=6, label='Over-conservativeness')
    ax.plot(d['alphas'], d['accuracy'], 'D--', color='#8e44ad', linewidth=2,   markersize=6, label='Accuracy')

    ax.axvline(x=d['best_alpha'], color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.annotate(f"α={d['best_alpha']}\n(optimal)", 
                xy=(d['best_alpha'], max(d['honesty'])),
                xytext=(d['best_alpha'] + max(d['alphas'])*0.08, max(d['honesty']) - 3),
                fontsize=9, color='gray',
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.2))

    ax.axhline(y=d['baseline_honesty'], color='#2c3e50', linestyle=':', linewidth=1, alpha=0.4)
    ax.text(max(d['alphas'])*0.7, d['baseline_honesty'] + 1, 'Baseline honesty', 
            fontsize=8, color='#2c3e50', alpha=0.6)

    ax.set_xlabel('Steering Strength (α)', fontsize=12)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title(f'ITI Steering: {method} Model\n(TriviaQA, n=500)', fontsize=13)
    ax.legend(fontsize=10, loc='center right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(20, 105)

    plt.tight_layout()
    fname = method.lower().replace('-', '_').replace(' ', '_')
    plt.savefig(os.path.join(OUTPUT_DIR, f'iti_alpha_{fname}.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, f'iti_alpha_{fname}.png'), dpi=300, bbox_inches='tight')
    print(f"  Saved iti_alpha_{fname}.png")
    plt.close()

# ============================================================================
# COMBINED HONESTY COMPARISON CHART
# ============================================================================

fig, ax = plt.subplots(figsize=(9, 6))

colors  = ['#2c3e50', '#27ae60', '#e74c3c']
markers = ['o', 's', '^']

for (method, d), color, marker in zip(data.items(), colors, markers):
    ax.plot(d['alphas'], d['honesty'], marker=marker, color=color,
            linewidth=2.5, markersize=7, label=method)
    ax.axhline(y=d['baseline_honesty'], color=color, linestyle=':', 
               linewidth=1, alpha=0.4)

ax.set_xlabel('Steering Strength (α)', fontsize=12)
ax.set_ylabel('Honesty Score (%)', fontsize=12)
ax.set_title('ITI Steering: Honesty Comparison Across Methods\n(TriviaQA, n=500)', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim(50, 85)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'iti_honesty_comparison.pdf'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, 'iti_honesty_comparison.png'), dpi=300, bbox_inches='tight')
print(f"  Saved iti_honesty_comparison.png")
plt.close()

print(f"\n✓ All figures saved to {OUTPUT_DIR}")