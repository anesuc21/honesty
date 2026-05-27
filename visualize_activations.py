# visualize_activations.py
# Generate visualizations of ITI head activations for paper

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import json
import os

PROBE_DIR  = "/workspace/honesty/probe/"
OUTPUT_DIR = "/workspace/honesty/figure/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# LOAD DATA
# ============================================================================

print("Loading activations...")
# Load head rankings to find best head
with open(os.path.join(PROBE_DIR, 'iti_head_rankings.json')) as f:
    rankings = json.load(f)

# Sort to find best heads
sorted_heads = sorted(rankings.items(), key=lambda x: -x[1])
best_head_key = sorted_heads[0][0]  # e.g. "15_28"
best_layer, best_head = map(int, best_head_key.split('_'))
print(f"Best head: layer {best_layer}, head {best_head} (acc={sorted_heads[0][1]:.4f})")

# Load stored activations
idk_arrays     = np.load(os.path.join(PROBE_DIR, 'iti_idk_arrays.npy'), allow_pickle=True).item()
correct_arrays = np.load(os.path.join(PROBE_DIR, 'iti_correct_arrays.npy'), allow_pickle=True).item()

HEAD_DIM = 128
NUM_HEADS = 32

def get_head_vecs(arrays, layer, head):
    """Extract per-head vectors from stored arrays."""
    key = (layer, head)
    if key in arrays:
        return arrays[key]
    return None

# ============================================================================
# PLOT 1 — PCA of best head activations
# ============================================================================

print("\nPlot 1: PCA of best head activations...")

idk_vecs     = get_head_vecs(idk_arrays, best_layer, best_head)
correct_vecs = get_head_vecs(correct_arrays, best_layer, best_head)

if idk_vecs is not None and correct_vecs is not None:
    n = min(len(idk_vecs), len(correct_vecs), 500)
    X = np.vstack([idk_vecs[:n], correct_vecs[:n]])
    y = np.array([0]*n + [1]*n)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X_pca[y==0, 0], X_pca[y==0, 1], alpha=0.4, s=20,
               color='#e74c3c', label='IDK responses')
    ax.scatter(X_pca[y==1, 0], X_pca[y==1, 1], alpha=0.4, s=20,
               color='#2ecc71', label='Correct responses')

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=12)
    ax.set_title(f'PCA of Attention Head Activations\n(Layer {best_layer}, Head {best_head} — Top Truth-Sensitive Head)', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'pca_best_head_confverb.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'pca_best_head_confverb.png'), dpi=300, bbox_inches='tight')
    print(f"  Saved pca_best_head.png")
    plt.close()

# ============================================================================
# PLOT 2 — Head probe accuracy heatmap
# ============================================================================

print("\nPlot 2: Head accuracy heatmap...")

NUM_LAYERS = 32
acc_matrix = np.zeros((NUM_LAYERS, NUM_HEADS))

for key, acc in rankings.items():
    layer, head = map(int, key.split('_'))
    acc_matrix[layer, head] = acc

fig, ax = plt.subplots(figsize=(16, 8))
im = ax.imshow(acc_matrix, aspect='auto', cmap='RdYlGn',
               vmin=0.45, vmax=0.85)

plt.colorbar(im, ax=ax, label='Probe Accuracy')
ax.set_xlabel('Attention Head Index', fontsize=12)
ax.set_ylabel('Layer Index', fontsize=12)
ax.set_title('Truth-Sensitivity of Attention Heads\n(Probe Accuracy: IDK vs Correct Responses)', fontsize=13)
ax.set_xticks(range(0, NUM_HEADS, 4))
ax.set_yticks(range(0, NUM_LAYERS, 4))

# Mark top 48 heads
top_48_keys = [k for k, _ in sorted_heads[:48]]
for key in top_48_keys:
    l, h = map(int, key.split('_'))
    ax.add_patch(plt.Rectangle((h-0.5, l-0.5), 1, 1,
                                fill=False, edgecolor='blue', linewidth=0.8))

# Mark best head
ax.add_patch(plt.Rectangle((best_head-0.5, best_layer-0.5), 1, 1,
                            fill=False, edgecolor='black', linewidth=2.5))
ax.text(best_head, best_layer, '★', ha='center', va='center',
        fontsize=8, color='black', fontweight='bold')

blue_patch = mpatches.Patch(edgecolor='blue', facecolor='none', label='Top 48 heads (steered)')
black_patch = mpatches.Patch(edgecolor='black', facecolor='none', label='Best head')
ax.legend(handles=[blue_patch, black_patch], fontsize=10, loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'head_accuracy_heatmap_confverb.pdf'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, 'head_accuracy_heatmap_confverb.png'), dpi=300, bbox_inches='tight')
print(f"  Saved head_accuracy_heatmap.png")
plt.close()

# ============================================================================
# PLOT 3 — Honesty vs Alpha curve
# ============================================================================

print("\nPlot 3: Honesty vs Alpha...")

alphas   = [0,    0.1,  0.2,  0.3,  0.4,  0.5,  1.0,  2.0,  3.0]
honesty  = [71.84, 73.43, 74.42, 75.74, 78.46, 75.62, 70.27, 56.05, 52.01]
prudence = [55.20, 68.29, 72.22, 76.80, 85.48, 82.09, 90.54, 99.44, 99.47]
overcons = [11.51, 21.43, 23.38, 25.32, 28.57, 30.84, 50.00, 87.34, 95.45]
accuracy = [80.27, 66.80, 66.80, 67.00, 67.20, 64.80, 57.40, 42.80, 29.20]

fig, ax = plt.subplots(figsize=(9, 6))

ax.plot(alphas, honesty,  'o-', color='#2c3e50', linewidth=2.5, markersize=7, label='Honesty')
ax.plot(alphas, prudence, 's--', color='#27ae60', linewidth=2, markersize=6, label='Prudence')
ax.plot(alphas, overcons, '^--', color='#e74c3c', linewidth=2, markersize=6, label='Over-conservativeness')
ax.plot(alphas, accuracy, 'D--', color='#8e44ad', linewidth=2, markersize=6, label='Accuracy')

# Mark best alpha
ax.axvline(x=0.4, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
ax.annotate('α=0.4\n(optimal)', xy=(0.4, 78.46), xytext=(0.7, 76),
            fontsize=10, color='gray',
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.2))

# Mark baseline
ax.axhline(y=71.84, color='#2c3e50', linestyle=':', linewidth=1, alpha=0.5)
ax.text(2.5, 72.5, 'Baseline honesty', fontsize=9, color='#2c3e50', alpha=0.7)

ax.set_xlabel('Steering Strength (α)', fontsize=12)
ax.set_ylabel('Score (%)', fontsize=12)
ax.set_title('ITI Steering: Effect of Alpha on Honesty Metrics\n(Probe-Absolute Model, TriviaQA, n=500)', fontsize=13)
ax.legend(fontsize=11, loc='center right')
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.1, 3.2)
ax.set_ylim(20, 105)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'iti_alpha_sweep.pdf'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, 'iti_alpha_sweep.png'), dpi=300, bbox_inches='tight')
print(f"  Saved iti_alpha_sweep.png")
plt.close()

# ============================================================================
# PLOT 4 — PCA with steering direction arrow
# ============================================================================

print("\nPlot 4: PCA with steering direction...")

if idk_vecs is not None and correct_vecs is not None:
    n = min(len(idk_vecs), len(correct_vecs), 500)
    X = np.vstack([idk_vecs[:n], correct_vecs[:n]])
    y = np.array([0]*n + [1]*n)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    idk_mean_2d     = X_pca[y==0].mean(axis=0)
    correct_mean_2d = X_pca[y==1].mean(axis=0)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X_pca[y==0, 0], X_pca[y==0, 1], alpha=0.3, s=15,
               color='#e74c3c', label='IDK responses')
    ax.scatter(X_pca[y==1, 0], X_pca[y==1, 1], alpha=0.3, s=15,
               color='#2ecc71', label='Correct responses')

    # Plot means
    ax.scatter(*idk_mean_2d,     s=150, color='#c0392b', zorder=5, marker='*')
    ax.scatter(*correct_mean_2d, s=150, color='#27ae60', zorder=5, marker='*')

    # Plot steering direction arrow
    ax.annotate('', xy=idk_mean_2d, xytext=correct_mean_2d,
                arrowprops=dict(arrowstyle='->', color='#2c3e50',
                               lw=2.5, mutation_scale=20))
    ax.text((idk_mean_2d[0]+correct_mean_2d[0])/2 + 0.3,
            (idk_mean_2d[1]+correct_mean_2d[1])/2,
            'Steering\ndirection', fontsize=10, color='#2c3e50',
            ha='left', va='center')

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=12)
    ax.set_title(f'ITI Steering Direction in Activation Space\n(Layer {best_layer}, Head {best_head})', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'pca_steering_direction.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'pca_steering_direction.png'), dpi=300, bbox_inches='tight')
    print(f"  Saved pca_steering_direction.png")
    plt.close()

print("\n✓ All figures saved to", OUTPUT_DIR)
print("  pca_best_head.png")
print("  head_accuracy_heatmap.png")
print("  iti_alpha_sweep.png")
print("  pca_steering_direction.png")