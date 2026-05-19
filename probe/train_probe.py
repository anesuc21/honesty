# probe/train_probe.py
# Step 2: Train probes comparing last token vs mean pooling
# Tries logistic regression and MLP on single and multi-layer representations

import os
import json
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# ============================================================================
# CONFIGURATION
# ============================================================================

HIDDEN_STATES_PATH = "/workspace/honesty/probe/hidden_states.npz"
PROBE_OUTPUT_DIR   = "/workspace/honesty/probe/"
PROBE_LAYERS       = [16, 20, 24, 28, 31]
TEST_SIZE          = 0.2
RANDOM_SEED        = 42

# ============================================================================
# LOAD
# ============================================================================

print(f"Loading hidden states: {HIDDEN_STATES_PATH}")
data   = np.load(HIDDEN_STATES_PATH)
labels = data['labels']
print(f"✓ Loaded {len(labels)} samples")
print(f"  Known: {labels.sum()}, Unknown: {(1-labels).sum()}")

# ============================================================================
# HELPERS
# ============================================================================

def split_and_scale(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, scaler

def train_logistic(X, y):
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)
    probe = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, C=1.0)
    probe.fit(X_train, y_train)
    acc   = accuracy_score(y_test, probe.predict(X_test))
    return acc, probe, scaler, y_test, probe.predict(X_test)

def train_mlp(X, y):
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)
    probe = MLPClassifier(
        hidden_layer_sizes=(512, 256, 64),
        activation='relu',
        max_iter=500,
        random_state=RANDOM_SEED,
        early_stopping=True,
        validation_fraction=0.1,
    )
    probe.fit(X_train, y_train)
    acc = accuracy_score(y_test, probe.predict(X_test))
    return acc, probe, scaler, y_test, probe.predict(X_test)

# ============================================================================
# COMPARE LAST TOKEN VS MEAN POOLING
# ============================================================================

all_results = {}

for pooling in ['last', 'mean']:
    print(f"\n{'='*60}")
    print(f"POOLING: {pooling.upper()} TOKEN")
    print(f"{'='*60}")

    # Single layer logistic
    for layer in PROBE_LAYERS:
        X   = data[f'layer_{layer}_{pooling}']
        acc, probe, scaler, y_test, y_pred = train_logistic(X, labels)
        key = f"logistic_{pooling}_layer{layer}"
        all_results[key] = {'accuracy': acc, 'probe': probe, 'scaler': scaler,
                            'layers': [layer], 'pooling': pooling, 'type': 'logistic'}
        print(f"  Logistic layer {layer:2d}: {acc:.4f}")

    # All layers concatenated logistic
    X   = np.concatenate([data[f'layer_{l}_{pooling}'] for l in PROBE_LAYERS], axis=1)
    acc, probe, scaler, y_test, y_pred = train_logistic(X, labels)
    key = f"logistic_{pooling}_all_layers"
    all_results[key] = {'accuracy': acc, 'probe': probe, 'scaler': scaler,
                        'layers': PROBE_LAYERS, 'pooling': pooling, 'type': 'logistic'}
    print(f"  Logistic all layers: {acc:.4f}")

    # MLP on best single layer
    best_layer = max(PROBE_LAYERS,
                     key=lambda l: all_results[f'logistic_{pooling}_layer{l}']['accuracy'])
    X   = data[f'layer_{best_layer}_{pooling}']
    acc, probe, scaler, y_test, y_pred = train_mlp(X, labels)
    key = f"mlp_{pooling}_layer{best_layer}"
    all_results[key] = {'accuracy': acc, 'probe': probe, 'scaler': scaler,
                        'layers': [best_layer], 'pooling': pooling, 'type': 'mlp'}
    print(f"  MLP layer {best_layer:2d}:      {acc:.4f}")

    # MLP on all layers
    X   = np.concatenate([data[f'layer_{l}_{pooling}'] for l in PROBE_LAYERS], axis=1)
    acc, probe, scaler, y_test, y_pred = train_mlp(X, labels)
    key = f"mlp_{pooling}_all_layers"
    all_results[key] = {'accuracy': acc, 'probe': probe, 'scaler': scaler,
                        'layers': PROBE_LAYERS, 'pooling': pooling, 'type': 'mlp'}
    print(f"  MLP all layers:      {acc:.4f}")

# ============================================================================
# SUMMARY AND BEST
# ============================================================================

print(f"\n{'='*60}")
print("SUMMARY — all approaches ranked")
print(f"{'='*60}")
for name, res in sorted(all_results.items(), key=lambda x: -x[1]['accuracy']):
    print(f"  {res['accuracy']:.4f}  {name}")

best_name = max(all_results, key=lambda k: all_results[k]['accuracy'])
best      = all_results[best_name]
print(f"\n✓ Best: {best_name} (accuracy={best['accuracy']:.4f})")

# Classification report
X = np.concatenate([data[f"layer_{l}_{best['pooling']}"] for l in best['layers']], axis=1)
_, X_test, _, y_test, scaler_best = split_and_scale(X, labels)
y_pred = best['probe'].predict(scaler_best.transform(X_test))
print(f"\nClassification report:")
print(classification_report(y_test, y_pred, target_names=['unknown', 'known']))

# ============================================================================
# SAVE
# ============================================================================

os.makedirs(PROBE_OUTPUT_DIR, exist_ok=True)
joblib.dump(best['probe'],  os.path.join(PROBE_OUTPUT_DIR, 'probe.pkl'))
joblib.dump(best['scaler'], os.path.join(PROBE_OUTPUT_DIR, 'scaler.pkl'))

config = {
    'best_name':     best_name,
    'best_accuracy': best['accuracy'],
    'best_layers':   best['layers'],
    'best_pooling':  best['pooling'],
    'probe_type':    best['type'],
    'all_results':   {k: v['accuracy'] for k, v in all_results.items()},
}
with open(os.path.join(PROBE_OUTPUT_DIR, 'probe_config.json'), 'w') as f:
    json.dump(config, f, indent=2)

print(f"\n✓ Saved to {PROBE_OUTPUT_DIR}")
print(f"  probe.pkl, scaler.pkl, probe_config.json")