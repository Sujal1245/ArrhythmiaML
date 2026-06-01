# evaluate.py
import os
os.environ['PYTHONUNBUFFERED'] = '1'

import matplotlib
matplotlib.use('MacOSX')

from pathlib import Path
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_curve, auc, f1_score, precision_recall_curve)
from sklearn.model_selection import GroupShuffleSplit

ROOT        = Path(__file__).resolve().parents[2]
DATA_DIR    = ROOT / 'data'    / 'afib'
MODELS_DIR  = ROOT / 'models'  / 'afib'
OUTPUTS_DIR = ROOT / 'outputs' / 'afib'

CLASS_NAMES = ['Normal', 'AFib']

# ── Load Data ──────────────────────────────────────────────────────────────
data       = np.load(DATA_DIR / 'afib_dataset.npz')
X_feat     = data['X_feat']
X_seq      = data['X_seq']
y          = data['y']
record_ids = data['record_ids']

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
_, test_idx = next(gss.split(X_feat, y, groups=record_ids))

X_f_te = X_feat[test_idx]
X_s_te = X_seq[test_idx]
y_te   = y[test_idx]

print(f"Test set : {len(y_te)} windows")
print(f"Normal   : {np.sum(y_te == 0)}")
print(f"AFib     : {np.sum(y_te == 1)}\n")

# ── Helpers ───────────────────────────────────────────────────────────────────

def best_f1_threshold(y_true, probs):
    """Return the threshold that maximises AFib F1 on the given set."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, probs)
    f1s = np.where(
        (precisions + recalls) == 0, 0,
        2 * precisions * recalls / (precisions + recalls + 1e-8)
    )
    best_idx = np.argmax(f1s[:-1])   # last element has no matching threshold
    return thresholds[best_idx], f1s[best_idx]


def apply_threshold(probs, threshold):
    return (probs >= threshold).astype(int)


# ── Random Forest ─────────────────────────────────────────────────────────────
print("Evaluating Random Forest...")
rf       = joblib.load(MODELS_DIR / 'afib_rf_model.pkl')
scaler   = joblib.load(MODELS_DIR / 'afib_scaler.pkl')
rr_stats = np.load(MODELS_DIR / 'afib_rr_stats.npy')

X_f_te_sc = scaler.transform(X_f_te)
rf_probs  = rf.predict_proba(X_f_te_sc)[:, 1]

rf_thresh, _ = best_f1_threshold(y_te, rf_probs)
rf_pred      = apply_threshold(rf_probs, rf_thresh)
print(f"  Optimal threshold : {rf_thresh:.3f}")

# ── 1D-CNN ────────────────────────────────────────────────────────────────────
print("Evaluating 1D-CNN...")
cnn           = tf.keras.models.load_model(MODELS_DIR / 'best_afib_cnn.keras')
X_s_te_n      = (X_s_te - rr_stats[0]) / rr_stats[1]
X_s_te_n      = X_s_te_n[..., np.newaxis]
cnn_probs_all = cnn.predict(X_s_te_n, batch_size=64, verbose=0)
cnn_probs     = cnn_probs_all[:, 1]

cnn_thresh, _ = best_f1_threshold(y_te, cnn_probs)
cnn_pred      = apply_threshold(cnn_probs, cnn_thresh)
print(f"  Optimal threshold : {cnn_thresh:.3f}")

# ── Hybrid Model ─────────────────────────────────────────────────────────────
print("Evaluating Hybrid Model...")
try:
    hybrid       = tf.keras.models.load_model(MODELS_DIR / 'best_afib_hybrid.keras')
    h_scaler     = joblib.load(MODELS_DIR / 'afib_hybrid_scaler.pkl')
    h_rr_stats   = np.load(MODELS_DIR / 'afib_hybrid_rr_stats.npy')

    X_f_te_hsc = h_scaler.transform(X_f_te)
    X_s_te_hn  = (X_s_te - h_rr_stats[0]) / h_rr_stats[1]
    X_s_te_hn  = X_s_te_hn[..., np.newaxis]

    hybrid_probs_all = hybrid.predict([X_s_te_hn, X_f_te_hsc], batch_size=64, verbose=0)
    hybrid_probs     = hybrid_probs_all[:, 1]

    hybrid_thresh, _ = best_f1_threshold(y_te, hybrid_probs)
    hybrid_pred      = apply_threshold(hybrid_probs, hybrid_thresh)
    print(f"  Optimal threshold : {hybrid_thresh:.3f}\n")
    HAS_HYBRID = True
except Exception as e:
    print(f"  SKIP Hybrid Model: {e}\n")
    HAS_HYBRID = False

# ── Reports ───────────────────────────────────────────────────────────────────
print("── Random Forest (tuned threshold) ────────────")
print(classification_report(y_te, rf_pred, target_names=CLASS_NAMES))
print("── 1D-CNN (tuned threshold) ────────────────────")
print(classification_report(y_te, cnn_pred, target_names=CLASS_NAMES))
if HAS_HYBRID:
    print("── Hybrid Model (tuned threshold) ──────────────")
    print(classification_report(y_te, hybrid_pred, target_names=CLASS_NAMES))

fpr_rf,  tpr_rf,  _ = roc_curve(y_te, rf_probs)
fpr_cnn, tpr_cnn, _ = roc_curve(y_te, cnn_probs)
auc_rf  = auc(fpr_rf,  tpr_rf)
auc_cnn = auc(fpr_cnn, tpr_cnn)

if HAS_HYBRID:
    fpr_h, tpr_h, _ = roc_curve(y_te, hybrid_probs)
    auc_h = auc(fpr_h, tpr_h)

# ── Plots ─────────────────────────────────────────────────────────────────────
rows = 3 if HAS_HYBRID else 2
fig, axes = plt.subplots(rows, 3, figsize=(18, 5 * rows))
fig.suptitle('AFib Detection — Model Evaluation (Tuned Thresholds)', fontsize=14)

# Confusion Matrices
cm_rf = confusion_matrix(y_te, rf_pred)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
axes[0, 0].set_title(f'Random Forest CM\n(thresh={rf_thresh:.3f})')
axes[0, 0].set_ylabel('True'); axes[0, 0].set_xlabel('Predicted')

cm_cnn = confusion_matrix(y_te, cnn_pred)
sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='Greens', ax=axes[0, 1],
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
axes[0, 1].set_title(f'1D-CNN CM\n(thresh={cnn_thresh:.3f})')
axes[0, 1].set_ylabel('True'); axes[0, 1].set_xlabel('Predicted')

if HAS_HYBRID:
    cm_h = confusion_matrix(y_te, hybrid_pred)
    sns.heatmap(cm_h, annot=True, fmt='d', cmap='Oranges', ax=axes[0, 2],
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    axes[0, 2].set_title(f'Hybrid Model CM\n(thresh={hybrid_thresh:.3f})')
    axes[0, 2].set_ylabel('True'); axes[0, 2].set_xlabel('Predicted')
else:
    axes[0, 2].axis('off')

# ROC & PR Comparison
axes[1, 0].plot(fpr_rf,  tpr_rf,  label=f'RF (AUC={auc_rf:.3f})', color='steelblue', alpha=0.8)
axes[1, 0].plot(fpr_cnn, tpr_cnn, label=f'CNN (AUC={auc_cnn:.3f})', color='green', alpha=0.8)
if HAS_HYBRID:
    axes[1, 0].plot(fpr_h, tpr_h, label=f'Hybrid (AUC={auc_h:.3f})', color='darkorange', linewidth=2)
axes[1, 0].plot([0, 1], [0, 1], 'k--', alpha=0.3)
axes[1, 0].set_title('ROC Curve Comparison')
axes[1, 0].set_xlabel('FPR'); axes[1, 0].set_ylabel('TPR')
axes[1, 0].legend(fontsize=9); axes[1, 0].grid(alpha=0.3)

pr_p_rf,  pr_r_rf,  _ = precision_recall_curve(y_te, rf_probs)
pr_p_cnn, pr_r_cnn, _ = precision_recall_curve(y_te, cnn_probs)
axes[1, 1].plot(pr_r_rf,  pr_p_rf,  label='Random Forest', color='steelblue', alpha=0.8)
axes[1, 1].plot(pr_r_cnn, pr_p_cnn, label='1D-CNN', color='green', alpha=0.8)
if HAS_HYBRID:
    pr_p_h, pr_r_h, _ = precision_recall_curve(y_te, hybrid_probs)
    axes[1, 1].plot(pr_r_h, pr_p_h, label='Hybrid Model', color='darkorange', linewidth=2)
axes[1, 1].set_title('Precision-Recall Curve (AFib)')
axes[1, 1].set_xlabel('Recall'); axes[1, 1].set_ylabel('Precision')
axes[1, 1].legend(fontsize=9); axes[1, 1].grid(alpha=0.3)

# F1 Score Bars
m_rf  = classification_report(y_te, rf_pred,  output_dict=True)
m_cnn = classification_report(y_te, cnn_pred, output_dict=True)
labels = ['0', '1', 'weighted avg']
rf_f1s  = [m_rf[l]['f1-score']  for l in labels]
cnn_f1s = [m_cnn[l]['f1-score'] for l in labels]
x = np.arange(len(labels)); w = 0.25
axes[1, 2].bar(x - w, rf_f1s, w, label='RF', color='steelblue', alpha=0.7)
axes[1, 2].bar(x, cnn_f1s, w, label='CNN', color='green', alpha=0.7)
if HAS_HYBRID:
    m_h = classification_report(y_te, hybrid_pred, output_dict=True)
    h_f1s = [m_h[l]['f1-score'] for l in labels]
    axes[1, 2].bar(x + w, h_f1s, w, label='Hybrid', color='darkorange')

axes[1, 2].set_xticks(x)
axes[1, 2].set_xticklabels(['Normal', 'AFib', 'Weighted'], fontsize=9)
axes[1, 2].set_title('F1 Score Comparison')
axes[1, 2].set_ylim(0, 1.1); axes[1, 2].legend(fontsize=8); axes[1, 2].grid(axis='y', alpha=0.3)

# F1 Sweep
if HAS_HYBRID:
    ts = np.linspace(0.01, 0.99, 100)
    rf_sw = [f1_score(y_te, apply_threshold(rf_probs, t), pos_label=1, zero_division=0) for t in ts]
    cnn_sw = [f1_score(y_te, apply_threshold(cnn_probs, t), pos_label=1, zero_division=0) for t in ts]
    h_sw = [f1_score(y_te, apply_threshold(hybrid_probs, t), pos_label=1, zero_division=0) for t in ts]
    
    axes[2, 0].plot(ts, rf_sw, color='steelblue', alpha=0.6, label='RF')
    axes[2, 0].plot(ts, cnn_sw, color='green', alpha=0.6, label='CNN')
    axes[2, 0].plot(ts, h_sw, color='darkorange', linewidth=2, label='Hybrid')
    axes[2, 0].axvline(hybrid_thresh, color='darkorange', linestyle='--', alpha=0.5)
    axes[2, 0].set_title('AFib F1 vs Threshold Sweep')
    axes[2, 0].set_xlabel('Threshold'); axes[2, 0].set_ylabel('F1 Score')
    axes[2, 0].legend(fontsize=9); axes[2, 0].grid(alpha=0.3)
    
    # Hide unused subplots in 3rd row
    axes[2, 1].axis('off')
    axes[2, 2].axis('off')

plt.tight_layout()
out = OUTPUTS_DIR / 'afib_evaluation.png'
plt.savefig(out, dpi=150)
print(f"Saved → {out}")
plt.show()
