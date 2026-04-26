# evaluate_afib.py
import os
os.environ['PYTHONUNBUFFERED'] = '1'

import matplotlib
matplotlib.use('MacOSX')

import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_curve, auc)
from sklearn.model_selection import train_test_split

CLASS_NAMES = ['Normal', 'AFib']

# ── Load data ─────────────────────────────────────────────────────────────
data   = np.load('afib_dataset.npz')
X_feat = data['X_feat']
X_seq  = data['X_seq']
y      = data['y']

_, X_f_te, _, X_s_te, _, y_te = train_test_split(
    X_feat, X_seq, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Test set : {len(y_te)} windows")
print(f"Normal   : {np.sum(y_te == 0)}")
print(f"AFib     : {np.sum(y_te == 1)}\n")

# ── Evaluate Random Forest ────────────────────────────────────────────────
print("Evaluating Random Forest...")
rf       = joblib.load('afib_rf_model.pkl')
scaler   = joblib.load('afib_scaler.pkl')
rr_stats = np.load('afib_rr_stats.npy')

X_f_te_sc = scaler.transform(X_f_te)
rf_pred   = rf.predict(X_f_te_sc)
rf_probs  = rf.predict_proba(X_f_te_sc)[:, 1]

# ── Evaluate 1D-CNN ───────────────────────────────────────────────────────
print("Evaluating 1D-CNN...")
cnn           = tf.keras.models.load_model('best_afib_cnn.keras')
X_s_te_n      = (X_s_te - rr_stats[0]) / rr_stats[1]
X_s_te_n      = X_s_te_n[..., np.newaxis]

cnn_probs_all = cnn.predict(X_s_te_n, batch_size=64, verbose=0)
cnn_probs     = cnn_probs_all[:, 1]
cnn_pred      = np.argmax(cnn_probs_all, axis=1)

# ── Print reports ─────────────────────────────────────────────────────────
print("\n── Random Forest ──────────────────────────────")
print(classification_report(y_te, rf_pred, target_names=CLASS_NAMES))

print("── 1D-CNN ─────────────────────────────────────")
print(classification_report(y_te, cnn_pred, target_names=CLASS_NAMES))

# ── AUC scores ────────────────────────────────────────────────────────────
fpr_rf,  tpr_rf,  _ = roc_curve(y_te, rf_probs)
fpr_cnn, tpr_cnn, _ = roc_curve(y_te, cnn_probs)
auc_rf  = auc(fpr_rf,  tpr_rf)
auc_cnn = auc(fpr_cnn, tpr_cnn)

print(f"AUC — Random Forest : {auc_rf:.4f}")
print(f"AUC — 1D-CNN        : {auc_cnn:.4f}\n")

# ── Plots ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle('AFib Detection — Model Evaluation', fontsize=13)

# 1. RF Confusion Matrix
cm_rf = confusion_matrix(y_te, rf_pred)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
axes[0, 0].set_title('Random Forest — Confusion Matrix')
axes[0, 0].set_ylabel('True')
axes[0, 0].set_xlabel('Predicted')

# 2. CNN Confusion Matrix
cm_cnn = confusion_matrix(y_te, cnn_pred)
sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='Greens', ax=axes[0, 1],
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
axes[0, 1].set_title('1D-CNN — Confusion Matrix')
axes[0, 1].set_ylabel('True')
axes[0, 1].set_xlabel('Predicted')

# 3. ROC Curves
axes[1, 0].plot(fpr_rf,  tpr_rf,
                label=f'Random Forest (AUC = {auc_rf:.3f})',
                color='steelblue', linewidth=2)
axes[1, 0].plot(fpr_cnn, tpr_cnn,
                label=f'1D-CNN        (AUC = {auc_cnn:.3f})',
                color='green', linewidth=2)
axes[1, 0].plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Random baseline')
axes[1, 0].set_xlabel('False Positive Rate')
axes[1, 0].set_ylabel('True Positive Rate')
axes[1, 0].set_title('ROC Curve Comparison')
axes[1, 0].legend(fontsize=9)
axes[1, 0].grid(alpha=0.3)

# 4. F1 Score Comparison
metrics_rf  = classification_report(y_te, rf_pred,
                                     target_names=CLASS_NAMES,
                                     output_dict=True)
metrics_cnn = classification_report(y_te, cnn_pred,
                                     target_names=CLASS_NAMES,
                                     output_dict=True)

labels  = CLASS_NAMES + ['weighted avg']    # ← fixed lowercase
rf_f1s  = [metrics_rf[l]['f1-score']  for l in labels]
cnn_f1s = [metrics_cnn[l]['f1-score'] for l in labels]

x = np.arange(len(labels))
w = 0.35
axes[1, 1].bar(x - w/2, rf_f1s,  w, label='Random Forest', color='steelblue')
axes[1, 1].bar(x + w/2, cnn_f1s, w, label='1D-CNN',        color='green')
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(['Normal', 'AFib', 'Weighted avg'], fontsize=9)
axes[1, 1].set_ylim(0, 1.1)
axes[1, 1].set_ylabel('F1 Score')
axes[1, 1].set_title('F1 Score Comparison')
axes[1, 1].legend()
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('afib_evaluation.png', dpi=150)
plt.show()

print("Saved → afib_evaluation.png")
