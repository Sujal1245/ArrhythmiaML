# train_hybrid.py
import os
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, callbacks, Input, Model
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import joblib
import matplotlib.pyplot as plt

# Add project root to path for imports
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.common.paths import AFIB_DATASET, AFIB_MODELS_DIR, AFIB_OUTPUTS_DIR

AFIB_MODELS_DIR.mkdir(parents=True, exist_ok=True)
AFIB_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Load Data ──────────────────────────────────────────────────────────────
print(f"Loading dataset: {AFIB_DATASET}...")
data = np.load(AFIB_DATASET)
X_feat = data['X_feat']
X_seq = data['X_seq']
y = data['y']
record_ids = data['record_ids']

print(f"Dataset loaded: {len(y)} windows ({len(np.unique(record_ids))} patients)")
print(f"Normal (0): {np.sum(y == 0)} | AFib (1): {np.sum(y == 1)}")

# ── Patient-wise Split ─────────────────────────────────────────────────────
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_val_idx, test_idx = next(gss.split(X_feat, y, groups=record_ids))

X_f_tv, X_s_tv, y_tv = X_feat[train_val_idx], X_seq[train_val_idx], y[train_val_idx]
record_ids_tv = record_ids[train_val_idx]
X_f_te, X_s_te, y_te = X_feat[test_idx], X_seq[test_idx], y[test_idx]

gss_val = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
tr_idx, val_idx = next(gss_val.split(X_f_tv, y_tv, groups=record_ids_tv))

X_f_tr, X_s_tr, y_tr = X_f_tv[tr_idx], X_s_tv[tr_idx], y_tv[tr_idx]
X_f_val, X_s_val, y_val = X_f_tv[val_idx], X_s_tv[val_idx], y_tv[val_idx]

print(f"Train: {len(y_tr)} | Val: {len(y_val)} | Test: {len(y_te)}\n")

# ── Preprocessing ──────────────────────────────────────────────────────────
# Scale handcrafted features
scaler = StandardScaler()
X_f_tr_sc = scaler.fit_transform(X_f_tr)
X_f_val_sc = scaler.transform(X_f_val)
X_f_te_sc = scaler.transform(X_f_te)

# Normalize RR sequences
rr_mean = X_s_tr.mean()
rr_std = X_s_tr.std()
X_s_tr_n = (X_s_tr - rr_mean) / rr_std
X_s_val_n = (X_s_val - rr_mean) / rr_std
X_s_te_n = (X_s_te - rr_mean) / rr_std

# Add channel dimension for Conv1D
X_s_tr_n = X_s_tr_n[..., np.newaxis]
X_s_val_n = X_s_val_n[..., np.newaxis]
X_s_te_n = X_s_te_n[..., np.newaxis]

# Class weights
cw = compute_class_weight('balanced', classes=np.unique(y_tr), y=y_tr)
cw_dict = dict(enumerate(cw))

# ── Build Hybrid Model ─────────────────────────────────────────────────────
def build_hybrid_model(seq_shape=(30, 1), feat_shape=(23,)):
    # 1. Sequence Branch (CNN)
    seq_inp = Input(shape=seq_shape, name='seq_input')
    x = layers.Conv1D(32, 3, padding='same', activation='relu')(seq_inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    
    # 2. Feature Branch (MLP)
    feat_inp = Input(shape=feat_shape, name='feat_input')
    y = layers.Dense(32, activation='relu')(feat_inp)
    y = layers.BatchNormalization()(y)
    y = layers.Dropout(0.2)(y)
    
    # 3. Merge Branches
    merged = layers.concatenate([x, y])
    z = layers.Dense(64, activation='relu')(merged)
    z = layers.Dropout(0.3)(z)
    z = layers.Dense(32, activation='relu')(z)
    out = layers.Dense(2, activation='softmax')(z)
    
    return Model(inputs=[seq_inp, feat_inp], outputs=out)

num_features = X_f_tr.shape[1]
model = build_hybrid_model(feat_shape=(num_features,))
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ── Training ───────────────────────────────────────────────────────────────
cb = [
    callbacks.ModelCheckpoint(
        str(AFIB_MODELS_DIR / 'best_afib_hybrid.keras'),
        save_best_only=True, monitor='val_accuracy', verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, verbose=1
    ),
    callbacks.EarlyStopping(
        monitor='val_loss', patience=12, restore_best_weights=True, verbose=1
    ),
]

print("\nStarting Hybrid Model training...")
history = model.fit(
    [X_s_tr_n, X_f_tr_sc], y_tr,
    epochs=100,
    batch_size=64,
    validation_data=([X_s_val_n, X_f_val_sc], y_val),
    class_weight=cw_dict,
    callbacks=cb,
    verbose=1
)

# ── Save Results ───────────────────────────────────────────────────────────
model.save(AFIB_MODELS_DIR / 'afib_hybrid_model.keras')
joblib.dump(scaler, AFIB_MODELS_DIR / 'afib_hybrid_scaler.pkl')
np.save(AFIB_MODELS_DIR / 'afib_hybrid_rr_stats.npy', np.array([rr_mean, rr_std]))

print(f"\nSaved hybrid model and artifacts to {AFIB_MODELS_DIR}")

# ── Evaluation ─────────────────────────────────────────────────────────────
preds = model.predict([X_s_te_n, X_f_te_sc])
y_pred = np.argmax(preds, axis=1)
print("\nHybrid Model Test Results:")
print(classification_report(y_te, y_pred, target_names=['Normal', 'AFib']))

# ── Plot Training Curves ───────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history.history['accuracy'], label='train')
ax1.plot(history.history['val_accuracy'], label='val')
ax1.set_title('Hybrid Model Accuracy')
ax1.set_xlabel('Epoch')
ax1.legend()

ax2.plot(history.history['loss'], label='train')
ax2.plot(history.history['val_loss'], label='val')
ax2.set_title('Hybrid Model Loss')
ax2.set_xlabel('Epoch')
ax2.legend()

plt.tight_layout()
out_plot = AFIB_OUTPUTS_DIR / 'afib_hybrid_training.png'
plt.savefig(out_plot, dpi=150)
print(f"Saved training curves to {out_plot}")
plt.show()
