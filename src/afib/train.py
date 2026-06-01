# train.py
import os
os.environ['PYTHONUNBUFFERED'] = '1'

import matplotlib
matplotlib.use('MacOSX')

from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, callbacks
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import joblib
import matplotlib.pyplot as plt

ROOT        = Path(__file__).resolve().parents[2]
DATA_DIR    = ROOT / 'data'    / 'afib'
MODELS_DIR  = ROOT / 'models'  / 'afib'
OUTPUTS_DIR = ROOT / 'outputs' / 'afib'

MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

data       = np.load(DATA_DIR / 'afib_dataset.npz')
X_feat     = data['X_feat']
X_seq      = data['X_seq']
y          = data['y']
record_ids = data['record_ids']

print(f"Dataset loaded : {len(y)} windows  ({len(np.unique(record_ids))} patients)")
print(f"Normal (0)     : {np.sum(y == 0)}")
print(f"AFib   (1)     : {np.sum(y == 1)}\n")

# Patient-wise split — no patient appears in both train and test
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_val_idx, test_idx = next(gss.split(X_feat, y, groups=record_ids))

X_feat_tv, X_seq_tv, y_tv = X_feat[train_val_idx], X_seq[train_val_idx], y[train_val_idx]
record_ids_tv = record_ids[train_val_idx]
X_f_te, X_s_te, y_te = X_feat[test_idx], X_seq[test_idx], y[test_idx]

gss_val = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
tr_idx, val_idx = next(gss_val.split(X_feat_tv, y_tv, groups=record_ids_tv))

X_f_tr,  X_s_tr,  y_tr  = X_feat_tv[tr_idx],  X_seq_tv[tr_idx],  y_tv[tr_idx]
X_f_val, X_s_val, y_val = X_feat_tv[val_idx], X_seq_tv[val_idx], y_tv[val_idx]

print(f"Train : {len(y_tr)} | Val : {len(y_val)} | Test : {len(y_te)}\n")

# ── Random Forest ─────────────────────────────────────────────────────────
print("=" * 50)
print("Training Random Forest...")
print("=" * 50)

scaler     = StandardScaler()
X_f_tr_sc  = scaler.fit_transform(X_f_tr)
X_f_val_sc = scaler.transform(X_f_val)
X_f_te_sc  = scaler.transform(X_f_te)

rf = RandomForestClassifier(
    n_estimators=200, max_depth=15, min_samples_split=5,
    class_weight='balanced', random_state=42, n_jobs=-1, verbose=1
)
rf.fit(X_f_tr_sc, y_tr)

rf_pred = rf.predict(X_f_te_sc)
print("\nRandom Forest Results:")
print(classification_report(y_te, rf_pred, target_names=['Normal', 'AFib']))

joblib.dump(rf,     MODELS_DIR / 'afib_rf_model.pkl')
joblib.dump(scaler, MODELS_DIR / 'afib_scaler.pkl')
print(f"Saved → {MODELS_DIR / 'afib_rf_model.pkl'}")
print(f"Saved → {MODELS_DIR / 'afib_scaler.pkl'}\n")

FEATURE_NAMES = [
    'Mean RR', 'SDNN', 'RMSSD', 'CV', 'pNN50',
    'Max RR', 'Min RR', 'Range RR',
    'Shannon Entropy', 'IEF', 'SD1', 'SD2', 'SD1/SD2', 'Window Length',
    'LF Power', 'HF Power', 'LF/HF Ratio', 'Dom. Frequency',
    'Sample Entropy', 'Approx. Entropy', 'DFA', 'Higuchi FD', 'Katz FD'
]
importances = rf.feature_importances_
sorted_idx  = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 4))
plt.bar(range(len(importances)), importances[sorted_idx], color='steelblue')
plt.xticks(range(len(importances)),
           [FEATURE_NAMES[i] for i in sorted_idx],
           rotation=35, ha='right', fontsize=9)
plt.title('Random Forest — Feature Importance for AFib Detection')
plt.tight_layout()
out = OUTPUTS_DIR / 'rf_feature_importance.png'
plt.savefig(out, dpi=150)
print(f"Saved → {out}")
plt.show()

# ── 1D-CNN ────────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("Training 1D-CNN...")
print("=" * 50)

rr_mean   = X_s_tr.mean()
rr_std    = X_s_tr.std()
X_s_tr_n  = (X_s_tr  - rr_mean) / rr_std
X_s_val_n = (X_s_val - rr_mean) / rr_std
X_s_te_n  = (X_s_te  - rr_mean) / rr_std

np.save(MODELS_DIR / 'afib_rr_stats.npy', np.array([rr_mean, rr_std]))

X_s_tr_n  = X_s_tr_n[...,  np.newaxis]
X_s_val_n = X_s_val_n[..., np.newaxis]
X_s_te_n  = X_s_te_n[...,  np.newaxis]

cw      = compute_class_weight('balanced', classes=np.unique(y_tr), y=y_tr)
cw_dict = dict(enumerate(cw))
print(f"Class weights: {cw_dict}\n")


def build_cnn(input_shape=(30, 1)):
    inp = tf.keras.Input(shape=input_shape)
    x = layers.Conv1D(32,  3, padding='same', activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(64,  3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(2, activation='softmax')(x)
    return tf.keras.Model(inp, out)


cnn = build_cnn()
cnn.summary()
cnn.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

cb = [
    callbacks.ModelCheckpoint(
        str(MODELS_DIR / 'best_afib_cnn.keras'),
        save_best_only=True, monitor='val_accuracy', verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, verbose=1
    ),
    callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True, verbose=1
    ),
]

history = cnn.fit(
    X_s_tr_n, y_tr,
    epochs=60,
    batch_size=64,
    validation_data=(X_s_val_n, y_val),
    class_weight=cw_dict,
    callbacks=cb
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history.history['accuracy'],     label='train')
ax1.plot(history.history['val_accuracy'], label='val')
ax1.set_title('1D-CNN Accuracy'); ax1.set_xlabel('Epoch'); ax1.legend()
ax2.plot(history.history['loss'],     label='train')
ax2.plot(history.history['val_loss'], label='val')
ax2.set_title('1D-CNN Loss'); ax2.set_xlabel('Epoch'); ax2.legend()
plt.tight_layout()
out = OUTPUTS_DIR / 'afib_cnn_training.png'
plt.savefig(out, dpi=150)
print(f"Saved → {out}")
plt.show()

cnn.save(MODELS_DIR / 'afib_cnn_model.keras')
print(f"\nSaved → {MODELS_DIR / 'afib_cnn_model.keras'}")
print(f"Saved → {MODELS_DIR / 'best_afib_cnn.keras'}")
print("\nTraining complete!")
