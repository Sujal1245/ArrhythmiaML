# train_model.py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

# ── Load data ────────────────────────────────────────────────────────────
data = np.load('mitbih_processed.npz')
X, y = data['X'], data['y']

X = X[..., np.newaxis]        # shape: (N, 280, 1) — CNN expects channels dim

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
)

print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

# ── Class weights (dataset is heavily imbalanced — N >> V >> others) ──
class_weights = compute_class_weight(
    'balanced', classes=np.unique(y_train), y=y_train
)
cw_dict = dict(enumerate(class_weights))
print("Class weights:", cw_dict)

# ── Model ────────────────────────────────────────────────────────────────
def build_model(input_shape=(280, 1), num_classes=5):
    inp = tf.keras.Input(shape=input_shape)

    x = layers.Conv1D(32, 5, padding='same', activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(64, 5, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    out = layers.Dense(num_classes, activation='softmax')(x)
    return tf.keras.Model(inp, out)

model = build_model()
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ── Callbacks ────────────────────────────────────────────────────────────
cb = [
    callbacks.ModelCheckpoint('best_model.keras', save_best_only=True,
                               monitor='val_accuracy', verbose=1),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                patience=5, verbose=1),
    callbacks.EarlyStopping(monitor='val_loss', patience=10,
                            restore_best_weights=True)
]

# ── Train ────────────────────────────────────────────────────────────────
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=128,
    validation_data=(X_val, y_val),
    class_weight=cw_dict,
    callbacks=cb
)

# ── Plot training curves ─────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history.history['accuracy'],     label='train')
ax1.plot(history.history['val_accuracy'], label='val')
ax1.set_title('Accuracy'); ax1.legend()
ax2.plot(history.history['loss'],     label='train')
ax2.plot(history.history['val_loss'], label='val')
ax2.set_title('Loss'); ax2.legend()
plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)
plt.show()

model.save('arrhythmia_model.keras')
print("Model saved → arrhythmia_model.keras")