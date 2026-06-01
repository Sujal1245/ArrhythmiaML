# evaluate.py
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

ROOT        = Path(__file__).resolve().parents[2]
DATA_DIR    = ROOT / 'data'    / 'arrhythmia'
MODELS_DIR  = ROOT / 'models'  / 'arrhythmia'
OUTPUTS_DIR = ROOT / 'outputs' / 'arrhythmia'

CLASS_NAMES = ['Normal (N)', 'Supraventricular (S)',
               'Ventricular (V)', 'Fusion (F)', 'Unknown (Q)']

data = np.load(DATA_DIR / 'mitbih_processed.npz')
X, y = data['X'][..., np.newaxis], data['y']

_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model  = tf.keras.models.load_model(MODELS_DIR / 'best_model.keras')
y_pred = np.argmax(model.predict(X_test, batch_size=256), axis=1)

print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix — MIT-BIH Test Set')
plt.tight_layout()
out = OUTPUTS_DIR / 'confusion_matrix.png'
plt.savefig(out, dpi=150)
print(f"Saved → {out}")
plt.show()
