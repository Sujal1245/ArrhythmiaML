# explore.py
from pathlib import Path
import wfdb
import matplotlib.pyplot as plt
import numpy as np

OUTPUTS_DIR = Path(__file__).resolve().parents[2] / 'outputs' / 'arrhythmia'
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

record     = wfdb.rdrecord('100', pn_dir='mitdb')
annotation = wfdb.rdann('100', 'atr', pn_dir='mitdb')

print("Signal shape  :", record.p_signal.shape)
print("Sampling rate :", record.fs)
print("Beat symbols  :", set(annotation.symbol))

fs  = record.fs
ecg = record.p_signal[:, 0]

plt.figure(figsize=(14, 3))
plt.plot(np.arange(5 * fs) / fs, ecg[:5 * fs], linewidth=0.8, color='steelblue')
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (mV)")
plt.title("MIT-BIH Record 100 — Lead MLII (first 5 sec)")
plt.tight_layout()
out = OUTPUTS_DIR / 'sample_ecg.png'
plt.savefig(out, dpi=150)
print(f"Saved → {out}")
plt.show()
