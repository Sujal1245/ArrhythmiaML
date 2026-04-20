# explore_mitbih.py
import wfdb
import matplotlib.pyplot as plt
import numpy as np

# Download a single record to test (record 100 is a good starting point)
record = wfdb.rdrecord('100', pn_dir='mitdb')
annotation = wfdb.rdann('100', 'atr', pn_dir='mitdb')

print("Signal shape  :", record.p_signal.shape)   # (650000, 2) → 2 leads
print("Sampling rate :", record.fs)                # 360 Hz
print("Beat symbols  :", set(annotation.symbol))  # e.g. N, V, A, R ...

# Plot first 5 seconds of Lead I
fs = record.fs
ecg = record.p_signal[:, 0]   # Lead I (MLII)

plt.figure(figsize=(14, 3))
plt.plot(np.arange(5 * fs) / fs, ecg[:5 * fs], linewidth=0.8, color='steelblue')
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (mV)")
plt.title("MIT-BIH Record 100 — Lead MLII (first 5 sec)")
plt.tight_layout()
plt.savefig("sample_ecg.png", dpi=150)
plt.show()