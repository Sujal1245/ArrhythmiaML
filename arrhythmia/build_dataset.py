# build_dataset.py
import wfdb
import numpy as np
from scipy.signal import butter, filtfilt
from collections import Counter

# ── Config ──────────────────────────────────────────────────────────────
RECORDS = [
    '100','101','102','103','104','105','106','107','108','109',
    '111','112','113','114','115','116','117','118','119','121',
    '122','123','124','200','201','202','203','205','207','208',
    '209','210','212','213','214','215','217','219','220','221',
    '222','223','228','230','231','232','233','234'
]

# AAMI standard beat mapping
BEAT_MAP = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,   # Normal
    'A': 1, 'a': 1, 'J': 1, 'S': 1,             # Supraventricular ectopic
    'V': 2, 'E': 2,                              # Ventricular ectopic
    'F': 3,                                      # Fusion
    '/': 4, 'f': 4, 'Q': 4                       # Unknown / paced
}

FS         = 360      # sampling rate
WINDOW     = 280      # samples per beat segment (~0.78 sec)
BEFORE     = 100      # samples before R-peak
AFTER      = WINDOW - BEFORE

# ── Bandpass filter ──────────────────────────────────────────────────────
def bandpass(signal, fs=360, lo=0.5, hi=40.0):
    nyq = fs / 2.0
    b, a = butter(4, [lo / nyq, hi / nyq], btype='band')
    return filtfilt(b, a, signal)

# ── Main loop ────────────────────────────────────────────────────────────
all_beats, all_labels = [], []

for rec_id in RECORDS:
    print(f"Processing record {rec_id}...", end=' ')
    try:
        rec  = wfdb.rdrecord(rec_id, pn_dir='mitdb')
        ann  = wfdb.rdann(rec_id, 'atr', pn_dir='mitdb')
    except Exception as e:
        print(f"SKIP ({e})")
        continue

    ecg = bandpass(rec.p_signal[:, 0])          # filter Lead I

    r_peaks  = ann.sample
    symbols  = ann.symbol

    for peak, sym in zip(r_peaks, symbols):
        if sym not in BEAT_MAP:
            continue                             # ignore non-beat annotations
        start = peak - BEFORE
        end   = peak + AFTER
        if start < 0 or end > len(ecg):
            continue                             # skip edge beats

        segment = ecg[start:end]
        # Z-score normalise each beat independently
        segment = (segment - segment.mean()) / (segment.std() + 1e-8)
        all_beats.append(segment)
        all_labels.append(BEAT_MAP[sym])

    print(f"done ({len(all_beats)} total so far)")

X = np.array(all_beats, dtype=np.float32)
y = np.array(all_labels, dtype=np.int32)

print(f"\nDataset shape : {X.shape}")
print(f"Class distribution: {Counter(y)}")

np.savez('mitbih_processed.npz', X=X, y=y)
print("Saved → mitbih_processed.npz")