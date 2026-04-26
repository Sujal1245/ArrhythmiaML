# build_afib_dataset.py
import os
os.environ['PYTHONUNBUFFERED'] = '1'

import wfdb
import numpy as np
from scipy.stats import entropy
from collections import Counter

# ── Config ────────────────────────────────────────────────────────────────
RECORDS = [
    '04015', '04043', '04048', '04126', '04746', '04908', '04936',
    '05091', '05121', '05261', '06426', '06453', '06995', '07162',
    '07859', '07879', '07910', '08215', '08219', '08378', '08405',
    '08434', '08455'
]

SAMPLE_LIMIT = 450000    # 10 minutes at 250Hz
WINDOW_SIZE  = 30        # 30 RR intervals per window
STEP_SIZE    = 10        # sliding window step

# ── HRV Feature extraction ────────────────────────────────────────────────
def extract_hrv_features(rr_intervals):
    rr   = np.array(rr_intervals, dtype=np.float64)
    diff = np.diff(rr)

    mean_rr    = np.mean(rr)
    std_rr     = np.std(rr)
    rmssd      = np.sqrt(np.mean(diff ** 2))
    cv         = std_rr / (mean_rr + 1e-8)
    pnn50      = np.sum(np.abs(diff) > 50) / len(diff) * 100

    max_rr     = np.max(rr)
    min_rr     = np.min(rr)
    range_rr   = max_rr - min_rr

    hist, _    = np.histogram(rr, bins=10, density=True)
    hist       = hist + 1e-10
    sh_entropy = entropy(hist)

    ief  = np.sum(np.abs(diff) > 0.1 * mean_rr) / len(diff)
    sd1  = np.std(diff) / np.sqrt(2)
    sd2  = np.sqrt(2 * std_rr**2 - 0.5 * np.std(diff)**2 + 1e-8)

    return np.array([
        mean_rr, std_rr, rmssd, cv, pnn50,
        max_rr, min_rr, range_rr,
        sh_entropy, ief, sd1, sd2,
        len(rr)
    ], dtype=np.float32)

FEATURE_NAMES = [
    'Mean RR', 'SDNN', 'RMSSD', 'CV', 'pNN50',
    'Max RR', 'Min RR', 'Range RR',
    'Shannon Entropy', 'IEF', 'SD1', 'SD2',
    'Window Length'
]

# ── Main loop ─────────────────────────────────────────────────────────────
all_features  = []
all_sequences = []
all_labels    = []

print("Downloading and processing MIT-BIH AF Database...")
print("(First 10 minutes of each record only)\n")

for rec_id in RECORDS:
    print(f"Processing record {rec_id}...", end=' ', flush=True)

    # Load record and rhythm annotations
    try:
        rec = wfdb.rdrecord(rec_id, pn_dir='afdb', sampto=SAMPLE_LIMIT)
        rhy = wfdb.rdann(rec_id, 'atr',  pn_dir='afdb', sampto=SAMPLE_LIMIT)
    except Exception as e:
        print(f"SKIP ({e})")
        continue

    # Try qrsc first, fall back to qrs
    try:
        ann = wfdb.rdann(rec_id, 'qrsc', pn_dir='afdb', sampto=SAMPLE_LIMIT)
    except Exception:
        try:
            ann = wfdb.rdann(rec_id, 'qrs', pn_dir='afdb', sampto=SAMPLE_LIMIT)
        except Exception as e:
            print(f"SKIP — no QRS annotation ({e})")
            continue

    fs       = rec.fs
    r_peaks  = ann.sample
    rhy_samp = rhy.sample
    rhy_syms = rhy.aux_note

    # ── Rhythm label lookup ───────────────────────────────────────────────
    def get_rhythm_at(sample):
        label = 'N'
        for s, sym in zip(rhy_samp, rhy_syms):
            if s <= sample:
                label = sym.strip().replace('(', '')
            else:
                break
        return label

    # ── RR intervals in milliseconds ──────────────────────────────────────
    if len(r_peaks) < 2:
        print("too few peaks, skip")
        continue

    rr_ms = np.diff(r_peaks) / fs * 1000.0
    valid = (rr_ms > 300) & (rr_ms < 2000)
    rr_ms = rr_ms[valid]

    if len(rr_ms) < WINDOW_SIZE:
        print("too short, skip")
        continue

    # ── Sliding window extraction ─────────────────────────────────────────
    windows_this_record = 0
    for start in range(0, len(rr_ms) - WINDOW_SIZE, STEP_SIZE):
        window_rr = rr_ms[start:start + WINDOW_SIZE]

        mid_peak  = r_peaks[min(start + WINDOW_SIZE // 2, len(r_peaks) - 1)]
        rhythm    = get_rhythm_at(mid_peak)

        if 'AFIB' in rhythm:
            label = 1
        elif rhythm == 'N' or rhythm == '':
            label = 0
        else:
            continue    # skip flutter, junctional, etc.

        features = extract_hrv_features(window_rr)
        all_features.append(features)
        all_sequences.append(window_rr.astype(np.float32))
        all_labels.append(label)
        windows_this_record += 1

    print(f"done — {windows_this_record} windows")

# ── Save ──────────────────────────────────────────────────────────────────
if len(all_labels) == 0:
    print("\nERROR: No windows extracted. Check your internet connection.")
else:
    X_feat = np.array(all_features,  dtype=np.float32)
    X_seq  = np.array(all_sequences, dtype=np.float32)
    y      = np.array(all_labels,    dtype=np.int32)

    print(f"\n{'='*45}")
    print(f"Total windows  : {len(y)}")
    print(f"Normal (0)     : {np.sum(y == 0)}")
    print(f"AFib   (1)     : {np.sum(y == 1)}")
    print(f"Feature shape  : {X_feat.shape}")
    print(f"Sequence shape : {X_seq.shape}")
    print(f"{'='*45}")

    np.savez('afib_dataset.npz', X_feat=X_feat, X_seq=X_seq, y=y)
    print("Saved → afib_dataset.npz")