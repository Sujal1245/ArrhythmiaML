# build_dataset.py
import sys
from pathlib import Path
from collections import Counter
import wfdb
import numpy as np

# Add project root to path for imports
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.common.paths import ARR_DATA_DIR, ARR_DATASET
from src.common.config import BEAT_MAP, ARR_FS, ARR_WINDOW, ARR_BEFORE, ARR_AFTER
from src.common.signal_processing import bandpass_filter, normalize_segment

RECORDS = [
    '100','101','102','103','104','105','106','107','108','109',
    '111','112','113','114','115','116','117','118','119','121',
    '122','123','124','200','201','202','203','205','207','208',
    '209','210','212','213','214','215','217','219','220','221',
    '222','223','228','230','231','232','233','234'
]

all_beats, all_labels, all_record_ids = [], [], []

for rec_id in RECORDS:
    print(f"Processing record {rec_id}...", end=' ', flush=True)
    try:
        rec = wfdb.rdrecord(rec_id, pn_dir='mitdb')
        ann = wfdb.rdann(rec_id, 'atr', pn_dir='mitdb')
    except Exception as e:
        print(f"SKIP ({e})")
        continue

    ecg     = bandpass_filter(rec.p_signal[:, 0], fs=ARR_FS)
    r_peaks = ann.sample
    symbols = ann.symbol

    for peak, sym in zip(r_peaks, symbols):
        if sym not in BEAT_MAP:
            continue
        start = peak - ARR_BEFORE
        end   = peak + ARR_AFTER
        if start < 0 or end > len(ecg):
            continue

        segment = ecg[start:end]
        segment = normalize_segment(segment)
        all_beats.append(segment)
        all_labels.append(BEAT_MAP[sym])
        all_record_ids.append(rec_id)

    print(f"done ({len(all_beats)} total so far)")

X          = np.array(all_beats,      dtype=np.float32)
y          = np.array(all_labels,     dtype=np.int32)
record_ids = np.array(all_record_ids, dtype=str)

print(f"\nDataset shape      : {X.shape}")
print(f"Class distribution : {Counter(y)}")

ARR_DATA_DIR.mkdir(parents=True, exist_ok=True)
np.savez(ARR_DATASET, X=X, y=y, record_ids=record_ids)
print(f"Saved → {ARR_DATASET}")
