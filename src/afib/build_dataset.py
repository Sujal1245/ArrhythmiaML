# build_dataset.py
import os
import sys
from pathlib import Path
from collections import Counter
import wfdb
import numpy as np
from scipy.stats import entropy
from scipy.signal import lombscargle
import antropy

# Add project root to path for imports
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.common.paths import AFIB_DATA_DIR, AFIB_DATASET
from src.common.config import AFIB_WINDOW_SIZE, AFIB_STEP_SIZE

os.environ['PYTHONUNBUFFERED'] = '1'

RAW_DATA_DIR = AFIB_DATA_DIR / 'raw'
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

RECORDS = [
    '04015', '04043', '04048', '04126', '04746', '04908', '04936',
    '05091', '05121', '05261', '06426', '06453', '06995', '07162',
    '07859', '07879', '07910', '08215', '08219', '08378', '08405',
    '08434', '08455'
]

# Set to None to process FULL 24-hour records
SAMPLE_LIMIT = None 


def extract_hrv_features(rr_intervals):
    rr   = np.array(rr_intervals, dtype=np.float64)
    diff = np.diff(rr)

    # Time-domain
    mean_rr   = np.mean(rr)
    std_rr    = np.std(rr)
    rmssd     = np.sqrt(np.mean(diff ** 2))
    cv        = std_rr / (mean_rr + 1e-8)
    pnn50     = np.sum(np.abs(diff) > 50) / len(diff) * 100
    max_rr    = np.max(rr)
    min_rr    = np.min(rr)
    range_rr  = max_rr - min_rr

    hist, _    = np.histogram(rr, bins=10, density=True)
    hist       = hist + 1e-10
    sh_entropy = entropy(hist)

    ief = np.sum(np.abs(diff) > 0.1 * mean_rr) / len(diff)
    sd1 = np.std(diff) / np.sqrt(2)
    sd2_sq = 2 * std_rr ** 2 - 0.5 * np.std(diff) ** 2
    sd2 = np.sqrt(max(0, sd2_sq) + 1e-8)
    sd_ratio = sd1 / (sd2 + 1e-8)

    # Frequency-domain via Lomb-Scargle (works on unevenly sampled RR series)
    # Cumulative time axis in seconds
    t_s = np.cumsum(rr) / 1000.0
    t_s -= t_s[0]
    # Angular frequencies for VLF (0.003–0.04 Hz), LF (0.04–0.15 Hz), HF (0.15–0.4 Hz)
    freqs  = np.linspace(0.003, 0.4, 300)
    omegas = 2 * np.pi * freqs
    rr_norm = rr - mean_rr      # zero-mean for lombscargle
    pgram   = lombscargle(t_s, rr_norm, omegas, normalize=True)

    lf_mask = (freqs >= 0.04)  & (freqs < 0.15)
    hf_mask = (freqs >= 0.15)  & (freqs < 0.4)
    lf_power   = np.trapezoid(pgram[lf_mask], freqs[lf_mask])
    hf_power   = np.trapezoid(pgram[hf_mask], freqs[hf_mask])
    lf_hf_ratio = lf_power / (hf_power + 1e-8)
    dom_freq    = freqs[np.argmax(pgram)]

    # Non-linear features (AntroPy)
    samp_en = antropy.sample_entropy(rr)
    app_en  = antropy.app_entropy(rr)
    dfa     = antropy.detrended_fluctuation(rr)
    hfd     = antropy.higuchi_fd(rr)
    kfd     = antropy.katz_fd(rr)

    return np.array([
        mean_rr, std_rr, rmssd, cv, pnn50,
        max_rr, min_rr, range_rr,
        sh_entropy, ief, sd1, sd2, sd_ratio,
        len(rr),
        lf_power, hf_power, lf_hf_ratio, dom_freq,
        samp_en, app_en, dfa, hfd, kfd
    ], dtype=np.float32)


if __name__ == "__main__":
    all_features  = []
    all_sequences = []
    all_labels    = []
    all_record_ids = []

    print("Processing MIT-BIH AF Database...")
    if SAMPLE_LIMIT:
        print(f" (Limited to first {SAMPLE_LIMIT} samples per record)\n")
    else:
        print(" (Processing FULL 24-hour records)\n")

    for rec_id in RECORDS:
        print(f"Processing record {rec_id}...", end=' ', flush=True)
        
        # Check if files exist locally in RAW_DATA_DIR
        local_path = RAW_DATA_DIR / rec_id
        use_local  = local_path.with_suffix('.hea').exists()
        
        try:
            if use_local:
                rec = wfdb.rdrecord(str(local_path), sampto=SAMPLE_LIMIT)
                rhy = wfdb.rdann(str(local_path), 'atr', sampto=SAMPLE_LIMIT)
            else:
                rec = wfdb.rdrecord(rec_id, pn_dir='afdb', sampto=SAMPLE_LIMIT)
                rhy = wfdb.rdann(rec_id, 'atr',  pn_dir='afdb', sampto=SAMPLE_LIMIT)
        except Exception as e:
            print(f"SKIP ({e})")
            continue

        try:
            if use_local:
                ann = wfdb.rdann(str(local_path), 'qrsc', sampto=SAMPLE_LIMIT)
            else:
                ann = wfdb.rdann(rec_id, 'qrsc', pn_dir='afdb', sampto=SAMPLE_LIMIT)
        except Exception:
            try:
                if use_local:
                    ann = wfdb.rdann(str(local_path), 'qrs', sampto=SAMPLE_LIMIT)
                else:
                    ann = wfdb.rdann(rec_id, 'qrs', pn_dir='afdb', sampto=SAMPLE_LIMIT)
            except Exception as e:
                print(f"SKIP — no QRS annotation ({e})")
                continue

        fs       = rec.fs
        r_peaks  = ann.sample
        rhy_samp = rhy.sample
        rhy_syms = rhy.aux_note

        def get_rhythm_at(sample):
            label = 'N'
            for s, sym in zip(rhy_samp, rhy_syms):
                if s <= sample:
                    label = sym.strip().replace('(', '')
                else:
                    break
            return label

        if len(r_peaks) < 2:
            print("too few peaks, skip")
            continue

        rr_all = np.diff(r_peaks) / fs * 1000.0
        # The rhythm label should be checked at the time of the intervals.
        # We use the peak at the end of each interval as its timestamp.
        peak_times_all = r_peaks[1:]

        # Filter for physiological plausibility while maintaining alignment
        valid = (rr_all > 300) & (rr_all < 2000)
        rr_ms = rr_all[valid]
        peak_times = peak_times_all[valid]

        if len(rr_ms) < AFIB_WINDOW_SIZE:
            print("too short, skip")
            continue

        windows_this_record = 0
        for start in range(0, len(rr_ms) - AFIB_WINDOW_SIZE, AFIB_STEP_SIZE):
            window_rr = rr_ms[start:start + AFIB_WINDOW_SIZE]
            
            # Use the actual timestamp of the middle RR-interval in this window
            mid_time = peak_times[start + AFIB_WINDOW_SIZE // 2]
            rhythm = get_rhythm_at(mid_time)

            if 'AFIB' in rhythm:
                label = 1
            elif rhythm == 'N' or rhythm == '':
                label = 0
            else:
                continue

            features = extract_hrv_features(window_rr)
            all_features.append(features)
            all_sequences.append(window_rr.astype(np.float32))
            all_labels.append(label)
            all_record_ids.append(rec_id)
            windows_this_record += 1

        print(f"done — {windows_this_record} windows")

    if len(all_labels) == 0:
        print("\nERROR: No windows extracted. Check your internet connection.")
    else:
        X_feat     = np.array(all_features,   dtype=np.float32)
        X_seq      = np.array(all_sequences,  dtype=np.float32)
        y          = np.array(all_labels,     dtype=np.int32)
        record_ids = np.array(all_record_ids, dtype=str)

        # Replace any inf/NaN introduced by frequency-domain or entropy features
        bad = ~np.isfinite(X_feat)
        if bad.any():
            col_medians = np.nanmedian(np.where(np.isfinite(X_feat), X_feat, np.nan), axis=0)
            X_feat[bad] = np.take(col_medians, np.where(bad)[1])

        print(f"\n{'=' * 45}")
        print(f"Total windows  : {len(y)}")
        print(f"Normal (0)     : {np.sum(y == 0)}")
        print(f"AFib   (1)     : {np.sum(y == 1)}")
        print(f"Feature shape  : {X_feat.shape}")
        print(f"Sequence shape : {X_seq.shape}")
        print(f"{'=' * 45}")

        AFIB_DATA_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(AFIB_DATASET, X_feat=X_feat, X_seq=X_seq, y=y, record_ids=record_ids)
        print(f"Saved → {AFIB_DATASET}")

