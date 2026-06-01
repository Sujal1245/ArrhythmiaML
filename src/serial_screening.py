# serial_screening.py
#
# Use Case: 2-Minute Screening Tool
# Input:   Raw ECG signal (CSV or NumPy)
# Output:  Final Diagnostic Report (Normal / Referral)
#
# This script bridges the gap between research data and real-world application.

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
import joblib
from scipy.signal import find_peaks

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.common.signal_processing import bandpass_filter, normalize_segment
from src.afib.build_dataset import extract_hrv_features

# ── Config ─────────────────────────────────────────────────────────────────
ARR_FS = 360
AFIB_WINDOW_SIZE = 30
AFIB_THRESHOLD = 0.846

# ── Load Models ────────────────────────────────────────────────────────────
MODELS_DIR = ROOT / 'models'

print("Initializing Screening Engine...")
try:
    arr_model   = tf.keras.models.load_model(MODELS_DIR / 'arrhythmia' / 'best_model.keras')
    afib_hybrid = tf.keras.models.load_model(MODELS_DIR / 'afib' / 'best_afib_hybrid.keras')
    afib_hsc    = joblib.load(MODELS_DIR / 'afib' / 'afib_hybrid_scaler.pkl')
    h_rr_stats  = np.load(MODELS_DIR / 'afib' / 'afib_hybrid_rr_stats.npy')
    print("✓ All models loaded successfully.\n")
except Exception as e:
    print(f"✗ Error loading models: {e}")
    sys.exit(1)

def run_screening(raw_signal, fs=360):
    """
    Processes a raw ECG signal and returns a diagnostic summary.
    """
    print(f"Processing {len(raw_signal)/fs:.1f}s of ECG data...")
    
    # 1. Preprocessing
    clean_ecg = bandpass_filter(raw_signal, fs=fs)
    
    # 2. R-Peak Detection (Simplified for serial use)
    # Using a simple peak finder on normalized signal
    norm_ecg = (clean_ecg - np.mean(clean_ecg)) / (np.std(clean_ecg) + 1e-8)
    peaks, _ = find_peaks(norm_ecg, distance=fs*0.4, height=1.0) # 0.4s min between beats (150bpm)
    
    if len(peaks) < 32:
        return {"error": "Insufficient heartbeats detected for a reliable screen. Please ensure the sensor is stable."}

    # 3. Arrhythmia Classification (Beat-by-Beat)
    arr_counts = [0] * 5
    ARR_NAMES  = ['Normal', 'Supraventricular', 'Ventricular (PVC)', 'Fusion', 'Unknown']
    
    for peak in peaks:
        start, end = peak - 100, peak + 180
        if start < 0 or end > len(clean_ecg): continue
        
        seg = normalize_segment(clean_ecg[start:end])
        inp = seg.reshape(1, 280, 1).astype(np.float32)
        pred = np.argmax(arr_model.predict(inp, verbose=0)[0])
        arr_counts[pred] += 1

    # 4. AFib Detection (Window-based)
    rr_ms = np.diff(peaks) / fs * 1000.0
    valid_rr = rr_ms[(rr_ms > 300) & (rr_ms < 2000)]
    
    afib_windows = 0
    afib_detections = 0
    
    # Slide 30-beat windows through the detected RR intervals
    step = 10
    for i in range(0, len(valid_rr) - AFIB_WINDOW_SIZE, step):
        window = valid_rr[i:i+AFIB_WINDOW_SIZE]
        
        feat    = extract_hrv_features(window)
        feat_sc = afib_hsc.transform(feat.reshape(1, -1))
        
        seq     = (window - h_rr_stats[0]) / h_rr_stats[1]
        seq     = seq.reshape(1, 30, 1).astype(np.float32)
        
        probs = afib_hybrid.predict([seq, feat_sc], verbose=0)[0]
        if probs[1] >= AFIB_THRESHOLD:
            afib_detections += 1
        afib_windows += 1

    # 5. Generate Verdict
    afib_burden = (afib_detections / max(1, afib_windows)) * 100
    pvc_count   = arr_counts[2]
    total_abnorm = sum(arr_counts[1:])
    
    verdict = "🟢 NORMAL"
    color   = "GREEN"
    advice  = "Heart rhythm appears regular. Continue regular wellness checks."
    
    if afib_burden > 10 or pvc_count > 5 or total_abnorm > 15:
        verdict = "🔴 IRREGULAR (REFERRAL RECOMMENDED)"
        color   = "RED"
        advice  = "Significant irregularities detected. Please share this report with a healthcare professional for a 12-lead ECG."
    elif total_abnorm > 5:
        verdict = "🟡 MINOR IRREGULARITIES"
        color   = "YELLOW"
        advice  = "Occasional irregular beats found. Re-test when resting, or consult if symptoms persist."

    return {
        "verdict": verdict,
        "color": color,
        "afib_burden": f"{afib_burden:.1f}%",
        "arrhythmia_summary": {name: count for name, count in zip(ARR_NAMES, arr_counts)},
        "advice": advice,
        "total_beats": len(peaks)
    }

def print_report(res):
    if "error" in res:
        print(f"\n[!] SCREENING FAILED: {res['error']}\n")
        return

    print("="*50)
    print("         CARDIAC SCREENING REPORT")
    print("="*50)
    print(f" FINAL VERDICT : {res['verdict']}")
    print(f" TOTAL BEATS   : {res['total_beats']}")
    print(f" AFib BURDEN   : {res['afib_burden']}")
    print("-" * 50)
    print(" ARRHYTHMIA BREAKDOWN:")
    for name, count in res['arrhythmia_summary'].items():
        if count > 0:
            print(f"  - {name:20s}: {count}")
    print("-" * 50)
    print(f" CLINICAL ADVICE:")
    print(f" {res['advice']}")
    print("="*50)
    print("\nDisclaimer: This is an AI screening tool, not a medical diagnosis.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a serial screening on an ECG signal.")
    parser.add_argument("--file", type=str, help="Path to a .npy file containing a raw ECG signal.")
    parser.add_argument("--demo", action="store_true", help="Run a demo using a sample from the dataset.")
    args = parser.parse_args()

    # If no arguments provided, default to demo mode for easier use in IDEs like PyCharm
    if not args.file and not args.demo:
        print("No arguments provided. Defaulting to --demo mode...\n")
        args.demo = True

    if args.demo:
        print("Running Demo Screening using sample data...")
        # Load a random sample from existing dataset
        data = np.load(ROOT / 'data' / 'arrhythmia' / 'mitbih_processed.npz')
        # We'll just stitch some beats to simulate a 'real' signal
        raw = data['X'][:100].flatten() 
        results = run_screening(raw)
        print_report(results)
    elif args.file:
        raw = np.load(args.file)
        results = run_screening(raw)
        print_report(results)
    else:
        parser.print_help()
