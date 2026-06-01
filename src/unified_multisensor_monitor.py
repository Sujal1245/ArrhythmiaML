# unified_multisensor_monitor.py
#
# Unified Monitor for:
#   1. AD8232 (ECG) -> Arrhythmia Classification (CNN)
#   2. MAX30100 (PPG) -> AFib Detection (Hybrid Model)
#
# Optimized for macOS (MacOSX backend)

import os
import sys
import time
import collections
from pathlib import Path

import serial
import serial.tools.list_ports
import numpy as np

import matplotlib
matplotlib.use('MacOSX') 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button
from matplotlib.animation import FuncAnimation

import tensorflow as tf
import joblib

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.afib.build_dataset import extract_hrv_features

# ── Configuration ──────────────────────────────────────────────────────────
PORT    = '/dev/cu.usbserial-0001' # ← CHANGE THIS
BAUD    = 115200
FS_ECG  = 360
FS_PPG  = 100
AFIB_WIN_SIZE = 30
AFIB_THRESH   = 0.846

# Arrhythmia Model Constants
WIN_ARR = 280
BEFORE  = 100
AFTER   = WIN_ARR - BEFORE
CLASS_NAMES_ARR  = ['Normal (N)', 'Supraventricular (S)', 'Ventricular (V)', 'Fusion (F)', 'Unknown (Q)']
CLASS_COLORS_ARR = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']

# ── Load Models ────────────────────────────────────────────────────────────
MODELS_DIR = ROOT / 'models'
print("Loading Arrhythmia model...")
arr_model = tf.keras.models.load_model(MODELS_DIR / 'arrhythmia' / 'best_model.keras')

print("Loading AFib hybrid model...")
afib_hybrid = tf.keras.models.load_model(MODELS_DIR / 'afib' / 'best_afib_hybrid.keras')
afib_hsc    = joblib.load(MODELS_DIR / 'afib' / 'afib_hybrid_scaler.pkl')
h_rr_stats  = np.load(MODELS_DIR / 'afib' / 'afib_hybrid_rr_stats.npy')
print("✓ All models loaded.\n")

# ── Detectors ──────────────────────────────────────────────────────────────
class ECGPeakDetector:
    def __init__(self, fs=360):
        self.fs = fs
        self.min_rr = int(0.3 * fs)
        self.last_peak = 0
        self.threshold = None
        self.signal_level = 0.0
        self.noise_level = 0.0

    def detect(self, buf, cursor):
        if cursor < 30: return False
        indices = (np.arange(cursor - 10, cursor)) % len(buf)
        window = buf[indices]
        diff = np.diff(window)
        energy = np.sum(diff ** 2)
        if self.threshold is None:
            self.signal_level = energy
            self.threshold = 0.5 * energy
            return False
        if energy > self.threshold:
            self.signal_level = 0.125 * energy + 0.875 * self.signal_level
        else:
            self.noise_level = 0.125 * energy + 0.875 * self.noise_level
        self.threshold = self.noise_level + 0.25 * (self.signal_level - self.noise_level)
        if energy > self.threshold and (cursor - self.last_peak) > self.min_rr:
            self.last_peak = cursor
            return True
        return False

class PPGPeakDetector:
    def __init__(self, fs=100):
        self.fs = fs
        self.min_rr = int(0.4 * fs)
        self.last_peak = 0
        self.buffer = collections.deque(maxlen=fs)
        
    def detect(self, val, absolute_index):
        self.buffer.append(val)
        if len(self.buffer) < self.fs: return False
        if (absolute_index - self.last_peak) > self.min_rr:
            local_win = list(self.buffer)[-int(0.1*self.fs):]
            if val == max(local_win) and val > np.mean(self.buffer) * 1.02:
                self.last_peak = absolute_index
                return True
        return False

# ── State ──────────────────────────────────────────────────────────────────
ecg_buffer = np.zeros(FS_ECG * 10)
ecg_cursor = 0
ecg_last_processed = 0

ppg_buffer = np.zeros(FS_PPG * 10)
ppg_cursor = 0

current_temp = 36.5
temp_history = collections.deque([36.5]*60, maxlen=60)

rr_buffer = collections.deque(maxlen=AFIB_WIN_SIZE)
last_ppg_peak_time = time.time()
beat_count_ppg = 0

ecg_detector = ECGPeakDetector(FS_ECG)
ppg_detector = PPGPeakDetector(FS_PPG)

is_screening = False
screening_start_time = 0
SCREENING_DURATION = 60
screening_results = {'arr_counts': [0]*5, 'afib_windows': 0, 'afib_detections': 0, 'total_beats': 0, 'max_temp': 0.0}

# ── UI Layout ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 10), facecolor='#1a1a2e')
fig.suptitle('Unified Multi-Sensor Cardiac Monitor', color='white', fontsize=14, fontweight='bold')

gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.5, wspace=0.3)

ax_ecg  = fig.add_subplot(gs[0, :])
ax_ppg  = fig.add_subplot(gs[1, :])
ax_conf = fig.add_subplot(gs[2, 0])
ax_rr   = fig.add_subplot(gs[2, 1])
ax_stat = fig.add_subplot(gs[3, 0])
ax_temp = fig.add_subplot(gs[3, 1])

for ax in [ax_ecg, ax_ppg, ax_conf, ax_rr, ax_stat, ax_temp]:
    ax.set_facecolor('#0f0f1a')
    ax.tick_params(colors='#aaaaaa', labelsize=8)
    for spine in ax.spines.values(): spine.set_edgecolor('#333355')

# ... (ECG/PPG/Conf/RR code remains same) ...
# ECG
DISP_ECG = 5 * FS_ECG
ecg_line, = ax_ecg.plot(range(DISP_ECG), np.zeros(DISP_ECG), color='#00ff88', lw=0.8)
ax_ecg.set_title('Live ECG (AD8232)', color='#cccccc', fontsize=9)
beat_overlay = ax_ecg.text(0.01, 0.88, 'WAITING...', transform=ax_ecg.transAxes, color='white', fontweight='bold', bbox=dict(facecolor='#333355', alpha=0.8))

# PPG
DISP_PPG = 5 * FS_PPG
ppg_line, = ax_ppg.plot(range(DISP_PPG), np.zeros(DISP_PPG), color='#ff0066', lw=1.2)
ax_ppg.set_title('Live PPG (MAX30100)', color='#cccccc', fontsize=9)

# Confidence
arr_bars = ax_conf.barh(CLASS_NAMES_ARR, [0]*5, color=CLASS_COLORS_ARR, height=0.6)
ax_conf.set_xlim(0, 1)
ax_conf.set_title('Beat Classification', color='#cccccc', fontsize=9)
arr_texts = [ax_conf.text(0.01, i, '', color='white', va='center', fontsize=8) for i in range(5)]

# RR Intervals
rr_disp = np.ones(60) * 800
rr_line, = ax_rr.plot(range(60), rr_disp, color='#00ff88', marker='o', markersize=3, lw=1)
ax_rr.set_title('Pulse Timing (AFib)', color='#cccccc', fontsize=9)
ax_rr.set_ylim(400, 1500)

# Status
ax_stat.set_axis_off()
status_text = ax_stat.text(0.5, 0.5, 'INITIALIZING...', color='white', ha='center', va='center', fontsize=14, fontweight='bold')

# Temperature Plot
temp_line, = ax_temp.plot(range(60), list(temp_history), color='#ff9f43', lw=2)
ax_temp.set_title('Body Temperature (°C)', color='#cccccc', fontsize=9)
ax_temp.set_ylim(30, 42)
temp_overlay = ax_temp.text(0.95, 0.1, '--.-°C', transform=ax_temp.transAxes, color='#ff9f43', fontweight='bold', ha='right', fontsize=14)

# Overlay
screen_overlay = fig.text(0.5, 0.5, '', ha='center', va='center', color='white', 
                          bbox=dict(boxstyle='round,pad=1', facecolor='#1a1a2e', alpha=0.9, edgecolor='#00ff88'), 
                          visible=False, zorder=10)

# Button
ax_btn = fig.add_axes([0.43, 0.01, 0.14, 0.035])
btn = Button(ax_btn, 'START SCREENING', color='#2ecc71', hovercolor='#27ae60')
btn.label.set_color('white')

def on_click(event):
    global is_screening, screening_start_time, screening_results
    if not is_screening:
        is_screening = True
        screening_start_time = time.time()
        screening_results = {'arr_counts': [0]*5, 'afib_windows': 0, 'afib_detections': 0, 'total_beats': 0, 'max_temp': 0.0}
        btn.label.set_text('CANCEL')
        btn.color = '#e74c3c'
        screen_overlay.set_visible(True)
    else:
        is_screening = False
        btn.label.set_text('START SCREENING')
        btn.color = '#2ecc71'
        screen_overlay.set_visible(False)
btn.on_clicked(on_click)

# ── Serial ─────────────────────────────────────────────────────────────────
print(f"Connecting to {PORT}...")
try:
    ser = serial.Serial(PORT, BAUD, timeout=0.01)
    time.sleep(2)
    ser.reset_input_buffer()
    print("✓ Connected.")
except Exception as e:
    print(f"✗ Serial Error: {e}")
    sys.exit(1)

# ── Update ─────────────────────────────────────────────────────────────────
def update(frame):
    global ecg_cursor, ecg_last_processed, ppg_cursor, beat_count_ppg, last_ppg_peak_time, rr_disp, is_screening, current_temp

    while ser.in_waiting > 0:
        try:
            line = ser.readline().decode('utf-8').strip()
            if not line or ':' not in line: continue
            prefix, val_str = line.split(':')
            val = float(val_str)
            
            if prefix == 'E': # ECG Data
                ecg_buffer[ecg_cursor % len(ecg_buffer)] = val
                ecg_cursor += 1
            elif prefix == 'P': # PPG Data
                ppg_buffer[ppg_cursor % len(ppg_buffer)] = val
                if ppg_detector.detect(val, ppg_cursor):
                    now = time.time()
                    rr_val = (now - last_ppg_peak_time) * 1000.0
                    last_ppg_peak_time = now
                    if 400 < rr_val < 1500:
                        rr_buffer.append(rr_val)
                        rr_disp = np.roll(rr_disp, -1); rr_disp[-1] = rr_val
                        rr_line.set_ydata(rr_disp)
                        beat_count_ppg += 1
                        
                        if len(rr_buffer) == AFIB_WIN_SIZE and beat_count_ppg % 5 == 0:
                            window_rr = np.array(rr_buffer)
                            feat = extract_hrv_features(window_rr)
                            feat_sc = afib_hsc.transform(feat.reshape(1, -1))
                            seq = (window_rr - h_rr_stats[0]) / h_rr_stats[1]
                            seq = seq.reshape(1, 30, 1).astype(np.float32)
                            prob = afib_hybrid.predict([seq, feat_sc], verbose=0)[0][1]
                            is_afib = prob >= AFIB_THRESH
                            
                            # Live AFib status text removed as requested
                            if is_screening:
                                screening_results['afib_windows'] += 1
                                if is_afib: screening_results['afib_detections'] += 1
                ppg_cursor += 1
            elif prefix == 'T': # Temp Data
                current_temp = val
                temp_history.append(val)
                temp_line.set_ydata(list(temp_history))
                temp_overlay.set_text(f"{val:.1f}°C")
                if is_screening:
                    screening_results['max_temp'] = max(screening_results['max_temp'], val)
        except: continue

    # Process ECG Peaks for Arrhythmia
    for i in range(ecg_last_processed, ecg_cursor):
        if ecg_detector.detect(ecg_buffer, i):
            if is_screening: screening_results['total_beats'] += 1
            
            # Extract segment
            curr_p = i % len(ecg_buffer)
            start, end = (curr_p - BEFORE), (curr_p + AFTER)
            if start < 0 or end >= len(ecg_buffer):
                indices = np.arange(start, end) % len(ecg_buffer)
                seg = ecg_buffer[indices].copy()
            else:
                seg = ecg_buffer[start:end].copy()
            
            # Inference
            seg = (seg - np.mean(seg)) / (np.std(seg) + 1e-8)
            inp = seg.reshape(1, WIN_ARR, 1).astype(np.float32)
            probs = arr_model.predict(inp, verbose=0)[0]
            pred = np.argmax(probs)
            
            if is_screening: screening_results['arr_counts'][pred] += 1
            
            # Update UI
            beat_overlay.set_text(f" {CLASS_NAMES_ARR[pred]} ")
            beat_overlay.set_bbox(dict(facecolor=CLASS_COLORS_ARR[pred], alpha=0.8))
            for j, (bar, txt) in enumerate(zip(arr_bars, arr_texts)):
                bar.set_width(probs[j])
                txt.set_text(f"{probs[j]*100:.1f}%")
                txt.set_x(probs[j] + 0.01)
                bar.set_alpha(1.0 if j == pred else 0.3)
    ecg_last_processed = ecg_cursor

    # Update Waveforms
    e_idx = ecg_cursor % len(ecg_buffer)
    e_disp = np.roll(ecg_buffer, -e_idx)[-DISP_ECG:]
    ecg_line.set_ydata(e_disp)
    ax_ecg.set_ylim(min(e_disp)-100, max(e_disp)+100)

    p_idx = ppg_cursor % len(ppg_buffer)
    p_disp = np.roll(ppg_buffer, -p_idx)[-DISP_PPG:]
    ppg_line.set_ydata(p_disp)
    if len(p_disp) > 0:
        ymin, ymax = np.min(p_disp), np.max(p_disp)
        if ymax > ymin: ax_ppg.set_ylim(ymin - (ymax-ymin)*0.2, ymax + (ymax-ymin)*0.2)

    # Screening Logic
    if is_screening:
        elapsed = time.time() - screening_start_time
        rem = max(0, SCREENING_DURATION - elapsed)
        screen_overlay.set_text(f"SCREENING: {int(rem)}s\nBeats: {screening_results['total_beats']}")
        if elapsed >= SCREENING_DURATION:
            is_screening = False
            btn.label.set_text('START SCREENING')
            btn.color = '#2ecc71'
            res = screening_results
            
            # 1. Arrhythmia Analysis
            arr_counts = res['arr_counts']
            abnormal_arr = sum(arr_counts[1:])
            pvc_count = arr_counts[2]
            
            arr_risk = "LOW"
            if pvc_count > 5 or abnormal_arr > 15: arr_risk = "HIGH"
            elif abnormal_arr > 5: arr_risk = "MODERATE"
            
            # 2. AFib Analysis
            afib_ratio = (res['afib_detections'] / max(1, res['afib_windows']))
            afib_risk = "LOW"
            if afib_ratio > 0.15: afib_risk = "HIGH"
            elif afib_ratio > 0.05: afib_risk = "MODERATE"
            
            # 3. Temperature Analysis
            max_t = res['max_temp']
            temp_status = "NORMAL"
            if max_t > 37.8: temp_status = "FEVER"
            elif max_t < 35.5 and max_t > 0: temp_status = "LOW"

            report = (f"--- UNIFIED DIAGNOSTIC REPORT ---\n\n"
                      f"1. ARRHYTHMIA RISK : {arr_risk}\n"
                      f"   - Abnormal Beats: {abnormal_arr}\n"
                      f"   - PVC Count: {pvc_count}\n\n"
                      f"2. AFIB RISK       : {afib_risk}\n"
                      f"   - AFib Burden: {afib_ratio*100:.1f}%\n\n"
                      f"3. TEMPERATURE     : {max_t:.1f}°C\n"
                      f"   - Status: {temp_status}\n\n"
                      f"Total ECG Beats Processed: {res['total_beats']}\n\n"
                      f"Click Cancel to close.")
            
            screen_overlay.set_text(report)
            # Set border color based on highest risk
            final_color = "#2ecc71" # Green
            if arr_risk == "HIGH" or afib_risk == "HIGH" or temp_status == "FEVER":
                final_color = "#e74c3c" # Red
            elif arr_risk == "MODERATE" or afib_risk == "MODERATE":
                final_color = "#f39c12" # Yellow
            
            screen_overlay.get_bbox_patch().set_edgecolor(final_color)

    return ecg_line, ppg_line, rr_line

ani = FuncAnimation(fig, update, interval=30, cache_frame_data=False)
plt.show()
