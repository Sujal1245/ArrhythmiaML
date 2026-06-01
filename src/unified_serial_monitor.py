# unified_serial_monitor.py
#
# Unified Real-Time Screening Tool:
#   1. Connects to ESP32 via Serial (USB).
#   2. Displays Live ECG waveform and Beat Classification.
#   3. "Start Screening" button triggers a 2-minute diagnostic check.
#   4. Final Verdict & Recommendation report at the end.
#
# ESP32 Setup: Send raw ADC values (one per line) at ~360Hz.

import os
import sys
import time
import collections
from pathlib import Path

import serial
import serial.tools.list_ports
import numpy as np

import matplotlib
matplotlib.use('TkAgg') # Crucial for macOS interactivity
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button
from matplotlib.animation import FuncAnimation

import tensorflow as tf
import joblib
from scipy.signal import butter, filtfilt

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.common.signal_processing import normalize_segment
from src.afib.build_dataset import extract_hrv_features

# ── Configuration ──────────────────────────────────────────────────────────
PORT    = '/dev/cu.usbserial-0001'  # ← Change to your port
BAUD    = 115200
FS      = 360
WIN_ARR = 280
BEFORE  = 100
AFTER   = WIN_ARR - BEFORE
AFIB_WIN_SIZE = 30
AFIB_THRESH   = 0.846

CLASS_NAMES_ARR  = ['Normal (N)', 'Supraventricular (S)', 'Ventricular (V)', 'Fusion (F)', 'Unknown (Q)']
CLASS_COLORS_ARR = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']
CLASS_NAMES_AFIB = ['Normal', 'AFib']
CLASS_COLORS_AFIB = ['#2ecc71', '#e74c3c']

# ── Load Models ────────────────────────────────────────────────────────────
MODELS_DIR = ROOT / 'models'
print("Loading Arrhythmia model...")
arr_model = tf.keras.models.load_model(MODELS_DIR / 'arrhythmia' / 'best_model.keras')

print("Loading AFib hybrid model...")
afib_hybrid = tf.keras.models.load_model(MODELS_DIR / 'afib' / 'best_afib_hybrid.keras')
afib_hsc    = joblib.load(MODELS_DIR / 'afib' / 'afib_hybrid_scaler.pkl')
h_rr_stats  = np.load(MODELS_DIR / 'afib' / 'afib_hybrid_rr_stats.npy')
print("All models loaded.\n")

# ── Signal Processing Helpers ──────────────────────────────────────────────
class RPeakDetector:
    def __init__(self, fs=360):
        self.fs = fs
        self.min_rr = int(0.3 * fs) 
        self.last_peak = 0
        self.threshold = None
        self.signal_level = 0.0
        self.noise_level = 0.0

    def detect(self, buf, cursor):
        """
        Detects R-peaks using an adaptive energy threshold.
        cursor: monotonically increasing absolute sample index.
        buf: circular buffer of size FS*10.
        """
        buf_len = len(buf)
        if cursor < 30: return False
        
        # Get window using monotone indices mapped to circular buffer
        indices = (np.arange(cursor - 10, cursor)) % buf_len
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
        
        # Monotone comparison works across buffer wraps
        if energy > self.threshold and (cursor - self.last_peak) > self.min_rr:
            self.last_peak = cursor
            return True
        return False

# ── Mutable Simulation State ───────────────────────────────────────────────
raw_buffer = np.zeros(FS * 10, dtype=np.float32)
cursor = 0
last_processed_cursor = 0
beat_count = 0
rr_buffer = collections.deque(maxlen=AFIB_WIN_SIZE)
last_peak_time = time.time()

# Screening State
is_screening = False
screening_start_time = 0
SCREENING_DURATION = 120 # seconds
screening_results = {
    'arr_counts': [0]*5,
    'afib_windows': 0,
    'afib_detections': 0,
    'total_beats': 0
}

detector = RPeakDetector(FS)

# ── Figure Layout ──────────────────────────────────────────────────────────
plt.ion()
fig = plt.figure(figsize=(16, 10), facecolor='#1a1a2e')
fig.suptitle('Unified Cardiac Monitor — Live ESP32 Screening', color='white', fontsize=14, fontweight='bold')

gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.4, wspace=0.3, left=0.06, right=0.97, top=0.92, bottom=0.12)

ax_ecg   = fig.add_subplot(gs[0, :])
ax_conf  = fig.add_subplot(gs[1, :])
ax_rr    = fig.add_subplot(gs[2, 0])
ax_hist  = fig.add_subplot(gs[2, 1])
ax_totals = fig.add_subplot(gs[3, :])

for ax in [ax_ecg, ax_conf, ax_rr, ax_hist, ax_totals]:
    ax.set_facecolor('#0f0f1a')
    ax.tick_params(colors='#aaaaaa', labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333355')

# ECG Waveform
DISP_LEN = 6 * FS
ecg_line, = ax_ecg.plot(np.linspace(0, 6, DISP_LEN), np.zeros(DISP_LEN), color='#00ff88', lw=0.8)
ax_ecg.set_ylim(-100, 4000)
ax_ecg.set_title('Live ECG Signal', color='#cccccc', fontsize=9)

# Confidence Bars
arr_bars = ax_conf.barh(CLASS_NAMES_ARR, [0]*5, color=CLASS_COLORS_ARR, height=0.6)
ax_conf.set_xlim(0, 1)
ax_conf.set_title('Beat Classification Confidence', color='#cccccc', fontsize=9)
arr_texts = [ax_conf.text(0.01, i, '', color='white', va='center', fontweight='bold', fontsize=8) for i in range(5)]

# RR Intervals
rr_disp = np.ones(60) * 800
rr_line, = ax_rr.plot(range(60), rr_disp, color='#00ff88', lw=1)
ax_rr.set_ylim(300, 1500)
ax_rr.set_title('Rolling RR Intervals (ms)', color='#cccccc', fontsize=9)

# Beat History
scat = ax_hist.scatter([], [], c=[], s=30, zorder=3)
ax_hist.set_xlim(0, 40)
ax_hist.set_ylim(-0.5, 4.5)
ax_hist.set_yticks(range(5))
ax_hist.set_yticklabels(CLASS_NAMES_ARR, fontsize=7)
ax_hist.set_title('Beat History', color='#cccccc', fontsize=9)
history_cls = []; history_col = []

# Overlays
beat_overlay = ax_ecg.text(0.01, 0.88, 'Waiting...', transform=ax_ecg.transAxes, color='white', fontweight='bold', bbox=dict(facecolor='#333355', alpha=0.8))
status_text = ax_ecg.text(0.99, 0.88, 'Connecting...', transform=ax_ecg.transAxes, color='#aaaaaa', ha='right', bbox=dict(facecolor='#1a1a2e', alpha=0.7))

# UI Components
ax_screen = fig.add_axes([0.44, 0.03, 0.12, 0.05])
btn_screen = Button(ax_screen, 'START SCREENING', color='#2ecc71', hovercolor='#27ae60')
btn_screen.label.set_color('white')
btn_screen.label.set_fontweight('bold')

screen_overlay = fig.text(
    0.5, 0.5, '', ha='center', va='center', fontsize=18, fontweight='bold', color='white',
    bbox=dict(boxstyle='round,pad=1.5', facecolor='#1a1a2e', alpha=0.95, edgecolor='#00ff88'),
    visible=False, zorder=10
)

# ── Serial Connection ──────────────────────────────────────────────────────
print(f"Connecting to {PORT}...")
try:
    ser = serial.Serial(PORT, BAUD, timeout=0.01)
    time.sleep(2)
    ser.flushInput()
    print("✓ Connected to Serial.")
except Exception as e:
    print(f"✗ Serial Error: {e}")
    print("\nAvailable Ports:")
    for p in serial.tools.list_ports.comports(): print(f"  {p.device}")
    sys.exit(1)

# ── Callbacks ──────────────────────────────────────────────────────────────
def on_screen_clicked(event):
    global is_screening, screening_start_time, screening_results
    if not is_screening:
        is_screening = True
        screening_start_time = time.time()
        screening_results = {'arr_counts': [0]*5, 'afib_windows': 0, 'afib_detections': 0, 'total_beats': 0}
        btn_screen.label.set_text('CANCEL')
        btn_screen.color = '#e74c3c'
        screen_overlay.set_visible(True)
    else:
        is_screening = False
        btn_screen.label.set_text('START SCREENING')
        btn_screen.color = '#2ecc71'
        screen_overlay.set_visible(False)

btn_screen.on_clicked(on_screen_clicked)

def on_overlay_click(event):
    if screen_overlay.get_visible() and not is_screening:
        screen_overlay.set_visible(False)
        fig.canvas.draw_idle()

fig.canvas.mpl_connect('button_press_event', on_overlay_click)

# ── Animation Loop ─────────────────────────────────────────────────────────
def update(frame):
    global cursor, last_processed_cursor, beat_count, rr_disp, scat, history_cls, history_col, is_screening, last_peak_time

    # Read all available data from Serial
    while ser.in_waiting > 0:
        try:
            line = ser.readline().decode('utf-8').strip()
            if not line: continue
            val = float(line)
            raw_buffer[cursor % len(raw_buffer)] = val
            cursor += 1
        except:
            continue

    if cursor < DISP_LEN: return

    # Update ECG Waveform
    idx = cursor % len(raw_buffer)
    disp = np.roll(raw_buffer, -idx)[-DISP_LEN:]
    ecg_line.set_ydata(disp)
    ax_ecg.set_ylim(min(disp)-100, max(disp)+100)

    # ── Screening Logic ──
    if is_screening:
        elapsed = time.time() - screening_start_time
        remaining = max(0, SCREENING_DURATION - elapsed)
        screen_overlay.set_text(f"SCREENING IN PROGRESS\n\n{int(remaining)}s Remaining\n\nKEEP STILL")
        
        if elapsed >= SCREENING_DURATION:
            is_screening = False
            btn_screen.label.set_text('START SCREENING')
            btn_screen.color = '#2ecc71'
            
            res = screening_results
            arr_counts = res['arr_counts']
            abnormal = sum(arr_counts[1:])
            afib_ratio = (res['afib_detections'] / max(1, res['afib_windows']))
            
            verdict = "🟢 NORMAL HEART RHYTHM"
            color = "#2ecc71"
            advice = "Your heart rhythm appears regular. This tool is for screening and does not replace professional medical advice."
            
            if afib_ratio > 0.1:
                verdict = "🔴 ATRIAL FIBRILLATION DETECTED"
                color = "#e74c3c"
                advice = "Significant signs of AFib detected. Please consult a cardiologist for a diagnostic 12-lead ECG."
            elif arr_counts[2] > 5 or abnormal > 20:
                verdict = "🔴 FREQUENT IRREGULARITIES"
                color = "#e74c3c"
                advice = "Frequent abnormal beats (PVCs/Arrhythmias) detected. A follow-up with a specialist is recommended."
            elif abnormal > 5:
                verdict = "🟡 OCCASIONAL IRREGULARITIES"
                color = "#f39c12"
                advice = "A few irregular beats were found. This can be normal, but monitor for symptoms like palpitations."

            breakdown = []
            for i in range(1, 5):
                if arr_counts[i] > 0:
                    breakdown.append(f" • {CLASS_NAMES_ARR[i]}: {arr_counts[i]}")
            breakdown_str = "\n".join(breakdown) if breakdown else " None detected."

            report = (f"--- CARDIAC SCREENING REPORT ---\n\n"
                      f"RESULT: {verdict}\n\n"
                      f"Beats Processed: {res['total_beats']}\n"
                      f"AFib Burden: {afib_ratio*100:.1f}%\n\n"
                      f"Abnormal Beats Breakdown:\n{breakdown_str}\n\n"
                      f"RECOMMENDATION:\n{advice}\n\n"
                      f"Click anywhere to close.")
            screen_overlay.set_text(report)
            screen_overlay.get_bbox_patch().set_edgecolor(color)

    # ── Peak Detection & Inference Loop ──
    # Check all samples that have arrived since the last frame
    for i in range(last_processed_cursor, cursor):
        if detector.detect(raw_buffer, i):
            now = time.time()
            rr_val = (now - last_peak_time) * 1000.0 # ms
            last_peak_time = now
            
            if 300 < rr_val < 2000: # Physiological filter
                rr_buffer.append(rr_val)
                rr_disp = np.roll(rr_disp, -1)
                rr_disp[-1] = rr_val
                rr_line.set_ydata(rr_disp)
            
            beat_count += 1
            if is_screening: screening_results['total_beats'] += 1
            
            # Arrhythmia Inference
            curr_p = i % len(raw_buffer)
            start, end = (curr_p - BEFORE), (curr_p + AFTER)
            # Need to handle circular buffer indexing for the segment
            if start < 0 or end >= len(raw_buffer):
                # Segment wraps around buffer end
                indices = np.arange(start, end) % len(raw_buffer)
                seg = raw_buffer[indices].copy()
            else:
                seg = raw_buffer[start:end].copy()
                
            seg = (seg - np.mean(seg)) / (np.std(seg) + 1e-8)
            inp = seg.reshape(1, WIN_ARR, 1).astype(np.float32)
            probs = arr_model.predict(inp, verbose=0)[0]
            pred = np.argmax(probs)
            
            if is_screening: screening_results['arr_counts'][pred] += 1
            
            # Update UI
            for j, (bar, txt) in enumerate(zip(arr_bars, arr_texts)):
                bar.set_width(probs[j])
                txt.set_text(f"{probs[j]*100:.1f}%")
                txt.set_x(probs[j] + 0.01)
                bar.set_alpha(1.0 if j == pred else 0.3)
            
            beat_overlay.set_text(f" {CLASS_NAMES_ARR[pred]} ")
            beat_overlay.set_bbox(dict(facecolor=CLASS_COLORS_ARR[pred], alpha=0.8))
            
            history_cls.append(pred)
            history_col.append(CLASS_COLORS_ARR[pred])
            if len(history_cls) > 40: history_cls.pop(0); history_col.pop(0)
            scat.set_offsets(np.c_[range(len(history_cls)), history_cls])
            scat.set_color(history_col)

            # AFib Inference (Every 10 beats)
            if len(rr_buffer) == AFIB_WIN_SIZE and beat_count % 10 == 0:
                window_rr = np.array(rr_buffer, dtype=np.float32)
                feat = extract_hrv_features(window_rr)
                feat_sc = afib_hsc.transform(feat.reshape(1, -1))
                seq = (window_rr - h_rr_stats[0]) / h_rr_stats[1]
                seq = seq.reshape(1, 30, 1).astype(np.float32)
                
                afib_probs = afib_hybrid.predict([seq, feat_sc], verbose=0)[0]
                is_afib = 1 if afib_probs[1] >= AFIB_THRESH else 0
                
                if is_screening:
                    screening_results['afib_windows'] += 1
                    if is_afib: screening_results['afib_detections'] += 1
                
                ax_rr.set_facecolor('#4a1a1a' if is_afib else '#0f0f1a')

    last_processed_cursor = cursor
    status_text.set_text(f"● LIVE | Beats: {beat_count}")
    return ecg_line, scat

ani = FuncAnimation(fig, update, interval=30, cache_frame_data=False)
plt.show(block=True)
