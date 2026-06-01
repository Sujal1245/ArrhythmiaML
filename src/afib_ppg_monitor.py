# afib_ppg_monitor.py
# Optimized for macOS (Using MacOSX backend and FuncAnimation)

import os
import sys
import time
import collections
from pathlib import Path

import serial
import serial.tools.list_ports
import numpy as np

import matplotlib
matplotlib.use('MacOSX') # Matches working unified_dashboard.py
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
FS      = 100
AFIB_WIN_SIZE = 30
AFIB_THRESH   = 0.846

# ── Load AFib Models ───────────────────────────────────────────────────────
MODELS_DIR = ROOT / 'models' / 'afib'
print("Loading AFib models...")
afib_hybrid = tf.keras.models.load_model(MODELS_DIR / 'best_afib_hybrid.keras')
afib_hsc    = joblib.load(MODELS_DIR / 'afib_hybrid_scaler.pkl')
h_rr_stats  = np.load(MODELS_DIR / 'afib_hybrid_rr_stats.npy')

# ── PPG Peak Detector ──────────────────────────────────────────────────────
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
            # Lowered threshold from 1.05 to 1.02 for better sensitivity
            if val == max(local_win) and val > np.mean(self.buffer) * 1.02:
                self.last_peak = absolute_index
                return True
        return False

# ── State ──────────────────────────────────────────────────────────────────
raw_buffer = np.zeros(FS * 10)
cursor = 0
beat_count = 0
rr_buffer = collections.deque(maxlen=AFIB_WIN_SIZE)
last_peak_time = time.time()
is_screening = False
screening_start_time = 0
SCREENING_DURATION = 60
screening_results = {'afib_windows': 0, 'afib_detections': 0, 'total_beats': 0}
detector = PPGPeakDetector(FS)

# ── UI Layout ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 8), facecolor='#1a1a2e')
fig.suptitle('AFib Pulse Monitor — MAX30100 PPG', color='white', fontsize=14)

gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.4)
ax_ppg = fig.add_subplot(gs[0]); ax_ppg.set_facecolor('#0f0f1a')
ax_rr  = fig.add_subplot(gs[1]); ax_rr.set_facecolor('#0f0f1a')
ax_stat = fig.add_subplot(gs[2]); ax_stat.set_facecolor('#0f0f1a'); ax_stat.set_axis_off()

for ax in [ax_ppg, ax_rr]:
    ax.tick_params(colors='#aaaaaa', labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333355')

DISP_LEN = 5 * FS
ppg_line, = ax_ppg.plot(range(DISP_LEN), np.zeros(DISP_LEN), color='#ff0066', lw=1.5)
ax_ppg.set_title('Live PPG Signal (Pulse)', color='#cccccc', fontsize=9)

rr_disp = np.ones(60) * 800
rr_line, = ax_rr.plot(range(60), rr_disp, color='#00ff88', marker='o', markersize=3, lw=1)
ax_rr.set_title('Beat-to-Beat Intervals (ms)', color='#cccccc', fontsize=9)
ax_rr.set_ylim(400, 1500)

afib_text = ax_stat.text(0.5, 0.5, 'WAITING FOR DATA', color='white', ha='center', va='center', fontsize=20, fontweight='bold')
screen_overlay = fig.text(0.5, 0.5, '', ha='center', va='center', color='white', 
                          bbox=dict(boxstyle='round,pad=1', facecolor='#1a1a2e', alpha=0.9, edgecolor='#ff0066'), 
                          visible=False, zorder=10)

ax_btn = fig.add_axes([0.4, 0.02, 0.2, 0.05])
btn = Button(ax_btn, 'START AFIB SCREEN', color='#ff0066', hovercolor='#cc0055')
btn.label.set_color('white')
btn.label.set_fontweight('bold')

def on_click(event):
    global is_screening, screening_start_time, screening_results
    if not is_screening:
        is_screening = True
        screening_start_time = time.time()
        screening_results = {'afib_windows': 0, 'afib_detections': 0, 'total_beats': 0}
        btn.label.set_text('CANCEL')
        screen_overlay.set_visible(True)
    else:
        is_screening = False
        btn.label.set_text('START AFIB SCREEN')
        screen_overlay.set_visible(False)
btn.on_clicked(on_click)

# ── Serial Connection ──────────────────────────────────────────────────────
print(f"Connecting to {PORT}...")
try:
    ser = serial.Serial(PORT, BAUD, timeout=0.01)
    time.sleep(2)
    ser.reset_input_buffer()
    print("✓ Connected.")
except Exception as e:
    print(f"✗ Serial Error: {e}")
    print("\nAvailable Ports:")
    for p in serial.tools.list_ports.comports():
        print(f"  {p.device}")
    sys.exit(1)

# ── Animation Update ───────────────────────────────────────────────────────
def update(frame):
    global cursor, beat_count, last_peak_time, rr_disp, is_screening

    # Read Serial
    while ser.in_waiting > 0:
        try:
            line = ser.readline().decode('utf-8').strip()
            if not line: continue
            val = float(line)
            raw_buffer[cursor % len(raw_buffer)] = val
            
            if detector.detect(val, cursor):
                now = time.time()
                rr_val = (now - last_peak_time) * 1000.0
                last_peak_time = now
                
                if 400 < rr_val < 1500:
                    rr_buffer.append(rr_val)
                    rr_disp = np.roll(rr_disp, -1)
                    rr_disp[-1] = rr_val
                    rr_line.set_ydata(rr_disp)
                    beat_count += 1
                    if is_screening: screening_results['total_beats'] += 1

                if len(rr_buffer) == AFIB_WIN_SIZE and beat_count % 5 == 0:
                    window_rr = np.array(rr_buffer)
                    feat = extract_hrv_features(window_rr)
                    feat_sc = afib_hsc.transform(feat.reshape(1, -1))
                    seq = (window_rr - h_rr_stats[0]) / h_rr_stats[1]
                    seq = seq.reshape(1, 30, 1).astype(np.float32)
                    prob = afib_hybrid.predict([seq, feat_sc], verbose=0)[0][1]
                    is_afib = prob >= AFIB_THRESH
                    afib_text.set_text('AFIB DETECTED' if is_afib else 'NORMAL RHYTHM')
                    afib_text.set_color('#ff0066' if is_afib else '#00ff88')
                    if is_screening:
                        screening_results['afib_windows'] += 1
                        if is_afib: screening_results['afib_detections'] += 1
            cursor += 1
        except: pass

    # Update Plot
    idx = cursor % len(raw_buffer)
    disp = np.roll(raw_buffer, -idx)[-DISP_LEN:]
    ppg_line.set_ydata(disp)
    
    if len(disp) > 0:
        ymin, ymax = np.min(disp), np.max(disp)
        if ymax > ymin:
            ax_ppg.set_ylim(ymin - (ymax-ymin)*0.2, ymax + (ymax-ymin)*0.2)

    if is_screening:
        elapsed = time.time() - screening_start_time
        rem = max(0, SCREENING_DURATION - elapsed)
        screen_overlay.set_text(f"SCREENING: {int(rem)}s\nBeats: {screening_results['total_beats']}")
        if elapsed >= SCREENING_DURATION:
            is_screening = False
            btn.label.set_text('START AFIB SCREEN')
            burden = (screening_results['afib_detections'] / max(1, screening_results['afib_windows'])) * 100
            res = "🔴 AFIB DETECTED" if burden > 15 else "🟢 NORMAL"
            screen_overlay.set_text(f"REPORT\n\nResult: {res}\nBurden: {burden:.1f}%\n\nClick Cancel to Close")

    return ppg_line, rr_line

ani = FuncAnimation(fig, update, interval=30, cache_frame_data=False)
plt.show()
