# serial_inference.py
import os
os.environ['PYTHONUNBUFFERED'] = '1'

import matplotlib
matplotlib.use('MacOSX')

from pathlib import Path
import serial
import serial.tools.list_ports
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import entropy
import time
import collections

ROOT       = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / 'models' / 'afib'

PORT    = '/dev/cu.usbserial-0001'   # ← change to your port
BAUD    = 115200
WINDOW  = 30
MIN_RR  = 300
MAX_RR  = 2000

CLASS_NAMES  = ['Normal', 'AFib']
COLORS       = ['#2ecc71', '#e74c3c']
VOTE_WINDOW  = 5
VOTE_THRESH  = 3


class TemporalVoter:
    def __init__(self, window=VOTE_WINDOW, thresh=VOTE_THRESH):
        self.buf    = collections.deque(maxlen=window)
        self.thresh = thresh

    def update(self, pred):
        self.buf.append(pred)
        return int(sum(self.buf) >= self.thresh)

print("Loading models...")
cnn      = tf.keras.models.load_model(MODELS_DIR / 'best_afib_cnn.keras')
rf       = joblib.load(MODELS_DIR / 'afib_rf_model.pkl')
scaler   = joblib.load(MODELS_DIR / 'afib_scaler.pkl')
rr_stats = np.load(MODELS_DIR / 'afib_rr_stats.npy')
print("Models ready.\n")


def extract_hrv_features(rr_intervals):
    rr   = np.array(rr_intervals, dtype=np.float64)
    diff = np.diff(rr)
    mean_rr   = np.mean(rr);  std_rr  = np.std(rr)
    rmssd     = np.sqrt(np.mean(diff ** 2))
    cv        = std_rr / (mean_rr + 1e-8)
    pnn50     = np.sum(np.abs(diff) > 50) / len(diff) * 100
    max_rr    = np.max(rr);   min_rr  = np.min(rr)
    range_rr  = max_rr - min_rr
    hist, _   = np.histogram(rr, bins=10, density=True)
    hist      = hist + 1e-10
    sh_entropy = entropy(hist)
    ief       = np.sum(np.abs(diff) > 0.1 * mean_rr) / len(diff)
    sd1       = np.std(diff) / np.sqrt(2)
    sd2       = np.sqrt(2 * std_rr ** 2 - 0.5 * np.std(diff) ** 2 + 1e-8)
    return np.array([
        mean_rr, std_rr, rmssd, cv, pnn50,
        max_rr, min_rr, range_rr,
        sh_entropy, ief, sd1, sd2, len(rr)
    ], dtype=np.float32)


plt.ion()
fig = plt.figure(figsize=(14, 9), facecolor='#1a1a2e')
fig.suptitle('Live AFib Detector — MAX30100 + ESP32  (USB)',
             color='white', fontsize=12, fontweight='bold')

gs = gridspec.GridSpec(3, 2, figure=fig,
                       hspace=0.5, wspace=0.35,
                       left=0.07, right=0.97, top=0.91, bottom=0.07)

ax_rr    = fig.add_subplot(gs[0, :])
ax_rf    = fig.add_subplot(gs[1, 0])
ax_cnn   = fig.add_subplot(gs[1, 1])
ax_hist  = fig.add_subplot(gs[2, 0])
ax_count = fig.add_subplot(gs[2, 1])

for ax in [ax_rr, ax_rf, ax_cnn, ax_hist, ax_count]:
    ax.set_facecolor('#0f0f1a')
    ax.tick_params(colors='#aaaaaa', labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333355')

DISP_BEATS = 60
rr_disp    = np.ones(DISP_BEATS) * 800
rr_line,   = ax_rr.plot(range(DISP_BEATS), rr_disp,
                         color='#00ff88', linewidth=1.0)
ax_rr.set_xlim(0, DISP_BEATS)
ax_rr.set_ylim(300, 1500)
ax_rr.set_ylabel('RR interval (ms)', color='#aaaaaa', fontsize=8)
ax_rr.set_xlabel('Beats',            color='#aaaaaa', fontsize=8)
ax_rr.set_title('Live RR intervals from MAX30100',
                color='#cccccc', fontsize=9, pad=4)
ax_rr.axhline(y=600,  color='#e74c3c', linewidth=0.5, linestyle='--',
              alpha=0.4, label='100 bpm')
ax_rr.axhline(y=1000, color='#3498db', linewidth=0.5, linestyle='--',
              alpha=0.4, label='60 bpm')
ax_rr.legend(fontsize=7, loc='upper right',
             labelcolor='#aaaaaa', facecolor='#1a1a2e')

rf_bars  = ax_rf.barh(CLASS_NAMES, [0, 0], color=COLORS, height=0.5)
ax_rf.set_xlim(0, 1)
ax_rf.set_title('Random Forest confidence', color='#cccccc', fontsize=9, pad=4)
ax_rf.set_xlabel('Confidence', color='#aaaaaa', fontsize=8)
ax_rf.tick_params(axis='y', labelsize=8)
rf_texts = [ax_rf.text(0.02, i, '', va='center',
                        color='white', fontsize=8, fontweight='bold')
            for i in range(2)]

cnn_bars = ax_cnn.barh(CLASS_NAMES, [0, 0], color=COLORS, height=0.5)
ax_cnn.set_xlim(0, 1)
ax_cnn.set_title('1D-CNN confidence', color='#cccccc', fontsize=9, pad=4)
ax_cnn.set_xlabel('Confidence', color='#aaaaaa', fontsize=8)
ax_cnn.tick_params(axis='y', labelsize=8)
cnn_texts = [ax_cnn.text(0.02, i, '', va='center',
                          color='white', fontsize=8, fontweight='bold')
             for i in range(2)]

history_rf  = []
history_cnn = []
MAX_HIST    = 50
scat_rf     = ax_hist.scatter([], [], c=[], s=35, zorder=3)
scat_cnn    = ax_hist.scatter([], [], c=[], s=35, zorder=3, marker='^')
ax_hist.set_xlim(0, MAX_HIST)
ax_hist.set_ylim(-0.5, 1.5)
ax_hist.set_yticks([0, 1])
ax_hist.set_yticklabels(CLASS_NAMES, fontsize=8)
ax_hist.set_xlabel('Last windows', color='#aaaaaa', fontsize=8)
ax_hist.set_title('Prediction history  (● RF  ▲ CNN)',
                  color='#cccccc', fontsize=9, pad=4)
ax_hist.grid(axis='y', color='#222244', linewidth=0.5)

rf_counts  = [0, 0]
cnn_counts = [0, 0]
x          = np.arange(2); w = 0.35
rf_cbars   = ax_count.bar(x - w / 2, rf_counts,  w,
                           label='RF',  color=['#1a7a42', '#7a1a1a'])
cnn_cbars  = ax_count.bar(x + w / 2, cnn_counts, w,
                           label='CNN', color=COLORS, alpha=0.8)
ax_count.set_xticks(x)
ax_count.set_xticklabels(CLASS_NAMES, fontsize=8)
ax_count.set_ylabel('Count', color='#aaaaaa', fontsize=8)
ax_count.set_title('Cumulative predictions', color='#cccccc', fontsize=9, pad=4)
ax_count.legend(fontsize=7, labelcolor='#aaaaaa', facecolor='#1a1a2e')

pred_label = ax_rr.text(
    0.01, 0.85, 'Collecting beats...', transform=ax_rr.transAxes,
    color='white', fontsize=10, fontweight='bold',
    bbox=dict(boxstyle='round,pad=0.3', facecolor='#333355', alpha=0.8))
status_text = ax_rr.text(
    0.99, 0.85, 'Connecting...', transform=ax_rr.transAxes,
    color='#aaaaaa', fontsize=8, ha='right',
    bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e', alpha=0.7))

plt.show()

print(f"Connecting to {PORT} at {BAUD} baud...")
try:
    ser = serial.Serial(PORT, BAUD, timeout=2)
    time.sleep(2)
    ser.flushInput()
    print("Connected.\n")
except serial.SerialException as e:
    print(f"\nERROR: Could not open port {PORT}")
    print(f"Details: {e}")
    print("\nAvailable ports:")
    for p in serial.tools.list_ports.comports():
        print(f"  {p.device}  —  {p.description}")
    exit(1)

rr_buffer    = collections.deque(maxlen=WINDOW)
window_count = 0
agree_count  = 0
beat_count   = 0
rf_voter     = TemporalVoter()
cnn_voter    = TemporalVoter()

print(f"Waiting for {WINDOW} beats before first classification...")
print("Place finger firmly on MAX30100 sensor.\n")

try:
    while True:
        try:
            line = ser.readline().decode('utf-8').strip()
        except UnicodeDecodeError:
            continue

        if line in ('', 'READY', 'ERROR: MAX30100 not found.'):
            continue

        try:
            rr_val = float(line)
        except ValueError:
            continue

        if not (MIN_RR < rr_val < MAX_RR):
            continue

        beat_count += 1
        rr_buffer.append(rr_val)

        rr_disp = np.roll(rr_disp, -1)
        rr_disp[-1] = rr_val
        rr_line.set_ydata(rr_disp)
        rr_min = max(200,  rr_disp.min() - 100)
        rr_max = min(2000, rr_disp.max() + 100)
        ax_rr.set_ylim(rr_min, rr_max)

        status_text.set_text(
            f'● LIVE  |  Beats: {beat_count}  |  '
            f'HR: {60000 / rr_val:.0f} bpm')

        if len(rr_buffer) < WINDOW:
            remaining = WINDOW - len(rr_buffer)
            pred_label.set_text(f'  Collecting... {remaining} beats left  ')
            plt.pause(0.01)
            continue

        window_rr = np.array(rr_buffer, dtype=np.float32)

        feat     = extract_hrv_features(window_rr).reshape(1, -1)
        feat_sc  = scaler.transform(feat)
        rf_probs = rf.predict_proba(feat_sc)[0]
        rf_pred  = int(np.argmax(rf_probs))

        seq      = (window_rr - rr_stats[0]) / rr_stats[1]
        seq      = seq.reshape(1, WINDOW, 1).astype(np.float32)
        
        # Hybrid-style prediction (using the CNN branch logic for this serial script)
        probs = cnn.predict(seq, verbose=0)[0]
        # Use optimal threshold 0.846
        pred = 1 if probs[1] >= AFIB_THRESHOLD else 0

        for i, (bar, txt) in enumerate(zip(rf_bars, rf_texts)):
            bar.set_width(rf_probs[i])
            bar.set_alpha(1.0 if i == rf_pred else 0.35)
            txt.set_text(f'{rf_probs[i] * 100:.1f}%')
            txt.set_x(rf_probs[i] + 0.01)

        for i, (bar, txt) in enumerate(zip(cnn_bars, cnn_texts)):
            bar.set_width(cnn_prob[i])
            bar.set_alpha(1.0 if i == cnn_pred else 0.35)
            txt.set_text(f'{cnn_prob[i] * 100:.1f}%')
            txt.set_x(cnn_prob[i] + 0.01)

        pos = window_count % MAX_HIST
        history_rf.append((pos,  rf_pred,  COLORS[rf_pred]))
        history_cnn.append((pos, cnn_pred, COLORS[cnn_pred]))
        if len(history_rf) > MAX_HIST:
            history_rf.pop(0)
            history_cnn.pop(0)

        scat_rf.remove(); scat_cnn.remove()
        scat_rf  = ax_hist.scatter(
            [h[0] for h in history_rf],  [h[1] for h in history_rf],
            c=[h[2] for h in history_rf],  s=35, zorder=3)
        scat_cnn = ax_hist.scatter(
            [h[0] for h in history_cnn], [h[1] for h in history_cnn],
            c=[h[2] for h in history_cnn], s=35, zorder=3, marker='^')

        rf_counts[rf_pred]   += 1
        cnn_counts[cnn_pred] += 1
        for bar, val in zip(rf_cbars,  rf_counts):  bar.set_height(val)
        for bar, val in zip(cnn_cbars, cnn_counts): bar.set_height(val)
        ax_count.set_ylim(0, max(rf_counts + cnn_counts) * 1.15 + 1)

        agree = rf_pred == cnn_pred
        if agree:
            agree_count += 1

        rf_voted  = rf_voter.update(rf_pred)
        cnn_voted = cnn_voter.update(cnn_pred)
        voted     = int(rf_voted == 1 and cnn_voted == 1)

        pred_label.set_text(
            f'  RF: {CLASS_NAMES[rf_pred]}  |  '
            f'CNN: {CLASS_NAMES[cnn_pred]}'
            f'  ||  Voted: {CLASS_NAMES[voted]}  ')
        pred_label.set_bbox(dict(boxstyle='round,pad=0.3',
                                  facecolor=COLORS[voted], alpha=0.85))

        window_count += 1
        print(f"Window {window_count:4d} | "
              f"RF: {CLASS_NAMES[rf_pred]:6s} ({rf_probs[rf_pred]*100:.0f}%) | "
              f"CNN: {CLASS_NAMES[cnn_pred]:6s} ({cnn_prob[cnn_pred]*100:.0f}%) | "
              f"Voted: {CLASS_NAMES[voted]:6s} | "
              f"HR: {60000 / rr_val:.0f} bpm")

        plt.pause(0.01)

except KeyboardInterrupt:
    print(f"\nStopped.")
    print(f"Total beats   : {beat_count}")
    print(f"Total windows : {window_count}")
    if window_count > 0:
        print(f"RF  — Normal: {rf_counts[0]}  AFib: {rf_counts[1]}")
        print(f"CNN — Normal: {cnn_counts[0]}  AFib: {cnn_counts[1]}")
        print(f"Agreement: {agree_count}/{window_count} "
              f"({agree_count / window_count * 100:.1f}%)")
finally:
    ser.close()
    plt.ioff()
    plt.show()
