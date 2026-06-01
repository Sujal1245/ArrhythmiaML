# simulate.py
import os
os.environ['PYTHONUNBUFFERED'] = '1'

import matplotlib
matplotlib.use('MacOSX')

from pathlib import Path
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import entropy
import time
import collections

ROOT       = Path(__file__).resolve().parents[2]
DATA_DIR   = ROOT / 'data'   / 'afib'
MODELS_DIR = ROOT / 'models' / 'afib'

CLASS_NAMES  = ['Normal', 'AFib']
COLORS       = ['#2ecc71', '#e74c3c']
SLEEP_TIME   = 0.05
VOTE_WINDOW  = 5   # rolling window size
VOTE_THRESH  = 3   # votes needed to declare AFib


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

print("Loading afib_dataset.npz...")
data   = np.load(DATA_DIR / 'afib_dataset.npz')
X_feat = data['X_feat']
X_seq  = data['X_seq']
y      = data['y']
print(f"Loaded {len(y)} windows — Normal: {np.sum(y==0)} | AFib: {np.sum(y==1)}\n")

plt.ion()
fig = plt.figure(figsize=(14, 9), facecolor='#1a1a2e')
fig.suptitle('Live AFib Detector — afib_dataset.npz  '
             '(Random Forest + 1D-CNN)',
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
ax_rr.set_title('Rolling RR intervals', color='#cccccc', fontsize=9, pad=4)
ax_rr.axhline(y=600,  color='#e74c3c', linewidth=0.5,
              linestyle='--', alpha=0.4, label='100 bpm')
ax_rr.axhline(y=1000, color='#3498db', linewidth=0.5,
              linestyle='--', alpha=0.4, label='60 bpm')
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
ax_count.set_title('Cumulative predictions',
                   color='#cccccc', fontsize=9, pad=4)
ax_count.legend(fontsize=7, labelcolor='#aaaaaa', facecolor='#1a1a2e')

pred_label = ax_rr.text(
    0.01, 0.85, 'Starting...', transform=ax_rr.transAxes,
    color='white', fontsize=10, fontweight='bold',
    bbox=dict(boxstyle='round,pad=0.3', facecolor='#333355', alpha=0.8))
true_label = ax_rr.text(
    0.99, 0.85, '', transform=ax_rr.transAxes,
    color='#aaaaaa', fontsize=8, ha='right',
    bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e', alpha=0.7))

plt.show()

window_count = 0
agree_count  = 0
rf_voter     = TemporalVoter()
cnn_voter    = TemporalVoter()

for i in range(len(y)):
    window_rr  = X_seq[i]
    feat       = X_feat[i]
    true_class = int(y[i])

    feat_sc  = scaler.transform(feat.reshape(1, -1))
    rf_probs = rf.predict_proba(feat_sc)[0]
    rf_pred  = int(np.argmax(rf_probs))

    seq      = (window_rr - rr_stats[0]) / rr_stats[1]
    seq      = seq.reshape(1, 30, 1).astype(np.float32)
    cnn_prob = cnn.predict(seq, verbose=0)[0]
    cnn_pred = int(np.argmax(cnn_prob))

    rr_disp = np.roll(rr_disp, -1)
    rr_disp[-1] = window_rr[-1]
    rr_line.set_ydata(rr_disp)
    rr_min = max(200,  rr_disp.min() - 100)
    rr_max = min(2000, rr_disp.max() + 100)
    ax_rr.set_ylim(rr_min, rr_max)

    for j, (bar, txt) in enumerate(zip(rf_bars, rf_texts)):
        bar.set_width(rf_probs[j])
        bar.set_alpha(1.0 if j == rf_pred else 0.35)
        txt.set_text(f'{rf_probs[j] * 100:.1f}%')
        txt.set_x(rf_probs[j] + 0.01)

    for j, (bar, txt) in enumerate(zip(cnn_bars, cnn_texts)):
        bar.set_width(cnn_prob[j])
        bar.set_alpha(1.0 if j == cnn_pred else 0.35)
        txt.set_text(f'{cnn_prob[j] * 100:.1f}%')
        txt.set_x(cnn_prob[j] + 0.01)

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
        f'  RF: {CLASS_NAMES[rf_pred]}  |  CNN: {CLASS_NAMES[cnn_pred]}'
        f'  ||  Voted: {CLASS_NAMES[voted]}  ')
    pred_label.set_bbox(dict(boxstyle='round,pad=0.3',
                              facecolor=COLORS[voted], alpha=0.85))
    true_label.set_text(f'True: {CLASS_NAMES[true_class]}  ')

    window_count += 1
    if window_count % 20 == 0:
        print(f"Window {window_count:4d} | "
              f"RF: {CLASS_NAMES[rf_pred]:6s} ({rf_probs[rf_pred]*100:.0f}%) | "
              f"CNN: {CLASS_NAMES[cnn_pred]:6s} ({cnn_prob[cnn_pred]*100:.0f}%) | "
              f"Voted: {CLASS_NAMES[voted]:6s} | "
              f"True: {CLASS_NAMES[true_class]:6s} | "
              f"{'✓' if agree else '✗'}")

    plt.pause(SLEEP_TIME)

print(f"\nDone. {window_count} windows processed.")
print(f"RF  — Normal: {rf_counts[0]}  AFib: {rf_counts[1]}")
print(f"CNN — Normal: {cnn_counts[0]}  AFib: {cnn_counts[1]}")
print(f"Agreement: {agree_count}/{window_count} "
      f"({agree_count / window_count * 100:.1f}%)")

plt.ioff()
plt.show()
