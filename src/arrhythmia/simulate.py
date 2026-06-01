# simulate.py
import matplotlib
matplotlib.use('MacOSX')

from pathlib import Path
import numpy as np
import tensorflow as tf
import wfdb
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import butter, filtfilt
import time

ROOT        = Path(__file__).resolve().parents[2]
MODELS_DIR  = ROOT / 'models' / 'arrhythmia'
RECORDS_DIR = ROOT / 'data'   / 'arrhythmia' / 'records'

RECORD_ID   = '208'
FS          = 360
WINDOW      = 280
BEFORE      = 100
AFTER       = WINDOW - BEFORE
DISPLAY_SEC = 6

CLASS_NAMES  = ['Normal (N)', 'Supraventricular (S)',
                'Ventricular (V)', 'Fusion (F)', 'Unknown (Q)']
CLASS_COLORS = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']

print("Loading model...")
model = tf.keras.models.load_model(MODELS_DIR / 'best_model.keras')
print("Model ready.\n")

print(f"Loading MIT-BIH record {RECORD_ID}...")
RECORDS_DIR.mkdir(parents=True, exist_ok=True)
rec_path = RECORDS_DIR / RECORD_ID
if not rec_path.with_suffix('.hea').exists():
    print(f"  Downloading record {RECORD_ID} to {RECORDS_DIR} (one-time)...")
    wfdb.dl_database('mitdb', dl_dir=str(RECORDS_DIR), records=[RECORD_ID])
rec = wfdb.rdrecord(str(rec_path))
ann = wfdb.rdann(str(rec_path), 'atr')


def bandpass(signal, fs=360, lo=0.5, hi=40.0):
    nyq = fs / 2.0
    b, a = butter(4, [lo / nyq, hi / nyq], btype='band')
    return filtfilt(b, a, signal)


ecg     = bandpass(rec.p_signal[:, 0])
r_peaks = ann.sample
symbols = ann.symbol

BEAT_MAP = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,
    'A': 1, 'a': 1, 'J': 1, 'S': 1,
    'V': 2, 'E': 2,
    'F': 3,
    '/': 4, 'f': 4, 'Q': 4
}

plt.ion()
fig = plt.figure(figsize=(14, 8), facecolor='#1a1a2e')
fig.suptitle(f'Live Arrhythmia Detector  —  MIT-BIH Record {RECORD_ID}',
             color='white', fontsize=13, fontweight='bold')

gs = gridspec.GridSpec(3, 2, figure=fig,
                       hspace=0.45, wspace=0.35,
                       left=0.07, right=0.97, top=0.91, bottom=0.07)

ax_ecg   = fig.add_subplot(gs[0, :])
ax_conf  = fig.add_subplot(gs[1, :])
ax_hist  = fig.add_subplot(gs[2, 0])
ax_count = fig.add_subplot(gs[2, 1])

for ax in [ax_ecg, ax_conf, ax_hist, ax_count]:
    ax.set_facecolor('#0f0f1a')
    ax.tick_params(colors='#aaaaaa', labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333355')

disp_len  = DISPLAY_SEC * FS
ecg_disp  = np.zeros(disp_len)
time_axis = np.linspace(0, DISPLAY_SEC, disp_len)

ecg_line, = ax_ecg.plot(time_axis, ecg_disp, color='#00ff88',
                         linewidth=0.7, alpha=0.9)
ax_ecg.set_xlim(0, DISPLAY_SEC)
ax_ecg.set_ylim(-3, 3)
ax_ecg.set_ylabel('Amplitude', color='#aaaaaa', fontsize=8)
ax_ecg.set_xlabel('Time (s)', color='#aaaaaa', fontsize=8)
ax_ecg.set_title('Rolling ECG', color='#cccccc', fontsize=9, pad=4)

bars = ax_conf.barh(CLASS_NAMES, [0] * 5, color=CLASS_COLORS, height=0.55)
ax_conf.set_xlim(0, 1)
ax_conf.set_xlabel('Confidence', color='#aaaaaa', fontsize=8)
ax_conf.set_title('Model confidence — current beat', color='#cccccc',
                   fontsize=9, pad=4)
ax_conf.tick_params(axis='y', labelsize=7.5)

conf_text = [ax_conf.text(0.02, i, '', va='center',
                           color='white', fontsize=7.5, fontweight='bold')
             for i in range(5)]

history_classes = []
history_colors  = []
MAX_HISTORY     = 40
scat = ax_hist.scatter([], [], c=[], s=40, zorder=3)
ax_hist.set_xlim(0, MAX_HISTORY)
ax_hist.set_ylim(-0.5, 4.5)
ax_hist.set_yticks(range(5))
ax_hist.set_yticklabels(CLASS_NAMES, fontsize=6.5)
ax_hist.set_xlabel('Last beats', color='#aaaaaa', fontsize=8)
ax_hist.set_title('Beat history', color='#cccccc', fontsize=9, pad=4)
ax_hist.grid(axis='y', color='#222244', linewidth=0.5)

total_counts = [0] * 5
count_bars   = ax_count.bar(CLASS_NAMES, total_counts,
                             color=CLASS_COLORS, width=0.55)
ax_count.set_ylabel('Count', color='#aaaaaa', fontsize=8)
ax_count.set_title('Total beats classified', color='#cccccc',
                    fontsize=9, pad=4)
ax_count.tick_params(axis='x', labelsize=6, rotation=15)

beat_label = ax_ecg.text(0.01, 0.88, '', transform=ax_ecg.transAxes,
                          color='white', fontsize=11, fontweight='bold',
                          bbox=dict(boxstyle='round,pad=0.3',
                                    facecolor='#333355', alpha=0.7))
true_label = ax_ecg.text(0.99, 0.88, '', transform=ax_ecg.transAxes,
                          color='#aaaaaa', fontsize=8, ha='right',
                          bbox=dict(boxstyle='round,pad=0.3',
                                    facecolor='#1a1a2e', alpha=0.7))

plt.show()

print("Starting real-time simulation... (close the plot window to stop)\n")

beat_count = 0

for peak, sym in zip(r_peaks, symbols):
    if sym not in BEAT_MAP:
        continue
    start = peak - BEFORE
    end   = peak + AFTER
    if start < 0 or end > len(ecg):
        continue

    segment = ecg[start:end].copy()
    segment = (segment - segment.mean()) / (segment.std() + 1e-8)

    inp   = segment.reshape(1, WINDOW, 1).astype(np.float32)
    probs = model.predict(inp, verbose=0)[0]
    pred  = int(np.argmax(probs))
    true  = BEAT_MAP[sym]

    raw_seg  = ecg[start:end]
    ecg_disp = np.roll(ecg_disp, -WINDOW)
    ecg_disp[-WINDOW:] = raw_seg
    ecg_line.set_ydata(ecg_disp)
    ax_ecg.set_ylim(ecg_disp.min() - 0.3, ecg_disp.max() + 0.3)

    for i, (bar, txt) in enumerate(zip(bars, conf_text)):
        bar.set_width(probs[i])
        bar.set_alpha(1.0 if i == pred else 0.35)
        txt.set_text(f'{probs[i] * 100:.1f}%')
        txt.set_x(probs[i] + 0.01)

    history_classes.append(pred)
    history_colors.append(CLASS_COLORS[pred])
    if len(history_classes) > MAX_HISTORY:
        history_classes.pop(0)
        history_colors.pop(0)

    scat.remove()
    scat = ax_hist.scatter(range(len(history_classes)), history_classes,
                            c=history_colors, s=40, zorder=3)

    total_counts[pred] += 1
    for bar, val in zip(count_bars, total_counts):
        bar.set_height(val)
    ax_count.set_ylim(0, max(total_counts) * 1.15 + 1)

    correct = "✓" if pred == true else "✗"
    beat_label.set_text(f'  {CLASS_NAMES[pred]}  {correct}  ')
    beat_label.set_bbox(dict(boxstyle='round,pad=0.3',
                              facecolor=CLASS_COLORS[pred], alpha=0.85))
    true_label.set_text(f'True: {CLASS_NAMES[true]}  ')

    beat_count += 1
    if beat_count % 10 == 0:
        print(f"Beat {beat_count:4d} | Pred: {CLASS_NAMES[pred]:25s} "
              f"| True: {CLASS_NAMES[true]:25s} "
              f"| Conf: {probs[pred] * 100:.1f}%  "
              f"{'✓' if pred == true else '✗'}")

    fig.canvas.draw()
    fig.canvas.flush_events()

    rr_interval = AFTER / FS
    time.sleep(rr_interval * 0.6)

print(f"\nDone. Classified {beat_count} beats from record {RECORD_ID}.")
plt.ioff()
plt.show()
