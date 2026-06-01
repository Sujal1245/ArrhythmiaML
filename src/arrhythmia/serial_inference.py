# serial_inference.py
import matplotlib
matplotlib.use('MacOSX')

from pathlib import Path
import serial
import numpy as np
import tensorflow as tf
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import collections
import time

ROOT       = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / 'models' / 'arrhythmia'

PORT   = '/dev/cu.usbserial-0001'  # ← change to your port
BAUD   = 115200
FS     = 360
WINDOW = 280
BEFORE = 100
AFTER  = WINDOW - BEFORE

ADC_MAX = 4095
V_REF   = 3300

CLASS_NAMES  = ['Normal (N)', 'Supraventricular (S)',
                'Ventricular (V)', 'Fusion (F)', 'Unknown (Q)']
CLASS_COLORS = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']

print("Loading model...")
model = tf.keras.models.load_model(MODELS_DIR / 'best_model.keras')
print("Model ready.\n")


def bandpass(signal, fs=360, lo=0.5, hi=40.0):
    nyq = fs / 2.0
    b, a = butter(4, [lo / nyq, hi / nyq], btype='band')
    return filtfilt(b, a, signal)


class RPeakDetector:
    def __init__(self, fs=360):
        self.fs           = fs
        self.threshold    = None
        self.last_peak    = 0
        self.min_rr       = int(0.2 * fs)
        self.signal_level = 0.0
        self.noise_level  = 0.0

    def detect(self, buf, cursor):
        if cursor < 30:
            return False
        window = buf[max(0, cursor - 10):cursor]
        diff   = np.diff(window)
        energy = np.sum(diff ** 2)
        if self.threshold is None:
            self.signal_level = energy
            self.threshold    = 0.5 * energy
            return False
        if energy > self.threshold:
            self.signal_level = 0.125 * energy + 0.875 * self.signal_level
        else:
            self.noise_level  = 0.125 * energy + 0.875 * self.noise_level
        self.threshold = self.noise_level + \
                         0.25 * (self.signal_level - self.noise_level)
        samples_since_last = cursor - self.last_peak
        if energy > self.threshold and samples_since_last > self.min_rr:
            self.last_peak = cursor
            return True
        return False


plt.ion()
fig = plt.figure(figsize=(14, 8), facecolor='#1a1a2e')
fig.suptitle('Live Arrhythmia Detector  —  ESP32 + AD8232  (USB)',
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

DISP_LEN  = 6 * FS
ecg_disp  = np.zeros(DISP_LEN)
time_axis = np.linspace(0, 6, DISP_LEN)

ecg_line, = ax_ecg.plot(time_axis, ecg_disp, color='#00ff88',
                          linewidth=0.7, alpha=0.9)
ax_ecg.set_xlim(0, 6)
ax_ecg.set_ylim(-500, 3600)
ax_ecg.set_ylabel('ADC value', color='#aaaaaa', fontsize=8)
ax_ecg.set_xlabel('Time (s)',  color='#aaaaaa', fontsize=8)
ax_ecg.set_title('Live ECG from AD8232', color='#cccccc', fontsize=9, pad=4)

bars = ax_conf.barh(CLASS_NAMES, [0] * 5, color=CLASS_COLORS, height=0.55)
ax_conf.set_xlim(0, 1)
ax_conf.set_xlabel('Confidence', color='#aaaaaa', fontsize=8)
ax_conf.set_title('Model confidence — current beat',
                   color='#cccccc', fontsize=9, pad=4)
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
ax_count.set_ylabel('Count',  color='#aaaaaa', fontsize=8)
ax_count.set_title('Total beats classified', color='#cccccc',
                    fontsize=9, pad=4)
ax_count.tick_params(axis='x', labelsize=6, rotation=15)

beat_label = ax_ecg.text(
    0.01, 0.88, 'Waiting for signal...', transform=ax_ecg.transAxes,
    color='white', fontsize=11, fontweight='bold',
    bbox=dict(boxstyle='round,pad=0.3', facecolor='#333355', alpha=0.7))

status_text = ax_ecg.text(
    0.99, 0.88, 'Connecting...', transform=ax_ecg.transAxes,
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
    import serial.tools.list_ports
    for p in serial.tools.list_ports.comports():
        print(f"  {p.device}  —  {p.description}")
    exit(1)

BUFFER_SIZE = FS * 10
raw_buf     = np.zeros(BUFFER_SIZE, dtype=np.float32)
cursor      = 0
detector    = RPeakDetector(fs=FS)
beat_count  = 0
lost_frames = 0

print("Reading ECG... (Ctrl+C to stop)\n")

try:
    while True:
        try:
            line = ser.readline().decode('utf-8').strip()
        except UnicodeDecodeError:
            continue

        if line == 'READY' or line == '':
            continue

        if line == 'X':
            status_text.set_text('⚠ Check electrodes')
            lost_frames += 1
            fig.canvas.flush_events()
            continue

        try:
            value = float(line)
        except ValueError:
            continue

        raw_buf[cursor % BUFFER_SIZE] = value
        cursor += 1

        idx = cursor % BUFFER_SIZE
        if idx >= DISP_LEN:
            ecg_disp = raw_buf[idx - DISP_LEN:idx]
        else:
            ecg_disp = np.concatenate([
                raw_buf[BUFFER_SIZE - (DISP_LEN - idx):],
                raw_buf[:idx]
            ])

        ecg_line.set_ydata(ecg_disp)

        if cursor % 18 == 0:
            status_text.set_text(f'● LIVE  |  Beats: {beat_count}')
            fig.canvas.draw()
            fig.canvas.flush_events()

        if not detector.detect(raw_buf, cursor % BUFFER_SIZE):
            continue

        if cursor < BEFORE + AFTER:
            continue

        buf_idx = cursor % BUFFER_SIZE
        start   = buf_idx - BEFORE
        end     = buf_idx + AFTER

        if start >= 0 and end < BUFFER_SIZE:
            segment = raw_buf[start:end].copy()
        else:
            indices = [(start + i) % BUFFER_SIZE for i in range(WINDOW)]
            segment = raw_buf[indices]

        if len(segment) == WINDOW:
            try:
                segment = bandpass(segment.astype(np.float64))
                segment = (segment - segment.mean()) / (segment.std() + 1e-8)
            except Exception:
                continue
        else:
            continue

        inp   = segment.reshape(1, WINDOW, 1).astype(np.float32)
        probs = model.predict(inp, verbose=0)[0]
        pred  = int(np.argmax(probs))

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
        scat = ax_hist.scatter(range(len(history_classes)),
                                history_classes,
                                c=history_colors, s=40, zorder=3)

        total_counts[pred] += 1
        for bar, val in zip(count_bars, total_counts):
            bar.set_height(val)
        ax_count.set_ylim(0, max(total_counts) * 1.15 + 1)

        beat_label.set_text(f'  {CLASS_NAMES[pred]}  ')
        beat_label.set_bbox(dict(boxstyle='round,pad=0.3',
                                  facecolor=CLASS_COLORS[pred], alpha=0.85))

        beat_count += 1
        print(f"Beat {beat_count:4d} | {CLASS_NAMES[pred]:25s} "
              f"| Conf: {probs[pred] * 100:.1f}%")

except KeyboardInterrupt:
    print(f"\nStopped. Classified {beat_count} beats.")
finally:
    ser.close()
    plt.ioff()
    plt.show()
