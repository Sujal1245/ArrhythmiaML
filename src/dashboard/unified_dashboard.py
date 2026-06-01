# unified_dashboard.py
#
# Unified simulation dashboard:
#   - Arrhythmia beat classification  (MIT-BIH, 5-class CNN)
#   - AFib detection                  (AFDB, RF + 1D-CNN ensemble)
#   - Simulated body temperature
#
# Use ◀ ▶ buttons at the bottom to switch records for each sub-system.
# Requires mitbih_processed.npz and afib_dataset.npz to contain 'record_ids'.
# Run both build_dataset.py scripts once if the key is missing.
#
# Run: python src/dashboard/unified_dashboard.py

import matplotlib
matplotlib.use('MacOSX')

from pathlib import Path
import collections
import random
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button
from matplotlib.animation import FuncAnimation

# ── Paths ─────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / 'models'
DATA_DIR   = ROOT / 'data'

# ── Config ─────────────────────────────────────────────────────────────────
FS           = 360
WIN_ECG      = 280
BEFORE       = 100
AFTER        = WIN_ECG - BEFORE
DISPLAY_SEC  = 6
AFIB_EVERY   = 15       # advance one AFib window every N arrhythmia beats
SLEEP_SCALE  = 0.15     # speed relative to real cardiac timing

BEAT_MAP = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,
    'A': 1, 'a': 1, 'J': 1, 'S': 1,
    'V': 2, 'E': 2,
    'F': 3,
    '/': 4, 'f': 4, 'Q': 4
}

ARR_NAMES   = ['Normal (N)', 'Supraventricular (S)',
               'Ventricular (V)', 'Fusion (F)', 'Unknown (Q)']
ARR_COLORS  = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']
AFIB_NAMES  = ['Normal', 'AFib']
AFIB_COLORS = ['#2ecc71', '#e74c3c']
VOTE_WINDOW = 5   # rolling window size
VOTE_THRESH = 3   # AFib votes needed to confirm
AFIB_THRESHOLD = 0.846  # Optimal threshold from evaluation

class TemporalVoter:
    def __init__(self, window=VOTE_WINDOW, thresh=VOTE_THRESH):
        self.buf    = collections.deque(maxlen=window)
        self.thresh = thresh

    def update(self, pred):
        self.buf.append(pred)
        return int(sum(self.buf) >= self.thresh)

    def reset(self):
        self.buf.clear()

TEMP_NORMAL  = 36.8
TEMP_FEVER   = 37.5
TEMP_MIN     = 36.0
TEMP_MAX     = 38.5
TEMP_HISTORY = 60

# ── Load models ────────────────────────────────────────────────────────────
print("Loading arrhythmia model...")
arr_model = tf.keras.models.load_model(MODELS_DIR / 'arrhythmia' / 'best_model.keras')

print("Loading AFib hybrid model...")
afib_hybrid = tf.keras.models.load_model(MODELS_DIR / 'afib' / 'best_afib_hybrid.keras')
afib_hsc    = joblib.load(MODELS_DIR / 'afib' / 'afib_hybrid_scaler.pkl')
h_rr_stats  = np.load(MODELS_DIR / 'afib' / 'afib_hybrid_rr_stats.npy')
print("All models loaded.\n")

# ── Load datasets ──────────────────────────────────────────────────────────
print("Loading arrhythmia dataset...")
_arr = np.load(DATA_DIR / 'arrhythmia' / 'mitbih_processed.npz')
if 'record_ids' not in _arr:
    raise SystemExit(
        "\nERROR: 'record_ids' missing from mitbih_processed.npz.\n"
        "Re-run:  python src/arrhythmia/build_dataset.py"
    )
full_arr_X   = _arr['X']
full_arr_y   = _arr['y']
full_arr_ids = _arr['record_ids']
ARR_RECORDS  = sorted(np.unique(full_arr_ids).tolist(), key=int)

print("Loading AFib dataset...")
_afib = np.load(DATA_DIR / 'afib' / 'afib_dataset.npz')
if 'record_ids' not in _afib:
    raise SystemExit(
        "\nERROR: 'record_ids' missing from afib_dataset.npz.\n"
        "Re-run:  python src/afib/build_dataset.py"
    )
full_feat     = _afib['X_feat']
full_seq      = _afib['X_seq']
full_afib_y   = _afib['y']
full_afib_ids = _afib['record_ids']
AFIB_RECORDS  = sorted(np.unique(full_afib_ids).tolist())

print(f"Arrhythmia : {len(full_arr_y):,} beats  across {len(ARR_RECORDS)} records")
print(f"AFib       : {len(full_afib_y):,} windows across {len(AFIB_RECORDS)} records\n")

# ── Temperature simulation ─────────────────────────────────────────────────
_temp = TEMP_NORMAL
def next_temp():
    global _temp
    _temp += random.gauss(0, 0.03)
    _temp = max(TEMP_MIN, min(TEMP_MAX, _temp))
    return _temp
temp_history = [TEMP_NORMAL] * TEMP_HISTORY
# ── Mutable simulation state ───────────────────────────────────────────────
scenario_idx = 0
scenario_filter = 'All'

# Screening State
is_screening = False
screening_start_time = 0
SCREENING_DURATION = 120  # seconds
screening_results = {
    'arr_counts': [0]*5,
    'afib_windows': 0,
    'afib_detections': 0,
    'max_temp': 0,
    'min_temp': 100
}

arr_rec_idx  = 0
# ... rest of state ...
# Categorize AFib records for filtering
def get_afib_scenarios():
    scenarios = []
    for rec_id in AFIB_RECORDS:
        mask = full_afib_ids == rec_id
        labels = full_afib_y[mask]
        n_afib = np.sum(labels == 1)
        total = len(labels)
        ratio = n_afib / total
        
        if ratio > 0.9:
            cat = 'AFib-Predominant'
        elif ratio < 0.05:
            cat = 'Normal-Baseline'
        else:
            cat = 'Mixed'
        
        scenarios.append({
            'id': rec_id,
            'cat': cat,
            'stats': f'N:{total-n_afib} A:{n_afib}'
        })
    return scenarios

ALL_SCENARIOS = get_afib_scenarios()
filtered_indices = list(range(len(ALL_SCENARIOS)))

arr_rec_idx  = 0
afib_rec_idx = 0

arr_X = arr_y_cur = None
arr_beat_idx = 0

afib_feat_cur = afib_seq_cur = afib_y_cur = None
afib_win_idx  = 0
beat_count    = 0

afib_voter = TemporalVoter()

pending_arr_reload  = True   # triggers the initial record load
pending_afib_reload = True

# Display buffers
disp_len  = DISPLAY_SEC * FS
ecg_disp  = np.zeros(disp_len)
time_axis = np.linspace(0, DISPLAY_SEC, disp_len)
rr_disp   = np.ones(60) * 800.0

# Running counters / history
arr_counts  = [0] * 5
afib_counts = [0, 0]
MAX_BHIST = 40
MAX_AHIST = 50
bhist_cls = []; bhist_col = []
ahist_hybrid = []

# ── Figure layout (4 rows × 3 cols + control strip) ───────────────────────
fig = plt.figure(figsize=(18, 11), facecolor='#1a1a2e')
fig.suptitle('Unified Cardiac Monitor  —  Arrhythmia · AFib · Temperature',
             color='white', fontsize=13, fontweight='bold')

gs = gridspec.GridSpec(
    4, 3, figure=fig,
    hspace=0.52, wspace=0.32,
    left=0.06, right=0.97, top=0.93, bottom=0.13
)

ax_ecg   = fig.add_subplot(gs[0, :])
ax_conf  = fig.add_subplot(gs[1, :])
ax_rr    = fig.add_subplot(gs[2, :2])
ax_temp  = fig.add_subplot(gs[2, 2])
ax_bhist = fig.add_subplot(gs[3, 0])
ax_ahist = fig.add_subplot(gs[3, 1])
ax_count = fig.add_subplot(gs[3, 2])

for ax in [ax_ecg, ax_conf, ax_rr, ax_temp, ax_bhist, ax_ahist, ax_count]:
    ax.set_facecolor('#0f0f1a')
    ax.tick_params(colors='#aaaaaa', labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333355')

# ── Row 0: Rolling ECG ────────────────────────────────────────────────────
ecg_line, = ax_ecg.plot(time_axis, ecg_disp, color='#00ff88', linewidth=0.7, alpha=0.9)
ax_ecg.set_xlim(0, DISPLAY_SEC)
ax_ecg.set_ylim(-3, 3)
ax_ecg.set_ylabel('Amplitude (norm.)', color='#aaaaaa', fontsize=8)
ax_ecg.set_xlabel('Time (s, approx.)', color='#aaaaaa', fontsize=8)
ax_ecg.set_title('Rolling ECG  (stitched beat windows)',
                  color='#cccccc', fontsize=9, pad=4)

beat_overlay = ax_ecg.text(
    0.01, 0.86, '', transform=ax_ecg.transAxes,
    color='white', fontsize=10, fontweight='bold',
    bbox=dict(boxstyle='round,pad=0.3', facecolor='#333355', alpha=0.8))
true_overlay = ax_ecg.text(
    0.99, 0.86, '', transform=ax_ecg.transAxes,
    color='#aaaaaa', fontsize=8, ha='right',
    bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e', alpha=0.7))

# ── Row 1: Beat confidence bars ───────────────────────────────────────────
arr_bars = ax_conf.barh(ARR_NAMES, [0]*5, color=ARR_COLORS, height=0.55)
ax_conf.set_xlim(0, 1)
ax_conf.set_xlabel('Confidence', color='#aaaaaa', fontsize=8)
ax_conf.set_title('Beat classification confidence (5-class CNN)',
                   color='#cccccc', fontsize=9, pad=4)
ax_conf.tick_params(axis='y', labelsize=7.5)
arr_txts = [ax_conf.text(0.02, i, '', va='center',
                          color='white', fontsize=7.5, fontweight='bold')
            for i in range(5)]

# ── Row 2 left: Rolling RR intervals ─────────────────────────────────────
rr_line, = ax_rr.plot(range(60), rr_disp, color='#00ff88', linewidth=1.0)
ax_rr.set_xlim(0, 60)
ax_rr.set_ylim(300, 1500)
ax_rr.set_ylabel('RR interval (ms)', color='#aaaaaa', fontsize=8)
ax_rr.set_xlabel('Beats', color='#aaaaaa', fontsize=8)
ax_rr.set_title('Rolling RR intervals  (AFib module)', color='#cccccc', fontsize=9, pad=4)
ax_rr.axhline(600,  color='#e74c3c', linewidth=0.5, linestyle='--',
              alpha=0.4, label='100 bpm')
ax_rr.axhline(1000, color='#3498db', linewidth=0.5, linestyle='--',
              alpha=0.4, label='60 bpm')
ax_rr.legend(fontsize=7, loc='upper right', labelcolor='#aaaaaa', facecolor='#1a1a2e')
afib_overlay = ax_rr.text(
    0.01, 0.86, 'Waiting...', transform=ax_rr.transAxes,
    color='white', fontsize=9, fontweight='bold',
    bbox=dict(boxstyle='round,pad=0.3', facecolor='#333355', alpha=0.8))

# ── Row 2 right: Temperature ──────────────────────────────────────────────
temp_line, = ax_temp.plot(range(TEMP_HISTORY), temp_history,
                           color='#ff9f43', linewidth=1.2)
ax_temp.set_xlim(0, TEMP_HISTORY - 1)
ax_temp.set_ylim(35.5, 39.0)
ax_temp.set_ylabel('°C', color='#aaaaaa', fontsize=8)
ax_temp.set_xlabel('Readings', color='#aaaaaa', fontsize=8)
ax_temp.set_title('Body Temperature', color='#cccccc', fontsize=9, pad=4)
ax_temp.axhline(TEMP_FEVER, color='#e74c3c', linewidth=0.8,
                linestyle='--', alpha=0.7, label='Fever (37.5°C)')
ax_temp.legend(fontsize=7, labelcolor='#aaaaaa', facecolor='#1a1a2e')
temp_txt = ax_temp.text(
    0.97, 0.88, f'{TEMP_NORMAL:.1f}°C', transform=ax_temp.transAxes,
    color='#ff9f43', fontsize=12, fontweight='bold', ha='right',
    bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e', alpha=0.8))

# ── Row 3 left: Beat history ──────────────────────────────────────────────
bscat = ax_bhist.scatter([], [], c=[], s=35, zorder=3)
ax_bhist.set_xlim(0, MAX_BHIST)
ax_bhist.set_ylim(-0.5, 4.5)
ax_bhist.set_yticks(range(5))
ax_bhist.set_yticklabels(ARR_NAMES, fontsize=6)
ax_bhist.set_xlabel('Last beats', color='#aaaaaa', fontsize=8)
ax_bhist.set_title('Arrhythmia beat history', color='#cccccc', fontsize=9, pad=4)
ax_bhist.grid(axis='y', color='#222244', linewidth=0.5)

# ── Row 3 mid: AFib prediction history ───────────────────────────────────
ascat_hybrid = ax_ahist.scatter([], [], c=[], s=30, zorder=3)
ax_ahist.set_xlim(0, MAX_AHIST)
ax_ahist.set_ylim(-0.5, 1.5)
ax_ahist.set_yticks([0, 1])
ax_ahist.set_yticklabels(AFIB_NAMES, fontsize=8)
ax_ahist.set_xlabel('Last windows', color='#aaaaaa', fontsize=8)
ax_ahist.set_title('AFib history (Hybrid Model)', color='#cccccc', fontsize=9, pad=4)
ax_ahist.grid(axis='y', color='#222244', linewidth=0.5)

# ── Row 3 right: Running totals ───────────────────────────────────────────
x5 = np.arange(5)
arr_cbars = ax_count.bar(x5, arr_counts, color=ARR_COLORS, width=0.6)
ax_count.set_xticks(x5)
ax_count.set_xticklabels(['N', 'S', 'V', 'F', 'Q'], fontsize=8)
ax_count.set_ylabel('Count', color='#aaaaaa', fontsize=8)
ax_count.set_title('Running totals', color='#cccccc', fontsize=9, pad=4)
ax_count.grid(axis='y', color='#222244', linewidth=0.5)
afib_total_txt = ax_count.text(
    0.98, 0.95, '', transform=ax_count.transAxes,
    color='#aaaaaa', fontsize=7, ha='right', va='top')

# ── Control strip ─────────────────────────────────────────────────────────
fig.add_artist(plt.Line2D([0.03, 0.97], [0.105, 0.105],
                           transform=fig.transFigure,
                           color='#333355', linewidth=0.8))

# Unified record selector
_BTN_H = 0.05;  _BTN_Y = 0.025
ax_prev = fig.add_axes([0.31, _BTN_Y, 0.045, _BTN_H])
ax_next = fig.add_axes([0.645, _BTN_Y, 0.045, _BTN_H])
btn_prev = Button(ax_prev, '◀', color='#1a1a3e', hovercolor='#333366')
btn_next = Button(ax_next, '▶', color='#1a1a3e', hovercolor='#333366')
btn_prev.label.set_color('white')
btn_next.label.set_color('white')
for _ax in (ax_prev, ax_next):
    for sp in _ax.spines.values():
        sp.set_edgecolor('#555588')

scenario_label_txt = fig.text(0.5, 0.058, '',
                               ha='center', va='center',
                               color='white', fontsize=10, fontweight='bold')
fig.text(0.5, 0.030, 'Switch Patient Scenario (Synchronized Records)',
         ha='center', va='center', color='#666688', fontsize=8)

# Scenario Filter Button
ax_filter = fig.add_axes([0.80, _BTN_Y, 0.14, _BTN_H])
btn_filter = Button(ax_filter, 'Filter: All', color='#1a1a3e', hovercolor='#333366')
btn_filter.label.set_color('#00ff88')

# Start Screening Button
ax_screen = fig.add_axes([0.06, _BTN_Y, 0.12, _BTN_H])
btn_screen = Button(ax_screen, 'START SCREENING', color='#2ecc71', hovercolor='#27ae60')
btn_screen.label.set_color('white')
btn_screen.label.set_fontweight('bold')

for sp in list(ax_filter.spines.values()) + list(ax_screen.spines.values()):
    sp.set_edgecolor('#555588')

# Screening Overlay (Invisible by default)
screen_overlay = fig.text(
    0.5, 0.5, '',
    ha='center', va='center', fontsize=20, fontweight='bold',
    color='white', bbox=dict(boxstyle='round,pad=1', facecolor='#1a1a2e', alpha=0.9, edgecolor='#00ff88'),
    visible=False, zorder=10
)

# ── Reset / load helpers ───────────────────────────────────────────────────
def _reset_arr_display():
    global beat_count, bscat
    ecg_disp[:] = 0.0
    ecg_line.set_ydata(ecg_disp)
    ax_ecg.set_ylim(-3, 3)
    for i in range(5):
        arr_counts[i] = 0
        arr_cbars[i].set_height(0)
        arr_bars[i].set_width(0)
        arr_txts[i].set_text('')
    ax_count.set_ylim(0, 10)
    bhist_cls.clear(); bhist_col.clear()
    bscat.remove()
    bscat = ax_bhist.scatter([], [], c=[], s=35, zorder=3)
    beat_count = 0
    beat_overlay.set_text('')
    true_overlay.set_text('')


def _reset_afib_display():
    global ascat_hybrid
    rr_disp[:] = 800.0
    rr_line.set_ydata(rr_disp)
    ax_rr.set_ylim(300, 1500)
    afib_counts[0] = 0; afib_counts[1] = 0
    ahist_hybrid.clear()
    ascat_hybrid.remove()
    ascat_hybrid = ax_ahist.scatter([], [], c=[], s=30, zorder=3)
    afib_overlay.set_text('  Waiting...  ')
    afib_overlay.set_bbox(dict(boxstyle='round,pad=0.3', facecolor='#333355', alpha=0.8))
    afib_total_txt.set_text('')
    afib_voter.reset()


def load_scenario(f_idx):
    global scenario_idx, arr_rec_idx, afib_rec_idx
    global pending_arr_reload, pending_afib_reload
    
    if not filtered_indices:
        print("No scenarios match the current filter.")
        return

    scenario_idx = f_idx % len(filtered_indices)
    actual_afib_idx = filtered_indices[scenario_idx]
    
    # Arrhythmia record just cycles
    arr_rec_idx = actual_afib_idx % len(ARR_RECORDS)
    afib_rec_idx = actual_afib_idx
    
    pending_arr_reload = True
    pending_afib_reload = True
    
    scen = ALL_SCENARIOS[afib_rec_idx]
    scenario_label_txt.set_text(f'Scenario {scenario_idx+1}/{len(filtered_indices)}:  '
                                 f'AFib {scen["id"]} [{scen["cat"]}]  |  '
                                 f'Arrhythmia {ARR_RECORDS[arr_rec_idx]}')
    print(f"Loaded scenario {scenario_idx+1}: AFib {scen['id']} ({scen['cat']})")


def load_arr_record(rec_id):
    global arr_X, arr_y_cur, arr_beat_idx
    mask      = full_arr_ids == rec_id
    arr_X     = full_arr_X[mask]
    arr_y_cur = full_arr_y[mask]
    arr_beat_idx = 0
    _reset_arr_display()
    print(f"[ARR ] Switched to record {rec_id} — {len(arr_X):,} beats")


def load_afib_record(rec_id):
    global afib_feat_cur, afib_seq_cur, afib_y_cur, afib_win_idx
    mask          = full_afib_ids == rec_id
    afib_feat_cur = full_feat[mask]
    afib_seq_cur  = full_seq[mask]
    afib_y_cur    = full_afib_y[mask]
    afib_win_idx  = 0
    _reset_afib_display()
    n_norm = int(np.sum(afib_y_cur == 0))
    n_afib = int(np.sum(afib_y_cur == 1))
    print(f"[AFIB] Switched to record {rec_id} — "
          f"{len(afib_y_cur)} windows  Normal:{n_norm}  AFib:{n_afib}")


# ── Button callbacks ───────────────────────────────────────────────────────
def on_prev(_):
    load_scenario(scenario_idx - 1)

def on_next(_):
    load_scenario(scenario_idx + 1)

def on_filter_clicked(_):
    global scenario_filter, filtered_indices
    cats = ['All', 'AFib-Predominant', 'Normal-Baseline', 'Mixed']
    cur_idx = cats.index(scenario_filter)
    scenario_filter = cats[(cur_idx + 1) % len(cats)]
    
    if scenario_filter == 'All':
        filtered_indices = list(range(len(ALL_SCENARIOS)))
    else:
        filtered_indices = [i for i, s in enumerate(ALL_SCENARIOS) if s['cat'] == scenario_filter]
    
    btn_filter.label.set_text(f'Filter: {scenario_filter}')
    load_scenario(0)

import time

def on_screen_clicked(_):
    global is_screening, screening_start_time, screening_results
    if not is_screening:
        is_screening = True
        screening_start_time = time.time()
        screening_results = {
            'arr_counts': [0]*5,
            'afib_windows': 0,
            'afib_detections': 0,
            'max_temp': -100,
            'min_temp': 100
        }
        btn_screen.label.set_text('CANCELING...')
        btn_screen.color = '#e74c3c'
        screen_overlay.set_visible(True)
    else:
        is_screening = False
        btn_screen.label.set_text('START SCREENING')
        btn_screen.color = '#2ecc71'
        screen_overlay.set_visible(False)

def on_overlay_click(event):
    if screen_overlay.get_visible() and not is_screening:
        screen_overlay.set_visible(False)
        fig.canvas.draw_idle()

btn_prev.on_clicked(on_prev)
btn_next.on_clicked(on_next)
btn_filter.on_clicked(on_filter_clicked)
btn_screen.on_clicked(on_screen_clicked)
fig.canvas.mpl_connect('button_press_event', on_overlay_click)

# ── Animation update ───────────────────────────────────────────────────────
def animate(_):
    global arr_beat_idx, afib_win_idx, beat_count
    global pending_arr_reload, pending_afib_reload
    global ecg_disp, rr_disp
    global bscat, ascat_hybrid
    global is_screening, screening_results

    # ── Handle Screening Timer & UI ──────────────────────────────────────
    if is_screening:
        elapsed = time.time() - screening_start_time
        remaining = max(0, SCREENING_DURATION - elapsed)
        
        screen_overlay.set_text(f"SCREENING IN PROGRESS\n\n{int(remaining)}s Remaining\n\nDO NOT MOVE")
        
        if elapsed >= SCREENING_DURATION:
            is_screening = False
            btn_screen.label.set_text('START SCREENING')
            btn_screen.color = '#2ecc71'
            
            # Generate Verdict
            res = screening_results
            abnormal_arr = sum(res['arr_counts'][1:])
            afib_ratio = res['afib_detections'] / max(1, res['afib_windows'])
            
            verdict = "🟢 NORMAL RHYTHM"
            color = "#2ecc71"
            advice = "No immediate action required."
            
            # Logic: If >10% AFib detected OR >5 PVCs (V) OR >15 abnormal beats total
            if afib_ratio > 0.1 or res['arr_counts'][2] > 5 or abnormal_arr > 15:
                verdict = "🔴 IRREGULAR RHYTHM DETECTED"
                color = "#e74c3c"
                advice = "Please consult a cardiologist for a proper ECG."
            elif abnormal_arr > 5:
                verdict = "🟡 MINOR IRREGULARITIES"
                color = "#f39c12"
                advice = "Monitor symptoms and re-test if needed."

            report = (f"--- SCREENING REPORT ---\n\n"
                      f"RESULT: {verdict}\n\n"
                      f"AFib Burden: {afib_ratio*100:.1f}%\n"
                      f"Arrhythmia Beats: {abnormal_arr}\n"
                      f"Temp Range: {res['min_temp']:.1f}-{res['max_temp']:.1f} °C\n\n"
                      f"ADVICE: {advice}\n\n"
                      f"Click anywhere to close.")
            screen_overlay.set_text(report)
            screen_overlay.get_bbox_patch().set_edgecolor(color)
            return

    # ── Handle pending record switches ────────────────────────────────────
    if pending_arr_reload:
        load_arr_record(ARR_RECORDS[arr_rec_idx])
        pending_arr_reload = False

    if pending_afib_reload:
        load_afib_record(AFIB_RECORDS[afib_rec_idx])
        pending_afib_reload = False

    if arr_X is None or len(arr_X) == 0:
        return

    # Auto-loop when record is exhausted
    if arr_beat_idx >= len(arr_X):
        arr_beat_idx = 0

    # ── Arrhythmia inference ──────────────────────────────────────────────
    segment = arr_X[arr_beat_idx]
    true    = int(arr_y_cur[arr_beat_idx])
    arr_beat_idx += 1

    inp   = segment.reshape(1, WIN_ECG, 1).astype(np.float32)
    probs = arr_model.predict(inp, verbose=0)[0]
    pred  = int(np.argmax(probs))

    # Rolling ECG (stitched normalized beat windows)
    ecg_disp = np.roll(ecg_disp, -WIN_ECG)
    ecg_disp[-WIN_ECG:] = segment
    ecg_line.set_ydata(ecg_disp)
    ax_ecg.set_ylim(ecg_disp.min() - 0.3, ecg_disp.max() + 0.3)

    # Confidence bars
    for i, (bar, txt) in enumerate(zip(arr_bars, arr_txts)):
        bar.set_width(probs[i])
        bar.set_alpha(1.0 if i == pred else 0.35)
        txt.set_text(f'{probs[i]*100:.1f}%')
        txt.set_x(probs[i] + 0.01)

    if is_screening:
        screening_results['arr_counts'][pred] += 1

    correct = '✓' if pred == true else '✗'
    beat_overlay.set_text(f'  {ARR_NAMES[pred]}  {correct}  ')
    beat_overlay.set_bbox(dict(boxstyle='round,pad=0.3',
                                facecolor=ARR_COLORS[pred], alpha=0.85))
    true_overlay.set_text(f'True: {ARR_NAMES[true]}  ')

    bhist_cls.append(pred); bhist_col.append(ARR_COLORS[pred])
    if len(bhist_cls) > MAX_BHIST:
        bhist_cls.pop(0); bhist_col.pop(0)
    bscat.remove()
    bscat = ax_bhist.scatter(range(len(bhist_cls)), bhist_cls,
                              c=bhist_col, s=35, zorder=3)

    arr_counts[pred] += 1
    for bar, val in zip(arr_cbars, arr_counts):
        bar.set_height(val)
    ax_count.set_ylim(0, max(arr_counts) * 1.2 + 1)

    # ── Temperature ───────────────────────────────────────────────────────
    t = next_temp()
    temp_history.append(t); temp_history.pop(0)
    temp_line.set_ydata(temp_history)
    temp_txt.set_text(f'{t:.1f}°C')
    temp_txt.set_color('#e74c3c' if t >= TEMP_FEVER else '#ff9f43')
    
    if is_screening:
        screening_results['max_temp'] = max(screening_results['max_temp'], t)
        screening_results['min_temp'] = min(screening_results['min_temp'], t)

    # ── AFib update every AFIB_EVERY beats ───────────────────────────────
    beat_count += 1
    if (beat_count % AFIB_EVERY == 0
            and afib_feat_cur is not None
            and len(afib_y_cur) > 0):

        if afib_win_idx >= len(afib_y_cur):
            afib_win_idx = 0

        feat      = afib_feat_cur[afib_win_idx]
        window_rr = afib_seq_cur[afib_win_idx]
        true_afib = int(afib_y_cur[afib_win_idx])
        afib_win_idx += 1

        # Hybrid Model Inference
        feat_sc = afib_hsc.transform(feat.reshape(1, -1))
        seq     = (window_rr - h_rr_stats[0]) / h_rr_stats[1]
        seq     = seq.reshape(1, 30, 1).astype(np.float32)
        
        probs = afib_hybrid.predict([seq, feat_sc], verbose=0)[0]
        # Use tuned threshold (0.846) for AFib class (index 1)
        pred  = 1 if probs[1] >= AFIB_THRESHOLD else 0
        
        if is_screening:
            screening_results['afib_windows'] += 1
            if pred == 1:
                screening_results['afib_detections'] += 1

        # Rolling RR
        rr_disp = np.roll(rr_disp, -1)
        rr_disp[-1] = window_rr[-1]
        rr_line.set_ydata(rr_disp)
        ax_rr.set_ylim(max(200,  rr_disp.min() - 100),
                       min(2000, rr_disp.max() + 100))

        # AFib overlay
        voted = afib_voter.update(pred)

        afib_overlay.set_text(
            f'  Hybrid: {AFIB_NAMES[pred]} ({probs[pred]*100:.0f}%)  '
            f'  ||  Voted: {AFIB_NAMES[voted]}  '
            f'  True: {AFIB_NAMES[true_afib]}  ')
        afib_overlay.set_bbox(dict(boxstyle='round,pad=0.3',
                                    facecolor=AFIB_COLORS[voted], alpha=0.85))

        # AFib history scatter
        ahist_hybrid.append((pred, AFIB_COLORS[pred]))
        if len(ahist_hybrid) > MAX_AHIST:
            ahist_hybrid.pop(0)
        
        ascat_hybrid.remove()
        ascat_hybrid = ax_ahist.scatter(
            range(len(ahist_hybrid)), [h[0] for h in ahist_hybrid],
            c=[h[1] for h in ahist_hybrid], s=30, zorder=3)

        afib_counts[pred] += 1
        afib_total_txt.set_text(
            f'AFib Hybrid — N:{afib_counts[0]}  A:{afib_counts[1]}')


# ── Run ────────────────────────────────────────────────────────────────────
interval_ms = max(50, int(AFTER / FS * SLEEP_SCALE * 1000))
ani = FuncAnimation(fig, animate, interval=interval_ms, cache_frame_data=False)

print("Dashboard running.")
print(f"  Arrhythmia records : {', '.join(ARR_RECORDS)}")
print(f"  AFib records       : {', '.join(AFIB_RECORDS)}")
print("  Use ◀ ▶ buttons to switch records. Use Filter button to categorize. Close window to stop.\n")

plt.show()
