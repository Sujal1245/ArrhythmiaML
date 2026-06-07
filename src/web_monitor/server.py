# src/web_monitor/server.py

import os
import sys
import time
import json
import asyncio
import threading
import collections
from pathlib import Path

import serial
import numpy as np
import tensorflow as tf
import joblib
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import socketio

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
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
print("✓ All models loaded.")

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
class GlobalState:
    def __init__(self):
        self.ecg_buffer = np.zeros(FS_ECG * 10)
        self.ecg_cursor = 0
        self.ecg_last_processed = 0
        self.ppg_buffer = np.zeros(FS_PPG * 10)
        self.ppg_cursor = 0
        self.current_temp = 36.5
        self.rr_buffer = collections.deque(maxlen=AFIB_WIN_SIZE)
        self.last_ppg_peak_time = time.time()
        self.beat_count_ppg = 0
        self.is_screening = False
        self.screening_start_time = 0
        self.screening_results = {}
        self.ecg_detector = ECGPeakDetector(FS_ECG)
        self.ppg_detector = PPGPeakDetector(FS_PPG)

state = GlobalState()

# ── FastAPI & Socket.io ────────────────────────────────────────────────────
app = FastAPI()
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')
socket_app = socketio.ASGIApp(sio, app)

main_loop = None

@app.on_event("startup")
async def startup_event():
    global main_loop
    main_loop = asyncio.get_event_loop()
    threading.Thread(target=serial_worker, daemon=True).start()

def emit_sio(event, data):
    if main_loop:
        main_loop.call_soon_threadsafe(lambda: asyncio.create_task(sio.emit(event, data)))

@app.get("/status")
async def get_status():
    return {"status": "running", "port": PORT, "is_screening": state.is_screening}

@sio.event
async def connect(sid, environ):
    print(f"Client connected: {sid}")

@sio.event
async def start_screening(sid):
    state.is_screening = True
    state.screening_start_time = time.time()
    state.screening_results = {
        'arr_counts': [0]*5,
        'afib_windows': 0,
        'afib_detections': 0,
        'total_beats': 0,
        'max_temp': 0.0
    }
    print("Screening started")
    await sio.emit('screening_status', {'active': True})

@sio.event
async def cancel_screening(sid):
    state.is_screening = False
    print("Screening cancelled")
    await sio.emit('screening_status', {'active': False})

# ── Serial Reader Thread ───────────────────────────────────────────────────
def serial_worker():
    print(f"Connecting to {PORT}...")
    try:
        ser = serial.Serial(PORT, BAUD, timeout=0.1)
        time.sleep(2)
        ser.reset_input_buffer()
        print("✓ Serial Connected.")
    except Exception as e:
        print(f"✗ Serial Error: {e}")
        return

    while True:
        if ser.in_waiting > 0:
            try:
                line = ser.readline().decode('utf-8').strip()
                if not line or ':' not in line: continue
                prefix, val_str = line.split(':')
                val = float(val_str)
                
                if prefix == 'E':
                    state.ecg_buffer[state.ecg_cursor % len(state.ecg_buffer)] = val
                    state.ecg_cursor += 1
                    if state.ecg_cursor % 4 == 0:
                        emit_sio('ecg_point', {'val': val})
                    
                    if state.ecg_detector.detect(state.ecg_buffer, state.ecg_cursor - 1):
                        process_ecg_peak(state.ecg_cursor - 1)
                        
                elif prefix == 'P':
                    state.ppg_buffer[state.ppg_cursor % len(state.ppg_buffer)] = val
                    state.ppg_cursor += 1
                    if state.ppg_cursor % 2 == 0:
                        emit_sio('ppg_point', {'val': val})
                        
                    if state.ppg_detector.detect(val, state.ppg_cursor - 1):
                        process_ppg_peak()
                        
                elif prefix == 'T':
                    state.current_temp = val
                    emit_sio('temp_update', {'val': val})
                    if state.is_screening:
                        state.screening_results['max_temp'] = max(state.screening_results['max_temp'], val)
            except: continue
        
        if state.is_screening:
            elapsed = time.time() - state.screening_start_time
            if elapsed >= 60:
                finish_screening()
        
        time.sleep(0.001)

def process_ecg_peak(cursor):
    if state.is_screening: state.screening_results['total_beats'] += 1
    
    curr_p = cursor % len(state.ecg_buffer)
    start, end = (curr_p - BEFORE), (curr_p + AFTER)
    if start < 0 or end >= len(state.ecg_buffer):
        indices = np.arange(start, end) % len(state.ecg_buffer)
        seg = state.ecg_buffer[indices].copy()
    else:
        seg = state.ecg_buffer[start:end].copy()
    
    seg = (seg - np.mean(seg)) / (np.std(seg) + 1e-8)
    inp = seg.reshape(1, WIN_ARR, 1).astype(np.float32)
    probs = arr_model.predict(inp, verbose=0)[0]
    pred = int(np.argmax(probs))
    
    if state.is_screening: state.screening_results['arr_counts'][pred] += 1
    
    emit_sio('beat_detected', {
        'class': CLASS_NAMES_ARR[pred],
        'color': CLASS_COLORS_ARR[pred],
        'probs': probs.tolist()
    })

def process_ppg_peak():
    now = time.time()
    rr_val = (now - state.last_ppg_peak_time) * 1000.0
    state.last_ppg_peak_time = now
    
    if 400 < rr_val < 1500:
        state.rr_buffer.append(rr_val)
        state.beat_count_ppg += 1
        
        emit_sio('rr_point', {'val': rr_val})
        
        if len(state.rr_buffer) == AFIB_WIN_SIZE and state.beat_count_ppg % 5 == 0:
            window_rr = np.array(state.rr_buffer)
            feat = extract_hrv_features(window_rr)
            feat_sc = afib_hsc.transform(feat.reshape(1, -1))
            seq = (window_rr - h_rr_stats[0]) / h_rr_stats[1]
            seq = seq.reshape(1, 30, 1).astype(np.float32)
            prob = float(afib_hybrid.predict([seq, feat_sc], verbose=0)[0][1])
            is_afib = prob >= AFIB_THRESH
            
            emit_sio('afib_update', {'prob': prob, 'is_afib': is_afib})
            
            if state.is_screening:
                state.screening_results['afib_windows'] += 1
                if is_afib: state.screening_results['afib_detections'] += 1

def finish_screening():
    state.is_screening = False
    res = state.screening_results
    
    arr_counts = res['arr_counts']
    # Exclude index 4 (Unknown/Q) from abnormal count to reduce noise-induced false positives
    abnormal_arr = sum(arr_counts[1:4]) 
    pvc_count = arr_counts[2]
    unknown_count = arr_counts[4]
    
    arr_risk = "LOW"
    if pvc_count > 5 or abnormal_arr > 15: arr_risk = "HIGH"
    elif abnormal_arr > 5: arr_risk = "MODERATE"
    
    afib_ratio = (res['afib_detections'] / max(1, res['afib_windows']))
    afib_risk = "LOW"
    if afib_ratio > 0.15: afib_risk = "HIGH"
    elif afib_ratio > 0.05: afib_risk = "MODERATE"
    
    max_t = res['max_temp']
    temp_status = "NORMAL"
    if max_t > 37.8: temp_status = "FEVER"
    elif max_t < 35.5 and max_t > 0: temp_status = "LOW"

    report = {
        'arr_risk': arr_risk,
        'abnormal_beats': abnormal_arr,
        'pvc_count': pvc_count,
        'unknown_beats': unknown_count,
        'afib_risk': afib_risk,
        'afib_burden': afib_ratio * 100,
        'max_temp': max_t,
        'temp_status': temp_status,
        'total_beats': res['total_beats']
    }
    
    emit_sio('screening_complete', report)

# Mount static files
app.mount("/", StaticFiles(directory=str(ROOT / "src" / "web_monitor" / "static"), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(socket_app, host="0.0.0.0", port=8000)
