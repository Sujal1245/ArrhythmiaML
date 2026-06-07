// src/web_monitor/static/app.js

const socket = io();

const CLASS_NAMES = ['Normal (N)', 'Supraventricular (S)', 'Ventricular (V)', 'Fusion (F)', 'Unknown (Q)'];
const CLASS_COLORS = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6'];

// ── Charts Initialization ────────────────────────────────────────────────
const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    scales: {
        x: { display: false },
        y: { 
            grid: { color: '#222' },
            ticks: { color: '#666', font: { size: 9 } }
        }
    },
    elements: {
        point: { radius: 0 },
        line: { tension: 0.1, borderWidth: 2 }
    },
    plugins: { legend: { display: false } }
};

const ecgCtx = document.getElementById('ecgChart').getContext('2d');
const ecgChart = new Chart(ecgCtx, {
    type: 'line',
    data: {
        labels: Array(400).fill(''),
        datasets: [{
            data: Array(400).fill(0),
            borderColor: '#00ff88',
        }]
    },
    options: chartOptions
});

const ppgCtx = document.getElementById('ppgChart').getContext('2d');
const ppgChart = new Chart(ppgCtx, {
    type: 'line',
    data: {
        labels: Array(400).fill(''),
        datasets: [{
            data: Array(400).fill(0),
            borderColor: '#ff0066',
        }]
    },
    options: chartOptions
});

const rrCtx = document.getElementById('rrChart').getContext('2d');
const rrChart = new Chart(rrCtx, {
    type: 'line',
    data: {
        labels: Array(100).fill(''),
        datasets: [{
            data: Array(100).fill(800),
            borderColor: '#00ff88',
            showLine: true,
            pointRadius: 2,
            pointBackgroundColor: '#00ff88'
        }]
    },
    options: {
        ...chartOptions,
        scales: {
            x: { display: false },
            y: { 
                min: 400, 
                max: 1500, 
                grid: { color: '#222' },
                ticks: { color: '#666', font: { size: 9 } }
            }
        }
    }
});

const tempCtx = document.getElementById('tempChart').getContext('2d');
const tempChart = new Chart(tempCtx, {
    type: 'line',
    data: {
        labels: Array(100).fill(''),
        datasets: [{
            data: Array(100).fill(36.5),
            borderColor: '#ff9f43',
        }]
    },
    options: {
        ...chartOptions,
        scales: {
            x: { display: false },
            y: { min: 30, max: 42, display: false }
        }
    }
});

// ── UI Components ────────────────────────────────────────────────────────
const confidenceBars = document.getElementById('confidence-bars');
CLASS_NAMES.forEach((name, i) => {
    const row = document.createElement('div');
    row.className = 'bar-row';
    row.innerHTML = `
        <div class="bar-label">${name}</div>
        <div class="bar-outer"><div class="bar-inner" id="bar-${i}" style="background-color: ${CLASS_COLORS[i]}"></div></div>
        <div class="bar-value" id="val-${i}">0%</div>
    `;
    confidenceBars.appendChild(row);
});

const beatLabel = document.getElementById('beat-label');
const statusText = document.getElementById('status-text');
const tempVal = document.getElementById('temp-val');
const screenBtn = document.getElementById('screen-btn');
const screeningTimer = document.getElementById('screening-timer');
const reportModal = document.getElementById('report-modal');
const reportBody = document.getElementById('report-body');

// ── Socket Events ────────────────────────────────────────────────────────
socket.on('connect', () => {
    document.getElementById('serial-status').innerText = 'Server: Connected';
    statusText.innerText = 'READY';
});

socket.on('ecg_point', (data) => {
    ecgChart.data.datasets[0].data.shift();
    ecgChart.data.datasets[0].data.push(data.val);
    ecgChart.update('none');
});

socket.on('ppg_point', (data) => {
    ppgChart.data.datasets[0].data.shift();
    ppgChart.data.datasets[0].data.push(data.val);
    ppgChart.update('none');
});

socket.on('rr_point', (data) => {
    rrChart.data.datasets[0].data.shift();
    rrChart.data.datasets[0].data.push(data.val);
    rrChart.update();
});

socket.on('temp_update', (data) => {
    tempVal.innerText = data.val.toFixed(1);
    tempChart.data.datasets[0].data.shift();
    tempChart.data.datasets[0].data.push(data.val);
    tempChart.update('none');
});

socket.on('beat_detected', (data) => {
    beatLabel.innerText = data.class;
    beatLabel.style.backgroundColor = data.color;
    
    data.probs.forEach((p, i) => {
        const bar = document.getElementById(`bar-${i}`);
        const val = document.getElementById(`val-${i}`);
        bar.style.width = `${p * 100}%`;
        val.innerText = `${(p * 100).toFixed(1)}%`;
        bar.style.opacity = (i === data.probs.indexOf(Math.max(...data.probs))) ? 1 : 0.3;
    });
});

socket.on('afib_update', (data) => {
    // We could show AFib probability somewhere
    if (data.is_afib) {
        statusText.innerText = 'AFIB DETECTED!';
        statusText.style.color = '#e74c3c';
    } else {
        statusText.innerText = 'STABLE';
        statusText.style.color = '#2ecc71';
    }
});

let screeningActive = false;
let screenStartTime = 0;

socket.on('screening_status', (data) => {
    screeningActive = data.active;
    if (screeningActive) {
        screenBtn.innerText = 'CANCEL';
        screenBtn.className = 'btn btn-red';
        screeningTimer.classList.remove('hidden');
        screenStartTime = Date.now();
        updateTimer();
    } else {
        screenBtn.innerText = 'START SCREENING';
        screenBtn.className = 'btn btn-green';
        screeningTimer.classList.add('hidden');
    }
});

function updateTimer() {
    if (!screeningActive) return;
    const elapsed = (Date.now() - screenStartTime) / 1000;
    const remaining = Math.max(0, 60 - elapsed);
    screeningTimer.innerText = `Screening: ${Math.floor(remaining)}s`;
    if (remaining > 0) setTimeout(updateTimer, 500);
}

socket.on('screening_complete', (report) => {
    screeningActive = false;
    screenBtn.innerText = 'START SCREENING';
    screenBtn.className = 'btn btn-green';
    screeningTimer.classList.add('hidden');

    const text = `
1. ARRHYTHMIA RISK : ${report.arr_risk}
   - Abnormal Beats: ${report.abnormal_beats}
   - PVC Count: ${report.pvc_count}
   - Unknown/Noise : ${report.unknown_beats}

2. AFIB RISK       : ${report.afib_risk}
   - AFib Burden: ${report.afib_burden.toFixed(1)}%

3. TEMPERATURE     : ${report.max_temp.toFixed(1)}°C
   - Status: ${report.temp_status}

Total ECG Beats Processed: ${report.total_beats}
    `;
    reportBody.innerText = text;
    reportModal.classList.remove('hidden');
    
    const finalColor = (report.arr_risk === 'HIGH' || report.afib_risk === 'HIGH' || report.temp_status === 'FEVER') ? '#e74c3c' : 
                       (report.arr_risk === 'MODERATE' || report.afib_risk === 'MODERATE') ? '#f39c12' : '#2ecc71';
    
    document.querySelector('.modal-content').style.borderColor = finalColor;
});

// ── Controls ─────────────────────────────────────────────────────────────
screenBtn.onclick = () => {
    if (screeningActive) {
        socket.emit('cancel_screening');
    } else {
        socket.emit('start_screening');
    }
};

document.getElementById('close-report').onclick = () => {
    reportModal.classList.add('hidden');
};
