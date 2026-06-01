# Shared configuration and constants

BEAT_MAP = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,
    'A': 1, 'a': 1, 'J': 1, 'S': 1,
    'V': 2, 'E': 2,
    'F': 3,
    '/': 4, 'f': 4, 'Q': 4
}

ARR_CLASS_NAMES = [
    'Normal (N)', 'Supraventricular (S)',
    'Ventricular (V)', 'Fusion (F)', 'Unknown (Q)'
]

ARR_CLASS_COLORS = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']

AFIB_CLASS_NAMES = ['Normal', 'AFib']
AFIB_CLASS_COLORS = ['#2ecc71', '#e74c3c']

# Signal parameters
ARR_FS = 360
ARR_WINDOW = 280
ARR_BEFORE = 100
ARR_AFTER = ARR_WINDOW - ARR_BEFORE

AFIB_WINDOW_SIZE = 30
AFIB_STEP_SIZE = 10
