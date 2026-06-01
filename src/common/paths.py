from pathlib import Path

ROOT        = Path(__file__).resolve().parents[2]
DATA_DIR    = ROOT / 'data'
MODELS_DIR  = ROOT / 'models'
OUTPUTS_DIR = ROOT / 'outputs'

# AFib specific paths
AFIB_DATA_DIR    = DATA_DIR / 'afib'
AFIB_MODELS_DIR  = MODELS_DIR / 'afib'
AFIB_OUTPUTS_DIR = OUTPUTS_DIR / 'afib'
AFIB_DATASET     = AFIB_DATA_DIR / 'afib_dataset.npz'

# Arrhythmia specific paths
ARR_DATA_DIR    = DATA_DIR / 'arrhythmia'
ARR_MODELS_DIR  = MODELS_DIR / 'arrhythmia'
ARR_OUTPUTS_DIR = OUTPUTS_DIR / 'arrhythmia'
ARR_DATASET     = ARR_DATA_DIR / 'mitbih_processed.npz'
ARR_RECORDS_DIR = ARR_DATA_DIR / 'records'

def ensure_dirs():
    """Ensure all required directories exist."""
    for d in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR,
              AFIB_DATA_DIR, AFIB_MODELS_DIR, AFIB_OUTPUTS_DIR,
              ARR_DATA_DIR, ARR_MODELS_DIR, ARR_OUTPUTS_DIR, ARR_RECORDS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
