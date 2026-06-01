# ArrhythmiaML

A machine learning project for ECG analysis, focusing on detecting general arrhythmias and Atrial Fibrillation (AFib).

## Project Structure

```
ArrhythmiaML/
├── data/               # Datasets (downloaded and processed)
│   ├── afib/           # MIT-BIH Atrial Fibrillation Database
│   └── arrhythmia/     # MIT-BIH Arrhythmia Database
├── models/             # Trained model files (.keras, .pkl)
├── outputs/            # Training curves and evaluation plots
└── src/
    ├── afib/           # AFib detection pipeline
    ├── arrhythmia/     # General arrhythmia classification pipeline
    └── dashboard/      # Unified simulation dashboard
```

## Features

- **Arrhythmia Classification:** 5-class classification using a 1D-CNN on the MIT-BIH Arrhythmia Database.
- **AFib Detection:** Ensemble approach using a 1D-CNN on RR intervals and a Random Forest on handcrafted HRV features from the MIT-BIH AF Database.
- **Unified Dashboard:** Real-time simulation dashboard visualizing ECG signals, beat classifications, and AFib detection.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd ArrhythmiaML
    ```

2.  **Create a virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    *(Currently no requirements.txt, but needs wfdb, numpy, scipy, antropy, tensorflow, scikit-learn, joblib, matplotlib)*
    ```bash
    pip install wfdb numpy scipy antropy tensorflow scikit-learn joblib matplotlib
    ```

## Usage

### 1. Build Datasets
Download and process the data from PhysioNet.
```bash
python src/afib/build_dataset.py
python src/arrhythmia/build_dataset.py
```

### 2. Train Models
Train the CNN and Random Forest models.
```bash
python src/afib/train.py
python src/arrhythmia/train.py
```

### 3. Run Dashboard
Launch the unified simulation dashboard.
```bash
python src/dashboard/unified_dashboard.py
```

## License
MIT
