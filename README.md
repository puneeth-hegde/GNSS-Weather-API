# GNSS Weather Prediction

## What This Project Is

This repository is your end-to-end weather intelligence pipeline for forecasting rainfall at IISC Bangalore using GNSS-derived atmospheric signals and standard meteorological variables.

You built the project in five connected layers:

1. Multi-source data ingestion and cleaning (2010-2025)
2. Feature engineering and sequence creation for time-series learning
3. Two-stage deep learning training (rain occurrence + rain amount)
4. Model compatibility repair/export for reliable deployment
5. Production-style inference API and deployable artifacts

In short, this is not only EDA. It is a complete data-to-serving ML workflow.

## Problem Statement

Given the most recent 48 hours of atmospheric context, predict precipitation 6 hours ahead.

You split this into two tasks:

1. Stage 1 classification: Will it rain or not?
2. Stage 2 regression: If rain is expected, how much rain (mm/hr)?

This two-stage setup handles rainfall sparsity better than a single direct regressor.

## Data Sources and Coverage

### 1. NASA POWER hourly data

- Temperature at 2 m
- Relative humidity at 2 m
- Surface pressure
- Precipitation

### 2. NGL GNSS troposphere data (IISC station)

- Zenith Total Delay (ZTD)
- Precipitable Water Vapor (PWV)
- Raw format: yearly `.zip` files containing daily `.trop.gz` files (5-minute records)

### Final integrated dataset characteristics

- Coverage window: 2010 to 2025
- Master hourly dataset: 134,715 rows (from notebook output)
- Final model feature count: 13
- Target: hourly precipitation (mm/hr)

## Features Used By Models

### Physical features

- `Temperature_C`
- `Relative_Humidity`
- `Surface_Pressure_kPa`
- `ZTD`
- `PWV`

### Time encoding features

- `hour_sin`, `hour_cos`
- `month_sin`, `month_cos`

### Rain persistence (lag) features

- `Precipitation_Lag1`
- `Precipitation_Lag3`
- `Precipitation_Lag6`
- `Precipitation_Lag24`

### Target

- `Precipitation_mm`

## End-to-End Pipeline You Implemented

### Step 1: Build master dataset

Implemented in notebook workflow under `training_scripts_and notebooks/creating_data.ipynb`:

- Read yearly NGL zip bundles and decompress nested `.trop.gz` files
- Parse `+TROP/SOLUTION` blocks for IISC records
- Convert epoch format (`YY:DOY:SSSSS`) to timestamps
- Apply physical sanity bounds for ZTD/PWV
- Resample GNSS 5-minute measurements to hourly
- Load NASA POWER hourly CSV and align units
- Merge GNSS + NASA streams on datetime
- Fill short/medium gaps and finalize a continuous hourly dataset
- Generate cyclic and lag features

### Step 2: Prepare supervised sequences

Implemented in `training_scripts_and notebooks/3dnumpy_array_data_prep.ipynb`:

- Scale inputs and target separately (`feature_scaler.gz`, `target_scaler.gz`)
- Build sliding windows with:
  - Lookback = 48 hours
  - Lead time = 6 hours ahead
- Export arrays for training:
  - `X_sequences.npy`
  - `y_targets.npy`

Notebook logs indicate sequence tensor shape around `(134662, 48, 13)`.

### Step 3: Train two-stage deep models

Primary training notebook: `training_scripts_and notebooks/two_staged_cnn_lstm.ipynb`

- Stage 1 (classifier): CNN + LSTM + attention -> sigmoid output
- Stage 2 (regressor): CNN + LSTM + attention -> linear output
- Chronological splits are used (time-aware training/validation/test)
- Class weighting is applied in Stage 1 to address imbalance
- Saved checkpoints:
  - `stage1_classifier.keras`
  - `stage2_regressor.keras`

Alternative architecture notebook: `training_scripts_and notebooks/Temporial_fusion_transfer.ipynb`

- Two-stage Temporal Fusion Transformer (TFT-like) implementation
- Keras-layer-safe custom blocks (GRN, VSN, transformer blocks)
- Saved classifier artifact:
  - `tft_stage1_classifier.keras`

### Step 4: Repair/export models for serving

You handled real-world serialization compatibility issues and deployment conversion:

- `cleaner.py` removes problematic `quantization_config` entries from `.keras` archives recursively
- `training_scripts_and notebooks/repairing_layers.ipynb` reloads custom-layer models and exports TensorFlow SavedModel folders:
  - `cnn_s1_saved/`
  - `tft_s1_saved/`
  - `cnn_s2_saved/`

This is an important production step: it decouples serving from fragile notebook-only custom-object loading behavior.

### Step 5: Serve predictions through API

Core serving logic in `live_inference.py` and API wrapper in `app.py`.

Inference flow:

1. Load `target_scaler.gz`
2. Load three SavedModel endpoints
3. Get Stage 1 probabilities from CNN and TFT classifiers
4. Ensemble with fixed weights:
   - `0.10 * CNN + 0.90 * TFT`
5. If ensemble probability > 0.5:
   - Run Stage 2 regressor
   - Clip scaled output to `[0, 1]`
   - Inverse-scale to mm/hr
6. Return status, probability, and predicted amount

## Current API Behavior

`app.py` exposes:

- `GET /` -> health/info message
- `GET /predict/demo-storm` -> runs `predict_rain(...)` on preloaded `demo_storm_data.npy`

Important details:

- CORS is open (`allow_origins=['*']`) for frontend integration
- API response returns a simple JSON payload with forecast status and amount
- This repository currently demonstrates inference with fixed demo input; user-upload/live stream endpoint is not yet implemented

## Repository File Guide

- `app.py`: FastAPI application and endpoints
- `live_inference.py`: model loading, ensembling, and two-stage prediction logic
- `cleaner.py`: `.keras` sanitizer for problematic config keys
- `upload.py`: Hugging Face Space upload helper for SavedModel folders
- `Dockerfile`: containerized API runtime setup
- `requirements.txt`: Python dependencies for serving
- `demo_storm_data.npy`: sample 48-hour feature tensor used by demo endpoint
- `feature_scaler.gz`: saved feature scaler from sequence-prep workflow
- `target_scaler.gz`: target scaler used for inverse transform in inference
- `stage1_classifier.keras`: Stage 1 CNN-LSTM classifier checkpoint
- `tft_stage1_classifier.keras`: Stage 1 TFT classifier checkpoint
- `stage2_regressor.keras`: Stage 2 regressor checkpoint
- `cnn_s1_saved/`, `tft_s1_saved/`, `cnn_s2_saved/`: serving-ready SavedModel exports
- `Data_preprocessing_eda/`: raw/processed source CSV assets for preprocessing and analysis
- `training_scripts_and notebooks/`: end-to-end notebook development history (EDA, prep, training, repair)

## Notebook Roles (What Each One Was Used For)

- `creating_data.ipynb`: parse and merge NASA + NGL data, build cleaned master dataset
- `EDA_Notebook.ipynb`: exploratory analysis, climatology, rain imbalance and predictor behavior checks
- `3dnumpy_array_data_prep.ipynb`: scale, sequence, and export model-ready tensors
- `two_staged_cnn_lstm.ipynb`: train/evaluate two-stage CNN-LSTM pipeline
- `Temporial_fusion_transfer.ipynb`: train/evaluate two-stage TFT-based alternative
- `repairing_layers.ipynb`: custom-layer reload and SavedModel export for deployment

## How To Run The API Locally

### Option 1: direct Python

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

### Option 2: Docker

```bash
docker build -t gnss-weather-api .
docker run -p 8000:8000 gnss-weather-api
```

Then call:

- `http://localhost:8000/`
- `http://localhost:8000/predict/demo-storm`

## Technical Decisions That Stand Out In This Project

1. You used GNSS moisture proxies (ZTD, PWV) as first-class predictors, not just standard weather variables.
2. You engineered a two-stage architecture that separates detection from intensity estimation.
3. You enforced time-aware splits and sequence-based supervision for realistic forecasting.
4. You solved Keras/TensorFlow portability issues through sanitization and SavedModel exports.
5. You wrapped the trained stack into an API-ready deployment structure.

## Known Practical Constraints

1. Current API endpoint uses a fixed demo tensor, so real-time ingestion is not yet part of this codebase.
2. `feature_scaler.gz` exists but is not used inside `live_inference.py`, which implies inference input is expected to be pre-scaled upstream.
3. TFT Stage 2 regressor artifact is not included in root serving flow; current deployed stack uses CNN Stage 2.

## Summary

This project demonstrates a complete applied ML lifecycle for GNSS-informed rainfall forecasting:

- raw geophysical data engineering,
- long-horizon temporal feature construction,
- two-stage deep learning design,
- serialization/compatibility hardening,
- and API-level deployment.

It is a strong implementation of moving from research notebooks to a serving-capable weather prediction system.
