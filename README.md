# GNSS Weather Prediction API

## Deployed on HuggingFace Spaces

This project is live and publicly accessible on **HuggingFace Spaces**, containerised with Docker and running on a CPU-only instance.

**Space URL:** https://huggingface.co/spaces/puneeth2025/gnss-weather-api

The full stack — FastAPI application, all three SavedModel folders, and the scalers — is packaged into a Docker image and deployed there. HuggingFace Spaces builds the image from the `Dockerfile` in this repository and serves the API on port 7860. No GPU is used; the models were explicitly exported as CPU-safe SavedModel format to match this environment.

---

## Live Demo

Test the live API endpoint directly:

`GET https://puneeth2025-gnss-weather-api.hf.space/predict/demo-storm`

Example response:
```json
{"status": "success", "data": {"status": "CLEAR", "probability": "46.0%", "predicted_amount": "0.0 mm/hr"}}
```

The demo endpoint uses a pre-scaled dry-period tensor. The CLEAR response with a non-zero probability is correct behavior. The model detected a 46% chance of rain, ran the regressor, but the regressor predicted less than 1.0 mm/hr which is a trace amount and not meteorologically meaningful, so the final status is CLEAR.

---

## What This Project Is

This repository is an end-to-end rainfall forecasting pipeline. It uses GNSS-derived atmospheric signals (Zenith Total Delay and Precipitable Water Vapor from the IISC GPS station) combined with standard meteorological variables from NASA to predict whether it will rain and how much, 6 hours in advance.

The project covers the entire lifecycle from raw data ingestion to a live REST API:

1. Multi-source data ingestion and cleaning (2010-2025)
2. Feature engineering and sequence creation for time-series learning
3. Two-stage deep learning training (rain occurrence + rain amount)
4. Model compatibility repair and export for reliable deployment
5. Production-style inference API running on HuggingFace Spaces

---

## Problem Statement

Given the most recent 48 hours of atmospheric measurements, predict precipitation 6 hours ahead.

This is split into two tasks:

1. Stage 1 classification: Will it rain or not?
2. Stage 2 regression: If rain is expected, how much rain in mm/hr?

A two-stage design is used instead of a single regressor because rainfall is heavily imbalanced — most hours are dry. A classifier that first decides whether rain will happen, followed by a regressor that only runs when rain is likely, handles this imbalance much better.

---

## Data Sources and Coverage

### 1. NASA POWER hourly data

Provides standard meteorological variables downloaded as hourly CSV:

- Temperature at 2 m
- Relative humidity at 2 m
- Surface pressure
- Precipitation (used as target)

### 2. NGL GNSS troposphere data (IISC station)

Provides GPS-derived atmospheric delay measurements which are strong proxies for moisture in the atmosphere:

- Zenith Total Delay (ZTD): total signal delay caused by the atmosphere
- Precipitable Water Vapor (PWV): estimated column water vapor derived from ZTD

Raw format: yearly `.zip` files, each containing daily `.trop.gz` files with 5-minute resolution records.

### Final integrated dataset

- Coverage: 2010 to 2025
- Total rows after merging and resampling to hourly: 134,715
- Features fed to the model: 13
- Target variable: hourly precipitation in mm/hr

---

## Features Used By Models

### Physical features

- `Temperature_C` — air temperature at 2 m
- `Relative_Humidity` — humidity at 2 m
- `Surface_Pressure_kPa` — atmospheric pressure at surface
- `ZTD` — GNSS zenith total delay, captures total atmospheric moisture column
- `PWV` — precipitable water vapor derived from ZTD

### Time encoding features

Sine and cosine encodings are used instead of raw integers so the model understands the cyclic nature of time (e.g. hour 23 is close to hour 0, December is close to January):

- `hour_sin`, `hour_cos`
- `month_sin`, `month_cos`

### Rain persistence (lag) features

Past rainfall values are included as features because recent rain is one of the strongest predictors of near-future rain:

- `Precipitation_Lag1` — rain 1 hour ago
- `Precipitation_Lag3` — rain 3 hours ago
- `Precipitation_Lag6` — rain 6 hours ago
- `Precipitation_Lag24` — rain 24 hours ago (same time yesterday)

### Target

- `Precipitation_mm` — hourly rainfall in mm/hr, 6 hours ahead

---

## End-to-End Pipeline

### Step 1: Build master dataset

Implemented in `training_scripts_and_notebooks/creating_data.ipynb`.

The GNSS raw data arrives as nested compressed archives. Each yearly `.zip` contains hundreds of daily `.trop.gz` files. Each file contains a `+TROP/SOLUTION` block with 5-minute resolution records for multiple stations. Only the IISC station records are extracted.

The epoch timestamps in GNSS files use the format `YY:DOY:SSSSS` (year, day-of-year, seconds-of-day) which are converted to standard datetime. Physical sanity bounds are applied to ZTD and PWV to remove sensor anomalies. The 5-minute GNSS records are then resampled to hourly by averaging.

The NASA POWER data arrives as a clean hourly CSV. After unit alignment it is merged with the GNSS stream on datetime. Short gaps are forward-filled, cyclic time features and lag features are generated, and the final dataset is exported as a single continuous hourly CSV.

### Step 2: Prepare supervised sequences

Implemented in `training_scripts_and_notebooks/3dnumpy_array_data_prep.ipynb`.

The flat hourly dataset is converted into 3D tensors for sequence modeling. A sliding window of 48 hours is used as the lookback, and the target is the precipitation value 6 hours after the end of each window. This produces a tensor of shape `(134662, 48, 13)` — 134,662 samples, each being 48 timesteps of 13 features.

Features and target are scaled separately. The feature scaler is saved as `feature_scaler.gz` and the target scaler as `target_scaler.gz`. Separating them is important because the target needs to be inverse-transformed independently at inference time to recover the actual mm/hr value.

### Step 3: Train two-stage deep models

**Primary model** (`training_scripts_and_notebooks/two_staged_cnn_lstm.ipynb`):

A CNN-LSTM architecture with attention is trained for both stages. Convolutional layers extract local temporal patterns, the LSTM captures long-range dependencies across the 48-hour window, and attention weighs the most relevant timesteps. Stage 1 uses a sigmoid output for binary rain/no-rain probability. Stage 2 uses a linear output for mm/hr regression. Class weighting is applied in Stage 1 to compensate for the heavy imbalance between rainy and dry hours. Training uses chronological splits so no future data leaks into validation or test.

Saved as `stage1_classifier.keras` and `stage2_regressor.keras`.

**Alternative model** (`training_scripts_and_notebooks/Temporal_fusion_transformer.ipynb`):

A Temporal Fusion Transformer (TFT) style architecture is implemented from scratch using custom Keras layers. The key components are:

- `GatedResidualNetwork` (GRN): applies gated non-linear transformations with residual connections and layer normalization
- `VariableSelectionNetwork` (VSN): learns which of the 13 input features are most relevant at each timestep using soft attention weights
- `TransformerBlock`: multi-head self-attention with feed-forward layers and layer normalization
- `PositionalEncoding`: adds positional information to the sequence before the transformer

Only the Stage 1 classifier was trained with the TFT architecture. Saved as `tft_stage1_classifier.keras`.

### Step 4: Repair and export models for serving

This step involved solving a chain of real-world serialization and deployment compatibility problems. See the Deployment Problems and Solutions section for the full breakdown.

The final output is three CPU-safe TensorFlow SavedModel folders that can be loaded with `tf.saved_model.load()` and called with `.serve()` without needing any custom class definitions:

- `cnn_s1_saved/` — CNN Stage 1 classifier
- `tft_s1_saved/` — TFT Stage 1 classifier
- `cnn_s2_saved/` — CNN Stage 2 regressor

### Step 5: Serve predictions through API

`live_inference.py` loads the three SavedModel folders and the target scaler at startup. When a prediction is requested it runs the following sequence:

1. Feed the 48-hour input tensor to `cnn_s1` and `tft_s1` separately to get two rain probability values
2. Combine them as `0.10 * CNN_probability + 0.90 * TFT_probability` — TFT gets 90% weight because it performed better
3. If the ensemble probability exceeds 0.35, run `cnn_s2` to predict the rainfall amount
4. Clip the scaled regressor output to `[0, 1]`, inverse-transform using `target_scaler` to recover mm/hr
5. Apply meteorological thresholds to assign a human-readable status

The 0.35 threshold is intentionally lower than 0.5 to make the system sensitive — it is better to run the regressor unnecessarily than to miss a rain event.

Rainfall status thresholds:

| Predicted amount | Status |
|---|---|
| > 15.0 mm/hr | SEVERE STORM EXPECTED |
| > 5.0 mm/hr | HEAVY RAIN EXPECTED |
| > 1.0 mm/hr | LIGHT RAIN EXPECTED |
| < 1.0 mm/hr | CLEAR (trace amount, not meaningful) |

`app.py` wraps this logic in a FastAPI application and exposes two endpoints.

---

## Deployment Problems and Solutions

---

### Problem 1: quantization_config crash on load

**Error:**
```
ValueError: Unrecognized keyword arguments passed to Dense: {'quantization_config': None}
```

**Cause:** Keras 3.13.2 (used in Colab to save the models) added a new `quantization_config` field to every Dense layer config. The deployment container was running an older Keras version that had never seen this field and crashed trying to deserialize it.

**What was tried first:** A `SafeDense` subclass was written to pop the unknown key before passing the config to the parent class. This was the right idea but insufficient on its own because the underlying issue was a Python version incompatibility that prevented the correct Keras version from installing at all.

**What actually fixed it:** Upgrading the Docker base image from Python 3.10 to Python 3.11, which allowed Keras 3.13.2 to install correctly and recognize `quantization_config` natively.

---

### Problem 2: Python version mismatch

**Error:**
```
ERROR: Could not find a version that satisfies the requirement keras==3.13.2
Keras 3.13.x Requires-Python >=3.11
```

**Cause:** Keras 3.x dropped support for Python 3.10. The original Dockerfile used `python:3.10-slim`.

**Fix:**
```dockerfile
FROM python:3.11-slim
```

With `requirements.txt`:
```
tensorflow==2.19.0
keras==3.13.2
numpy==1.26.4
```

---

### Problem 3: TransformerBlock weights not loading

**Error:**
```
ValueError: A total of 24 objects could not be loaded.
Layer 'key' expected 2 variables, but received 0 variables during loading.
```

**Cause:** Google silently updated the internal build of Keras 3.13.2 between the model save date (March 30, 2026) and deployment date (April 25, 2026) without changing the version number. The internal naming of `MultiHeadAttention` sublayers changed — `key_dense` became `key`, `query_dense` became `query` and so on. The weights were confirmed to exist inside the `.keras` archive by inspecting `model.weights.h5` directly, but Keras could not map them to the right layers because the path names no longer matched.

**Fix:** Patched Keras at runtime in Colab to suppress the crash and force-load the models despite the name mismatch, then immediately re-exported them as SavedModel format which stores weights by position rather than by name and is immune to this kind of internal change:

```python
import keras.src.saving.saving_lib as saving_lib
saving_lib._raise_loading_failure = lambda error_msgs: None

cnn_s1 = tf.keras.models.load_model("stage1_classifier.keras", custom_objects=CUSTOM_OBJECTS, compile=False)
tft_s1 = tf.keras.models.load_model("tft_stage1_classifier.keras", custom_objects=CUSTOM_OBJECTS, compile=False)
cnn_s2 = tf.keras.models.load_model("stage2_regressor.keras", custom_objects=CUSTOM_OBJECTS, compile=False)

cnn_s1.export("cnn_s1_saved")
tft_s1.export("tft_s1_saved")
cnn_s2.export("cnn_s2_saved")
```

---

### Problem 4: CudnnRNNV3 GPU-only op crash on CPU deployment

**Error:**
```
No OpKernel was registered to support Op 'CudnnRNNV3'
```

**Cause:** The first SavedModel export was run in Colab while the GPU was active. TensorFlow detected the GPU and permanently compiled the LSTM layers using `CudnnRNNV3`, a CUDA-only kernel. When these files were loaded on HuggingFace which is CPU-only, TensorFlow had no implementation for that op and crashed.

**What does not work:** Setting `os.environ["CUDA_VISIBLE_DEVICES"] = "-1"` inside `live_inference.py` on the HuggingFace server has no effect. The GPU kernel is already baked into the saved files. The server environment is irrelevant at that point.

**Fix:** Re-ran the export in Colab with the GPU hidden before TensorFlow was imported. This forced TensorFlow to compile the LSTM layers using the standard CPU kernel instead:

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Must come before any TensorFlow import

import tensorflow as tf
# load models and export using model.export()
```

The resulting SavedModel folders contain only CPU-compatible ops and run correctly on any machine.

---

### Final working stack

| Component | Value |
|---|---|
| Python | 3.11 |
| TensorFlow | 2.19.0 |
| Keras | 3.13.2 |
| Model format | TensorFlow SavedModel |
| Load method | `tf.saved_model.load()` + `.serve()` |
| Deployment | HuggingFace Spaces (Docker, CPU-only) |

---

## File Guide

### `app.py`

The FastAPI application. Loads `demo_scenarios.npy` once at startup into memory as `STORM_DATA`. Exposes two endpoints:

- `GET /` — returns a simple health check message confirming the API is running
- `GET /predict/demo-storm` — passes `STORM_DATA` to `predict_rain()` from `live_inference.py` and returns the forecast as JSON

CORS is fully open (`allow_origins=['*']`) so any frontend can call this API. The server runs on port 7860 which is required by HuggingFace Spaces.

### `live_inference.py`

The core inference engine. Runs at import time to load the target scaler and all three SavedModel folders into memory. Exposes a single function `predict_rain(live_48hr_data)` which accepts a pre-scaled numpy array of shape `(1, 48, 13)` and returns a dictionary with `status`, `probability`, and `predicted_amount`.

### `Dockerfile`

Defines the container environment. Uses `python:3.11-slim` as the base image, installs all dependencies from `requirements.txt`, copies all project files into `/app`, exposes port 7860, and starts the server with uvicorn.

### `requirements.txt`

Pins all Python dependencies for the serving environment. The critical pins are `tensorflow==2.19.0`, `keras==3.13.2`, and `numpy==1.26.4`. These must match the versions used in Colab when the models were saved.

### `demo_scenarios.npy`

A single pre-scaled input tensor of shape `(1, 48, 13)` representing 48 hours of atmospheric measurements. This is the fixed input used by the demo endpoint. All 13 feature values are already scaled to `[0, 1]` as expected by the model.

### `target_scaler.gz`

A gzip-compressed joblib file containing the fitted scaler for the precipitation target. Used in `live_inference.py` to convert the model's scaled output back to real mm/hr values via inverse transform.

### `feature_scaler.gz`

A gzip-compressed joblib file containing the fitted scaler for the 13 input features. Not used inside `live_inference.py` because the demo input is already pre-scaled. Required upstream when preparing real-time input from raw sensor data.

### `stage1_classifier.keras`, `tft_stage1_classifier.keras`, `stage2_regressor.keras`

The original model checkpoint files saved from Colab training. These are kept in the repository for reference and reproducibility but are not used by the serving stack. The SavedModel folders are used instead.

### `cnn_s1_saved/`, `tft_s1_saved/`, `cnn_s2_saved/`

The CPU-safe TensorFlow SavedModel exports. Each folder contains the full computation graph and weights in a format that is independent of Keras version and Python version. These are what `live_inference.py` actually loads at runtime.

### `upload.py`

A utility script used to upload the SavedModel folders to HuggingFace via the `huggingface_hub` Python API. Used during deployment when drag-and-drop upload was not practical for folder structures.

### `Data_preprocessing_eda/`

Contains raw and intermediate CSV files used during the data ingestion and preprocessing phase.

### `training_scripts_and_notebooks/`

Contains all Jupyter notebooks used during development, from raw data parsing through model training and export repair.

---

## Notebook Roles

| Notebook | Purpose |
|---|---|
| `creating_data.ipynb` | Parses NGL GNSS archives, merges with NASA POWER data, generates lag and cyclic features, exports master hourly CSV |
| `EDA_Notebook.ipynb` | Explores the dataset: rainfall distribution, seasonal patterns, feature correlations, class imbalance analysis |
| `3dnumpy_array_data_prep.ipynb` | Scales features and target, builds sliding window sequences, exports `X_sequences.npy` and `y_targets.npy` |
| `two_staged_cnn_lstm.ipynb` | Trains and evaluates the CNN-LSTM two-stage pipeline with chronological splits and class weighting |
| `Temporal_fusion_transformer.ipynb` | Trains and evaluates the TFT-based Stage 1 classifier using custom GRN, VSN, and TransformerBlock layers |
| `repairing_layers.ipynb` | Force-loads the original `.keras` files despite internal Keras naming changes and exports CPU-safe SavedModel folders |

---

## How To Run The API Locally

### Option 1: Direct Python

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

### Option 2: Docker

```bash
docker build -t gnss-weather-api .
docker run -p 7860:7860 gnss-weather-api
```

Then call:

- `http://localhost:7860/`
- `http://localhost:7860/predict/demo-storm`

---

## Known Practical Constraints

1. The API currently uses a fixed demo tensor. Real-time ingestion from live GNSS and NASA feeds is not yet implemented.
2. `feature_scaler.gz` is not used inside `live_inference.py`. Any real-time pipeline must scale incoming raw sensor data using this scaler before passing it to `predict_rain()`.
3. The TFT architecture was only used for Stage 1 classification. The Stage 2 regressor in the deployed stack is CNN-only.

---

## Summary

This project demonstrates a complete applied ML lifecycle for GNSS-informed rainfall forecasting: raw geophysical data engineering, long-horizon temporal feature construction, two-stage deep learning design, serialization and compatibility hardening across multiple Keras breaking changes, and API-level CPU deployment on HuggingFace Spaces.
