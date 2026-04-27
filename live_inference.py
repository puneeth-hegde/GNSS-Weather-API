import numpy as np
import tensorflow as tf
import joblib
import gzip
import io

def load_scaler(path):
    with gzip.open(path, 'rb') as f:
        return joblib.load(io.BytesIO(f.read()))

target_scaler = load_scaler("target_scaler.gz")

cnn_s1 = tf.saved_model.load("cnn_s1_saved")
tft_s1 = tf.saved_model.load("tft_s1_saved")
cnn_s2 = tf.saved_model.load("cnn_s2_saved")

def predict_rain(live_48hr_data):
    p_cnn = float(cnn_s1.serve(live_48hr_data)[0][0])
    p_tft = float(tft_s1.serve(live_48hr_data)[0][0])

    ensemble_prob = (0.10 * p_cnn) + (0.90 * p_tft)
    will_rain = ensemble_prob > 0.35

    amount_mm = 0.0
    status_text = "CLEAR"

    if will_rain:
        scaled_amount = cnn_s2.serve(live_48hr_data)
        scaled_amount = np.clip(scaled_amount, 0, 1)
        amount_mm = float(target_scaler.inverse_transform(scaled_amount)[0][0])
        amount_mm = max(0.0, amount_mm)

        if amount_mm > 5.0:
            status_text = "SEVERE STORM EXPECTED"
        else:
            status_text = "RAIN EXPECTED"

    return {
        'status': status_text,
        'probability': f"{round(ensemble_prob * 100, 1)}%",
        'predicted_amount': f"{round(amount_mm, 2)} mm/hr"
    }