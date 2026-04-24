import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import gzip
import io

# ====================================================================
# 1. CUSTOM LAYERS (Required so the AI doesn't crash on load)
# ====================================================================
class TemporalSumLayer(tf.keras.layers.Layer):
    def call(self, x): return tf.reduce_sum(x, axis=1)

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, **kwargs): super().__init__(**kwargs)
    def build(self, input_shape):
        time_steps, d_model = input_shape[1], input_shape[2]
        angles = np.arange(time_steps)[:, np.newaxis] / np.power(10000, (2*(np.arange(d_model)[np.newaxis, :]//2)) / d_model)
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        self.pe = tf.constant(angles[np.newaxis], dtype=tf.float32)
        super().build(input_shape)
    def call(self, x): return x + self.pe

class GatedResidualNetwork(tf.keras.layers.Layer):
    def __init__(self, units=128, dropout_rate=0.2, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dense_elu = tf.keras.layers.Dense(units, activation='elu')
        self.dense_linear = tf.keras.layers.Dense(units)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dense_gate = tf.keras.layers.Dense(units, activation='sigmoid')
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.dense_proj = None
    def build(self, input_shape):
        if input_shape[-1] != self.units: self.dense_proj = tf.keras.layers.Dense(self.units)
        super().build(input_shape)
    def call(self, x, training=False):
        residual = x if self.dense_proj is None else self.dense_proj(x)
        h = self.dense_linear(self.dense_elu(x))
        h = self.dropout(h, training=training) * self.dense_gate(h)
        return self.layer_norm(h + residual)

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model=128, n_heads=4, ff_dim=128, dropout_rate=0.2, **kwargs):
        super().__init__(**kwargs)
        self.attn = tf.keras.layers.MultiHeadAttention(num_heads=n_heads, key_dim=d_model//n_heads, dropout=dropout_rate)
        self.drop1, self.drop2, self.drop3 = [tf.keras.layers.Dropout(dropout_rate) for _ in range(3)]
        self.norm1, self.norm2 = [tf.keras.layers.LayerNormalization(epsilon=1e-6) for _ in range(2)]
        self.ff1 = tf.keras.layers.Dense(ff_dim, activation='gelu')
        self.ff2 = tf.keras.layers.Dense(d_model)
    def call(self, x, training=False):
        x = self.norm1(x + self.drop1(self.attn(x, x, training=training), training=training))
        return self.norm2(x + self.drop3(self.ff2(self.drop2(self.ff1(x), training=training)), training=training))

class VariableSelectionNetwork(tf.keras.layers.Layer):
    def __init__(self, n_features=13, d_model=128, dropout_rate=0.2, **kwargs):
        super().__init__(**kwargs)
        self.n_features = n_features
        self.feature_projs = [tf.keras.layers.Dense(d_model) for _ in range(n_features)]
        self.feature_grns = [GatedResidualNetwork(d_model, dropout_rate) for _ in range(n_features)]
        self.selection_proj = tf.keras.layers.Dense(d_model)
        self.selection_grn = GatedResidualNetwork(d_model, dropout_rate)
        self.selection_gate = tf.keras.layers.Dense(n_features, activation='softmax')
    def call(self, x, training=False):
        var_outs = [self.feature_grns[i](self.feature_projs[i](x[:, :, i:i+1]), training=training) for i in range(self.n_features)]
        weights = tf.expand_dims(self.selection_gate(self.selection_grn(self.selection_proj(x), training=training)), axis=-1)
        return tf.reduce_sum(tf.stack(var_outs, axis=2) * weights, axis=2)

CUSTOM_OBJECTS = {
    'TemporalSumLayer': TemporalSumLayer,
    'PositionalEncoding': PositionalEncoding,
    'GatedResidualNetwork': GatedResidualNetwork,
    'TransformerBlock': TransformerBlock,
    'VariableSelectionNetwork': VariableSelectionNetwork
}

# ====================================================================
# 2. LOAD YOUR FILES 
# ====================================================================
def load_scaler(path):
    with gzip.open(path, 'rb') as f:
        return joblib.load(io.BytesIO(f.read()))

# Load Translators (Scalers)
target_scaler = load_scaler("target_scaler.gz")

# Load Brains (Models)
cnn_s1 = load_model("stage1_classifier.keras", custom_objects=CUSTOM_OBJECTS, compile=False)
tft_s1 = load_model("tft_stage1_classifier.keras", custom_objects=CUSTOM_OBJECTS, compile=False)
cnn_s2 = load_model("stage2_regressor.keras", custom_objects=CUSTOM_OBJECTS, compile=False)


# ====================================================================
# 3. THE 90/10 ENSEMBLE ENGINE
# ====================================================================
def predict_rain(live_48hr_data):
    """Feeds demo data into the models and outputs the exact forecast."""
    
    # 1. Ask the Gatekeepers (Probability of Rain)
    p_cnn = float(cnn_s1.predict(live_48hr_data, verbose=0)[0][0])
    p_tft = float(tft_s1.predict(live_48hr_data, verbose=0)[0][0])
    
    # The Math (90% TFT, 10% CNN)
    ensemble_prob = (0.10 * p_cnn) + (0.90 * p_tft)
    will_rain = ensemble_prob > 0.5

    # 2. Ask the Specialist (Volume of Rain)
    amount_mm = 0.0
    if will_rain:
        scaled_amount = cnn_s2.predict(live_48hr_data, verbose=0)
        scaled_amount = np.clip(scaled_amount, 0, 1) # Prevent negative glitches
        amount_mm = float(target_scaler.inverse_transform(scaled_amount)[0][0])
        amount_mm = max(0.0, amount_mm)

    return {
        'status': 'RAIN EXPECTED' if will_rain else 'CLEAR',
        'probability': f"{round(ensemble_prob * 100, 1)}%",
        'predicted_amount': f"{round(amount_mm, 2)} mm/hr"
    }