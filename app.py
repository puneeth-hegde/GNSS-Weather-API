from fastapi import FastAPI
import numpy as np
import uvicorn

# Import your AI engine
from live_inference import predict_rain 

app = FastAPI(title="GNSS Weather Prediction API")

# Load the real storm data you just extracted from Colab
STORM_DATA = np.load("demo_scenarios.npy")

@app.get("/")
def home():
    return {"message": "GNSS Weather API is running."}

@app.get("/predict/demo-storm")
def predict_demo_storm():
    """Simple GET endpoint to prove the model predicts rain"""
    try:
        # Feed the real (1, 48, 13) storm data to the model
        forecast = predict_rain(STORM_DATA)
        return {"status": "success", "data": forecast}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    # Hugging Face requires port 7860
    uvicorn.run(app, host="0.0.0.0", port=7860)