from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import uvicorn

# Import your AI engine
from live_inference import predict_rain 

app = FastAPI(title="GNSS Weather Prediction API")

# CRITICAL: Allow Netlify to communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows any frontend to call this API
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your pre-saved demo data when the server boots up
STORM_DATA = np.load("demo_storm_data.npy")

@app.get("/")
def home():
    return {"message": "GNSS Weather API is running and ready for predictions."}

@app.get("/predict/demo-storm")
def predict_demo_storm():
    """Endpoint triggered by your Netlify dashboard"""
    try:
        # Pass the historical data into the AI
        forecast = predict_rain(STORM_DATA)
        return {"status": "success", "data": forecast}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)