import joblib
import numpy as np
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI(title="Crop Recommendation API")

# -----------------------------
# CORS (Allow frontend access)
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to your frontend domain later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Load Model & Encoders Once
# -----------------------------
print("Loading model and encoders...")

MODEL_PATH = "models/plant_model_top30.keras"
OHE_PATH = "models/onehot_encoder_top30.pkl"
LABEL_PATH = "models/label_encoder_top30.pkl"

model = tf.keras.models.load_model(MODEL_PATH)
ohe = joblib.load(OHE_PATH)
label_encoder = joblib.load(LABEL_PATH)

print("Model Loaded Successfully!")

# -----------------------------
# Request Schema
# -----------------------------
class PlantInput(BaseModel):
    soil: str
    region: str
    environment: str

# -----------------------------
# Health Check Route
# -----------------------------
@app.get("/")
def health_check():
    return {"status": "API is running successfully"}

# -----------------------------
# Prediction Route
# -----------------------------
@app.post("/predict")
def predict(data: PlantInput):
    try:
        input_data = [[
            data.soil.lower(),
            data.region.lower(),
            data.environment.lower()
        ]]

        input_encoded = ohe.transform(input_data)

        probs = model.predict(input_encoded, verbose=0)[0]

        top_k = 3
        top_indices = np.argsort(probs)[::-1][:top_k]

        plants = label_encoder.inverse_transform(top_indices)
        confidences = probs[top_indices]

        result = []
        for p, c in zip(plants, confidences):
            result.append({
                "plant": p,
                "confidence": float(c)
            })

        return {"recommendations": result}

    except Exception as e:
        return {"error": str(e)}