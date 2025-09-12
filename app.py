from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import numpy as np
import joblib
from PIL import Image
import io
from rembg import remove
import cv2
import pandas as pd

# -------------------------
# Initialize FastAPI app
# -------------------------
app = FastAPI(title="Rice Nitrogen Predictor")

# Allow CORS for your Flutter app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or put your frontend URL here
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Load the model
# -------------------------
try:
    model = joblib.load("model.pkl")
    MODEL_LOADED = True
except Exception as e:
    print(f"Failed to load model: {e}")
    model = None
    MODEL_LOADED = False

# -------------------------
# Health check endpoint
# -------------------------
@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": MODEL_LOADED}

# -------------------------
# Feature extraction function
# -------------------------
def extract_features_opencv(image: Image.Image):
    img_array = np.array(image)
    if img_array.ndim != 3 or img_array.shape[2] != 3:
        return None

    mean_R = np.mean(img_array[:, :, 0])
    mean_G = np.mean(img_array[:, :, 1])
    mean_B = np.mean(img_array[:, :, 2])
    sum_rgb = mean_R + mean_G + mean_B

    nr = mean_R / sum_rgb if sum_rgb else 0
    ng = mean_G / sum_rgb if sum_rgb else 0
    nb = mean_B / sum_rgb if sum_rgb else 0

    gmr = mean_G / mean_R if mean_R else 0
    gmb = mean_G / mean_B if mean_B else 0

    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    ycbcr = cv2.cvtColor(img_array, cv2.COLOR_RGB2YCrCb)

    L, a, b_val = np.mean(lab[:, :, 0]), np.mean(lab[:, :, 1]), np.mean(lab[:, :, 2])
    H_mean, S, Y_mean = np.mean(hsv[:, :, 0]), np.mean(hsv[:, :, 1]), np.mean(ycbcr[:, :, 0])
    gdr = (mean_G - mean_R) / (mean_G + mean_R) if (mean_G + mean_R) else 0
    VI = (2 * mean_G - mean_R - mean_B) / (2 * mean_G + mean_R + mean_B) if (2 * mean_G + mean_R + mean_B) else 0

    return pd.DataFrame([{
        'R': mean_R, 'G': mean_G, 'B': mean_B,
        'nr': nr, 'ng': ng, 'nb': nb,
        'gmr': gmr, 'gmb': gmb,
        'L': L, 'a': a, 'b': b_val,
        'gdr': gdr, 'H_mean': H_mean, 'Y_mean': Y_mean, 'S': S, 'VI': VI
    }])

# -------------------------
# Prediction endpoint
# -------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not MODEL_LOADED:
        return {"success": False, "detail": "Model not loaded"}

    try:
        # Read uploaded image
        contents = await file.read()
        # Remove background
        img_no_bg_bytes = remove(contents)
        image = Image.open(io.BytesIO(img_no_bg_bytes)).convert("RGB")

        # Extract features
        features_df = extract_features_opencv(image)
        if features_df is None:
            return {"success": False, "detail": "Failed to extract features"}

        # Predict
        prediction = model.predict(features_df)[0]

        # Decide status & suggestion
        if prediction < 20:
            status = "Low"
            suggestion = "Consider increasing nitrogen application."
        elif prediction < 40:
            status = "Moderate"
            suggestion = "Consider moderate nitrogen application."
        else:
            status = "High"
            suggestion = "Nitrogen level is sufficient."

        return {
            "success": True,
            "prediction": float(prediction),
            "status": status,
            "suggestion": suggestion
        }

    except Exception as e:
        return {"success": False, "detail": f"Prediction failed: {str(e)}"}

# -------------------------
# Run the app (local)
# -------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
