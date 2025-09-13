from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import joblib
import os
import logging

# ------------------------------
# Logging setup
# ------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------
# FastAPI app setup
# ------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------
# Load trained model
# ------------------------------
try:
    model = joblib.load("model.pkl")
    logger.info("✅ Model loaded successfully")
except Exception as e:
    logger.error(f"❌ Failed to load model.pkl: {e}")
    model = None  # Avoid crash, use dummy fallback if needed

# ------------------------------
# Feature extraction (16 specific features)
# ------------------------------
def extract_features_opencv(image_path: str):
    """Extract specific 16 features from the leaf image."""
    img = cv2.imread(image_path)
    if img is None:
        return [0.0]*16

    # Ensure 3 channels
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] > 3:
        img = img[:, :, :3]

    pixels = img.reshape(-1, 3).astype(np.float32)
    mean_R, mean_G, mean_B = np.mean(pixels[:, 2]), np.mean(pixels[:, 1]), np.mean(pixels[:, 0])
    sum_rgb = mean_R + mean_G + mean_B + 1e-6
    nr, ng, nb = mean_R / sum_rgb, mean_G / sum_rgb, mean_B / sum_rgb
    gmr, gmb = mean_G / (mean_R + 1e-6), mean_G / (mean_B + 1e-6)

    # Convert to LAB, HSV, YCrCb
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).reshape(-1, 3)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb).reshape(-1, 3)

    L, a_val, b_val = np.mean(lab[:,0]), np.mean(lab[:,1]), np.mean(lab[:,2])
    H_mean, S, Y_mean = np.mean(hsv[:,0]), np.mean(hsv[:,1]), np.mean(ycbcr[:,0])
    gdr = (mean_G - mean_R)/(mean_G + mean_R + 1e-6)
    VI = (2*mean_G - mean_R - mean_B)/(2*mean_G + mean_R + mean_B + 1e-6)

    return [
        float(mean_R), float(mean_G), float(mean_B),
        float(nr), float(ng), float(nb),
        float(gmr), float(gmb),
        float(L), float(a_val), float(b_val),
        float(gdr), float(H_mean), float(Y_mean),
        float(S), float(VI)
    ]

# ------------------------------
# Predict endpoint
# ------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict nitrogen status from uploaded leaf image."""
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        temp_path = "temp_upload.png"
        image.save(temp_path)

        features = extract_features_opencv(temp_path)
        if model:
            prediction = model.predict([features])[0]
        else:
            prediction = 42.0  # Dummy value if model not loaded

        # Determine nitrogen status
        if prediction < 30:
            status, suggestion, color = "Deficient", "Apply nitrogen-rich fertilizer immediately.", "red"
        elif prediction < 50:
            status, suggestion, color = "Moderate", "Consider moderate nitrogen application.", "orange"
        else:
            status, suggestion, color = "Sufficient", "Nitrogen level is sufficient. Maintain current management.", "green"

        if os.path.exists(temp_path):
            os.remove(temp_path)

        return {
            "prediction": float(prediction),
            "status": status,
            "suggestion": suggestion,
            "color": color,
            "success": True
        }

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {"error": str(e), "success": False}

# ------------------------------
# Root & health endpoints
# ------------------------------
@app.get("/")
async def root():
    return {"message": "Nitrogen Prediction API running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

# ------------------------------
# Run locally or on Render
# ------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
