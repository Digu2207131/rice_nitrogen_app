from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from rembg import remove
from io import BytesIO
from PIL import Image
import cv2
import numpy as np
import joblib
import os
import logging

# -------------------------------
# Logging
# -------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------
# App and CORS
# -------------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Load model
# -------------------------------
try:
    model = joblib.load("model.pkl")
    logger.info("✅ Model loaded successfully")
except Exception as e:
    logger.error(f"❌ Failed to load model: {e}")
    model = None

# -------------------------------
# Feature extraction (same as Streamlit)
# -------------------------------
def extract_features_opencv(image_path):
    img_bgr_alpha = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img_bgr_alpha is None or img_bgr_alpha.shape[2] < 3:
        return None

    if img_bgr_alpha.shape[2] == 4:
        bgr = img_bgr_alpha[:, :, :3]
        alpha = img_bgr_alpha[:, :, 3]
    else:
        bgr = img_bgr_alpha
        alpha = np.ones(bgr.shape[:2], dtype=np.uint8) * 255

    mask = alpha > 0
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)[mask]
    if rgb.size == 0:
        return None

    mean_R = np.mean(rgb[:, 0])
    mean_G = np.mean(rgb[:, 1])
    mean_B = np.mean(rgb[:, 2])
    sum_rgb = mean_R + mean_G + mean_B

    nr = mean_R / sum_rgb if sum_rgb else 0
    ng = mean_G / sum_rgb if sum_rgb else 0
    nb = mean_B / sum_rgb if sum_rgb else 0

    gmr = mean_G / mean_R if mean_R else 0
    gmb = mean_G / mean_B if mean_B else 0

    bgr_pixels = bgr[mask]
    lab = cv2.cvtColor(bgr_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
    hsv = cv2.cvtColor(bgr_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    ycbcr = cv2.cvtColor(bgr_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2YCrCb).reshape(-1, 3)

    L, a, b = np.mean(lab[:, 0]), np.mean(lab[:, 1]), np.mean(lab[:, 2])
    H_mean, S, Y_mean = np.mean(hsv[:, 0]), np.mean(hsv[:, 1]), np.mean(ycbcr[:, 0])
    gdr = (mean_G - mean_R) / (mean_G + mean_R) if (mean_G + mean_R) else 0
    VI = (2 * mean_G - mean_R - mean_B) / (2 * mean_G + mean_R + mean_B) if (2 * mean_G + mean_R + mean_B) else 0

    # Return feature array in correct order
    feature_list = [
        mean_R, mean_G, mean_B,
        nr, ng, nb,
        gmr, gmb,
        L, a, b,
        gdr, H_mean, Y_mean,
        S, VI
    ]
    return np.array(feature_list).reshape(1, -1)

# -------------------------------
# Prediction endpoint
# -------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not loaded"}

    try:
        # Read file and remove background
        image_bytes = await file.read()
        output_bytes = remove(image_bytes)
        img_no_bg = Image.open(BytesIO(output_bytes))

        # Save temporarily for OpenCV
        temp_path = "temp.png"
        img_no_bg.save(temp_path)

        # Extract features
        features = extract_features_opencv(temp_path)
        if features is None:
            return {"error": "Feature extraction failed"}

        # Predict
        spad_value = float(model.predict(features)[0])

        # Nitrogen status
        if spad_value < 30:
            status = "Deficient"
            suggestion = "Apply nitrogen-rich fertilizer immediately."
        elif spad_value < 50:
            status = "Moderate"
            suggestion = "Consider moderate nitrogen application."
        else:
            status = "Sufficient"
            suggestion = "Nitrogen level is sufficient. Maintain current practices."

        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)

        return {
            "prediction": spad_value,
            "status": status,
            "suggestion": suggestion
        }

    except Exception as e:
        return {"error": str(e)}

# -------------------------------
# Root endpoint
# -------------------------------
@app.get("/")
async def root():
    return {"message": "Rice Nitrogen Prediction API running!"}

# -------------------------------
# Run server
# -------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

