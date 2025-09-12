from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import io
import joblib
import cv2
import pandas as pd

# -------------------------
# Initialize FastAPI
# -------------------------
app = FastAPI(title="Rice Nitrogen Predictor")

# Allow CORS (for Flutter app)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your frontend URL if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Load RandomForest model
# -------------------------
try:
    model = joblib.load("model.pkl")
    MODEL_LOADED = True
except Exception as e:
    print(f"Failed to load model: {e}")
    model = None
    MODEL_LOADED = False

# -------------------------
# Feature extraction function
# -------------------------
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

    return pd.DataFrame([{
        'R': mean_R, 'G': mean_G, 'B': mean_B,
        'nr': nr, 'ng': ng, 'nb': nb,
        'gmr': gmr, 'gmb': gmb,
        'L': L, 'a': a, 'b': b,
        'gdr': gdr, 'H_mean': H_mean, 'Y_mean': Y_mean, 'S': S, 'VI': VI
    }])

def extract_features_fastapi(image: Image.Image):
    image.save("temp.png")
    df = extract_features_opencv("temp.png")
    if df is not None:
        return df.values
    return None

# -------------------------
# Health check
# -------------------------
@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": MODEL_LOADED}

# -------------------------
# Prediction endpoint
# -------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not MODEL_LOADED:
        return {"success": False, "detail": "Model not loaded"}

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        features = extract_features_fastapi(image)

        if features is None:
            return {"success": False, "detail": "Failed to extract features"}

        prediction = model.predict(features)[0]

        # Decide nitrogen status & suggestion
        if prediction < 30:
            status = "Deficient"
            suggestion = "Apply nitrogen-rich fertilizer immediately."
        elif prediction < 50:
            status = "Moderate"
            suggestion = "Consider moderate nitrogen application."
        else:
            status = "Sufficient"
            suggestion = "Nitrogen level is sufficient. Maintain current management."

        return {
            "success": True,
            "prediction": float(prediction),
            "status": status,
            "suggestion": suggestion
        }

    except Exception as e:
        return {"success": False, "detail": f"Prediction failed: {str(e)}"}

# -------------------------
# Run locally
# -------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
