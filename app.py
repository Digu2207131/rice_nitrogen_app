from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
from rembg import remove
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
    raise e

# ------------------------------
# Feature extraction (16 features)
# ------------------------------
def extract_features_opencv_safe(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return [0.0]*16

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] > 3:
        img = img[:, :, :3]

    pixels = img.reshape(-1, 3).astype(np.float32)
    mean_R, mean_G, mean_B = np.mean(pixels[:, 2]), np.mean(pixels[:, 1]), np.mean(pixels[:, 0])

    sum_rgb = mean_R + mean_G + mean_B + 1e-6
    nr, ng, nb = mean_R / sum_rgb, mean_G / sum_rgb, mean_B / sum_rgb
    gmr, gmb = mean_G / (mean_R + 1e-6), mean_G / (mean_B + 1e-6)

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
    try:
        logger.info(f"📥 Received file: {file.filename}")
        image_data = await file.read()

        # Remove background
        output_bytes = remove(image_data)
        img_no_bg = Image.open(BytesIO(output_bytes))

        # Save temporary image
        temp_path = "temp_upload.png"
        img_no_bg.save(temp_path)

        # Extract features & predict
        feature_list = extract_features_opencv_safe(temp_path)
        prediction = model.predict([feature_list])[0]

        # Determine nitrogen status & color
        if prediction < 30:
            status, suggestion, color = "Deficient", "Apply nitrogen-rich fertilizer immediately.", "red"
        elif prediction < 50:
            status, suggestion, color = "Moderate", "Consider moderate nitrogen application.", "orange"
        else:
            status, suggestion, color = "Sufficient", "Nitrogen level is sufficient. Maintain current management.", "green"

        # Cleanup
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
        logger.error(f"❌ Prediction error: {e}")
        return {"error": str(e), "success": False}

# ------------------------------
# Root & health endpoints
# ------------------------------
@app.get("/")
async def root():
    return {"message": "Nitrogen Prediction API is running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": hasattr(model, 'predict')
    }

# ------------------------------
# Run locally or on Render
# ------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # Render automatically sets PORT
    uvicorn.run(app, host="0.0.0.0", port=port)

