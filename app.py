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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Allow all origins for mobile app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model with better error handling
try:
    model = joblib.load("model.pkl")
    logger.info("Model loaded successfully")
except FileNotFoundError:
    logger.warning("model.pkl not found. Creating dummy model for testing.")
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=10)
    # Train with dummy data
    X_dummy = np.random.rand(10, 16)
    y_dummy = np.random.rand(10) * 100
    model.fit(X_dummy, y_dummy)
    logger.info("Dummy model created for testing")

# Feature extraction function - EXACTLY 16 FEATURES
def extract_features_opencv(image_path):
    try:
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

        L, a_val, b_val = np.mean(lab[:, 0]), np.mean(lab[:, 1]), np.mean(lab[:, 2])
        H_mean, S, V = np.mean(hsv[:, 0]), np.mean(hsv[:, 1]), np.mean(hsv[:, 2])
        Y_mean, Cr, Cb = np.mean(ycbcr[:, 0]), np.mean(ycbcr[:, 1]), np.mean(ycbcr[:, 2])

        gdr = (mean_G - mean_R) / (mean_G + mean_R) if (mean_G + mean_R) else 0
        VI = (2 * mean_G - mean_R - mean_B) / (2 * mean_G + mean_R + mean_B) if (2 * mean_G + mean_R + mean_B) else 0

        return {
            'R': float(mean_R), 'G': float(mean_G), 'B': float(mean_B),
            'nr': float(nr), 'ng': float(ng), 'nb': float(nb),
            'gmr': float(gmr), 'gmb': float(gmb),
            'L': float(L), 'a': float(a_val), 'b': float(b_val),
            'gdr': float(gdr), 'H_mean': float(H_mean), 'Y_mean': float(Y_mean),
            'S': float(S), 'VI': float(VI)
        }
    except Exception as e:
        logger.error(f"Error in feature extraction: {e}")
        return None

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        logger.info(f"Received prediction request for file: {file.filename}")

        # Read uploaded file
        image_data = await file.read()
        logger.info(f"File read successfully, size: {len(image_data)} bytes")

        # Remove background
        output_bytes = remove(image_data)
        img_no_bg = Image.open(BytesIO(output_bytes))
        logger.info("Background removed successfully")

        # Save temporary image
        temp_path = "temp_upload.png"
        img_no_bg.save(temp_path)
        logger.info(f"Temporary image saved: {temp_path}")

        # Extract features
        features = extract_features_opencv(temp_path)
        if features is None:
            logger.error("Feature extraction failed")
            return {"error": "Could not process image - feature extraction failed"}

        logger.info("Features extracted successfully")
        logger.info(f"Number of features extracted: {len(features)}")
        logger.info(f"Feature names: {list(features.keys())}")

        # Prepare features for prediction - EXACTLY 16 FEATURES
        feature_list = [
            features['R'], features['G'], features['B'],
            features['nr'], features['ng'], features['nb'],
            features['gmr'], features['gmb'],
            features['L'], features['a'], features['b'],
            features['gdr'], features['H_mean'], features['Y_mean'],
            features['S'], features['VI']
        ]

        logger.info(f"Number of features in prediction list: {len(feature_list)}")

        # Make prediction
        prediction = model.predict([feature_list])[0]
        logger.info(f"Prediction made: {prediction}")

        # Determine nitrogen status
        if prediction < 30:
            status = "Deficient"
            suggestion = "Apply nitrogen-rich fertilizer immediately."
        elif prediction < 50:
            status = "Moderate"
            suggestion = "Consider moderate nitrogen application."
        else:
            status = "Sufficient"
            suggestion = "Nitrogen level is sufficient. Maintain current management."

        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
            logger.info("Temporary file cleaned up")

        return {
            "prediction": float(prediction),
            "status": status,
            "suggestion": suggestion,
            "success": True
        }

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return {"error": str(e), "success": False}

@app.get("/")
async def root():
    return {"message": "Nitrogen Prediction API is running!"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "nitrogen-prediction-api",
        "model_loaded": hasattr(model, 'predict')
    }

@app.get("/debug-features")
async def debug_features():
    features = extract_features_opencv("temp_upload.png")
    if features:
        return {
            "number_of_features": len(features),
            "feature_names": list(features.keys()),
            "features": features
        }
    else:
        return {"error": "No features extracted - upload an image first"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # ✅ Use Render's $PORT
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
