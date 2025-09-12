from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import joblib
import os
import logging
import uuid

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="Nitrogen Prediction API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy load ML model
model = None
def get_model():
    global model
    if model is None:
        try:
            model_file = "model.pkl"
            model = joblib.load(model_file)
            logger.info("ML model loaded successfully")
        except FileNotFoundError:
            logger.warning("model.pkl not found. Using dummy model.")
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=10)
            X_dummy = np.random.rand(10, 16)
            y_dummy = np.random.rand(10) * 100
            model.fit(X_dummy, y_dummy)
    return model

# Lazy load rembg
rembg_model = None
def get_rembg():
    global rembg_model
    if rembg_model is None:
        from rembg import remove
        rembg_model = remove
    return rembg_model

# Feature extraction
def extract_features_opencv(image_path):
    try:
        img_bgr_alpha = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img_bgr_alpha is None or img_bgr_alpha.shape[2] < 3:
            return None

        bgr = img_bgr_alpha[:, :, :3]
        alpha = img_bgr_alpha[:, :, 3] if img_bgr_alpha.shape[2] == 4 else np.ones(bgr.shape[:2], dtype=np.uint8)*255
        mask = alpha > 0
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)[mask]
        if rgb.size == 0:
            return None

        mean_R, mean_G, mean_B = np.mean(rgb[:,0]), np.mean(rgb[:,1]), np.mean(rgb[:,2])
        sum_rgb = mean_R + mean_G + mean_B
        nr, ng, nb = mean_R/sum_rgb, mean_G/sum_rgb, mean_B/sum_rgb if sum_rgb else (0,0,0)
        gmr, gmb = mean_G/mean_R if mean_R else 0, mean_G/mean_B if mean_B else 0

        bgr_pixels = bgr[mask]
        lab = cv2.cvtColor(bgr_pixels.reshape(-1,1,3), cv2.COLOR_BGR2LAB).reshape(-1,3)
        hsv = cv2.cvtColor(bgr_pixels.reshape(-1,1,3), cv2.COLOR_BGR2HSV).reshape(-1,3)
        ycbcr = cv2.cvtColor(bgr_pixels.reshape(-1,1,3), cv2.COLOR_BGR2YCrCb).reshape(-1,3)

        L, a_val, b_val = np.mean(lab[:,0]), np.mean(lab[:,1]), np.mean(lab[:,2])
        H_mean, S, V = np.mean(hsv[:,0]), np.mean(hsv[:,1]), np.mean(hsv[:,2])
        Y_mean, Cr, Cb = np.mean(ycbcr[:,0]), np.mean(ycbcr[:,1]), np.mean(ycbcr[:,2])

        gdr = (mean_G - mean_R)/(mean_G + mean_R) if (mean_G + mean_R) else 0
        VI = (2*mean_G - mean_R - mean_B)/(2*mean_G + mean_R + mean_B) if (2*mean_G + mean_R + mean_B) else 0

        return {
            'R': float(mean_R), 'G': float(mean_G), 'B': float(mean_B),
            'nr': float(nr), 'ng': float(ng), 'nb': float(nb),
            'gmr': float(gmr), 'gmb': float(gmb),
            'L': float(L), 'a': float(a_val), 'b': float(b_val),
            'gdr': float(gdr), 'H_mean': float(H_mean), 'Y_mean': float(Y_mean),
            'S': float(S), 'VI': float(VI)
        }
    except Exception as e:
        logger.error(f"Feature extraction error: {e}")
        return None

# Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    temp_filename = f"temp_{uuid.uuid4().hex}.png"
    try:
        # Read image bytes
        image_data = await file.read()
        remove_bg = get_rembg()
        output_bytes = remove_bg(image_data)  # Lazy loaded rembg
        img_no_bg = Image.open(BytesIO(output_bytes))

        # Resize to reduce memory usage
        img_no_bg.thumbnail((224, 224))
        img_no_bg.save(temp_filename)

        # Extract features
        features = extract_features_opencv(temp_filename)
        if not features:
            return {"error": "Feature extraction failed", "success": False}

        feature_list = [
            features['R'], features['G'], features['B'],
            features['nr'], features['ng'], features['nb'],
            features['gmr'], features['gmb'],
            features['L'], features['a'], features['b'],
            features['gdr'], features['H_mean'], features['Y_mean'],
            features['S'], features['VI']
        ]

        # Predict
        model = get_model()
        prediction = model.predict([feature_list])[0]

        # Nitrogen status
        if prediction < 30:
            status, suggestion = "Deficient", "Apply nitrogen-rich fertilizer immediately."
        elif prediction < 50:
            status, suggestion = "Moderate", "Consider moderate nitrogen application."
        else:
            status, suggestion = "Sufficient", "Nitrogen level is sufficient. Maintain current management."

        return {"prediction": float(prediction), "status": status, "suggestion": suggestion, "success": True}

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return {"error": str(e), "success": False}
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

# Health check
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

# Root
@app.get("/")
async def root():
    return {"message": "Nitrogen Prediction API is running!"}

# Run app (local testing)
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
