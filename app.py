from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import joblib
import os
import logging
from rembg import remove, new_session
import uuid

# ------------------------------ Logging ------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------ App setup ------------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------ Globals ------------------------------
model = None
rembg_session = None

@app.on_event("startup")
async def startup_event():
    """Load model and rembg session at startup"""
    global model, rembg_session
    # Load ML model
    if os.path.exists("model.pkl"):
        model = joblib.load("model.pkl")
        logger.info("✅ Model loaded successfully")
    else:
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=10)
        X_dummy = np.random.rand(20, 16)
        y_dummy = np.random.rand(20)
        model.fit(X_dummy, y_dummy)
        logger.warning("⚠️ model.pkl not found, using dummy model")
    
    # Lightweight rembg session
    rembg_session = new_session("u2netp")
    logger.info("✅ rembg session initialized (u2netp)")

# ------------------------------ Feature extraction ------------------------------
def extract_features(img: np.ndarray):
    """Compute 16 features from RGB image"""
    if img is None:
        return [0.0]*16

    # Ensure image is RGB
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] > 3:
        img = img[:, :, :3]

    pixels = img.reshape(-1, 3).astype(np.float32)
    mean_B, mean_G, mean_R = np.mean(pixels, axis=0)
    sum_rgb = mean_R + mean_G + mean_B + 1e-6

    # Normalized RGB
    nr, ng, nb = mean_R/sum_rgb, mean_G/sum_rgb, mean_B/sum_rgb

    # Ratios
    gmr, gmb = mean_G/(mean_R+1e-6), mean_G/(mean_B+1e-6)

    # Convert to LAB, HSV, YCrCb
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).reshape(-1, 3)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb).reshape(-1, 3)

    L, a_val, b_val = np.mean(lab, axis=0)
    H_mean, S = np.mean(hsv[:,0]), np.mean(hsv[:,1])
    Y_mean = np.mean(ycbcr[:,0])

    # Vegetation indices
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

# ------------------------------ Prediction ------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global model, rembg_session
    temp_file = f"temp_{uuid.uuid4().hex}.png"

    try:
        # Read image
        contents = await file.read()
        img = Image.open(BytesIO(contents)).convert("RGB")

        # Remove background
        img_no_bg = remove(img, session=rembg_session)
        img_no_bg.save(temp_file)

        # Load as numpy array
        img_cv = cv2.imread(temp_file)
        features = extract_features(img_cv)

        # Predict
        prediction = model.predict([features])[0]

        # Nitrogen status
        if prediction < 30:
            status, suggestion, color = "Deficient", "Apply nitrogen-rich fertilizer immediately.", "red"
        elif prediction < 50:
            status, suggestion, color = "Moderate", "Consider moderate nitrogen application.", "orange"
        else:
            status, suggestion, color = "Sufficient", "Nitrogen level is sufficient. Maintain current management.", "green"

        return {
            "prediction": float(prediction),
            "status": status,
            "suggestion": suggestion,
            "color": color,
            "success": True
        }

    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        return {"error": str(e), "success": False}

    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

# ------------------------------ Root & Health ------------------------------
@app.get("/")
async def root():
    return {"message": "Nitrogen Prediction API running ✅"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": hasattr(model, "predict")}

# ------------------------------ Entry point ------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, workers=1)

