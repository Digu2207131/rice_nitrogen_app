from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import joblib
from PIL import Image
import io

# -------------------------
# Initialize FastAPI app
# -------------------------
app = FastAPI(title="Rice Nitrogen Predictor")

# Allow CORS for Flutter app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or your frontend URL
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
def extract_features(image: Image.Image):
    """
    Extract 16 features from the image to match your model training.
    Replace the placeholders with actual calculations.
    """
    image = image.convert("RGB").resize((100, 100))  # Resize to training size
    arr = np.array(image)
    
    features = np.zeros(16)
    
    # Example placeholders (replace with your actual features)
    features[0] = arr[:, :, 0].mean()   # mean R
    features[1] = arr[:, :, 1].mean()   # mean G
    features[2] = arr[:, :, 2].mean()   # mean B
    features[3] = arr[:, :, 0].std()    # std R
    features[4] = arr[:, :, 1].std()    # std G
    features[5] = arr[:, :, 2].std()    # std B
    # Fill in features[6] to features[15] with your actual 16 features
    # For example: texture metrics, color indices, etc.
    
    return features.reshape(1, -1)  # Ensure 2D input for RandomForest

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
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Extract features
        X = extract_features(image)

        # Predict using RandomForestRegressor
        prediction = model.predict(X)[0]

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
# Run app locally
# -------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
