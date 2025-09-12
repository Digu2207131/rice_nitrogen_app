from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import joblib
from PIL import Image
import io
import os

# -------------------------
# Initialize FastAPI app
# -------------------------
app = FastAPI(title="Rice Nitrogen Predictor")

# Allow CORS for your Flutter app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or use your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Load the model
# -------------------------
MODEL_LOADED = False
model_path = "model.pkl"

if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
        MODEL_LOADED = True
    except Exception as e:
        print(f"Failed to load model: {e}")
        model = None
else:
    print(f"Model file {model_path} not found")
    model = None

# -------------------------
# Health check endpoint
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
        # Read uploaded image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Resize or preprocess to match model input
        # Flatten RGB values for RandomForest
        img_array = np.array(image)
        img_flat = img_array.flatten().reshape(1, -1)  # Ensure 2D input

        # Predict
        prediction = model.predict(img_flat)[0]

        # Determine status & suggestion
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
# Run locally
# -------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=True)
