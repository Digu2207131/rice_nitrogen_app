from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import numpy as np
import joblib
from PIL import Image
import io

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
# Prediction endpoint
# -------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not MODEL_LOADED:
        return {"success": False, "detail": "Model not loaded"}

    try:
        # Read the uploaded image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Resize or preprocess if needed
        # Example: flatten RGB values for RandomForest
        img_array = np.array(image)
        img_flat = img_array.flatten().reshape(1, -1)  # Ensure 2D input

        # Predict using RandomForestRegressor
        prediction = model.predict(img_flat)[0]

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
