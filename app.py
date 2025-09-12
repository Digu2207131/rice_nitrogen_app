from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import joblib
from PIL import Image
import io
import numpy as np

app = FastAPI(title="Rice Nitrogen Prediction API")

# Global variable to store the model
model = None

# Load model on startup
@app.on_event("startup")
async def load_model():
    global model
    try:
        model = joblib.load("model.pkl")  # Ensure model.pkl is in the same folder
        print("Model loaded successfully")
    except Exception as e:
        print(f"Failed to load model: {e}")
        model = None

# Health endpoint
@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}

# Predict endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        # Resize or preprocess as per your model
        image = image.resize((128, 128))  # Example size
        image_array = np.array(image) / 255.0
        image_array = image_array.reshape(1, 128, 128, 3)  # Example for CNN

        # Make prediction
        prediction = model.predict(image_array)
        predicted_value = float(prediction[0])

        return JSONResponse(content={"prediction": predicted_value})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


# For local testing
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

