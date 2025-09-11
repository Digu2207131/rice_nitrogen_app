from setuptools import setup, find_packages

setup(
    name="nitrogen-prediction-api",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi==0.104.1",
        "uvicorn==0.24.0",
        "numpy==1.26.4",
        "opencv-python-headless==4.8.1.78",
        "Pillow==10.0.1",
        "scikit-learn==1.3.2",
        "joblib==1.3.2",
        "rembg==2.0.38",
        "python-multipart==0.0.6"
    ],
)