<p align="center">
  <img src="/assets/optanex-backend.jpg" alt="OptaNex Backend Banner" />
</p>

OptaNex Backend 🧠

AI-Powered Eye Screening API

The OptaNex backend provides FastAPI-based APIs for AI-powered preliminary eye health screening, supporting retinal disease analysis such as Diabetic Retinopathy (DR) and Age-Related Macular Degeneration (AMD).

This backend consumes trained deep-learning models and exposes inference endpoints used by the OptaNex frontend.

⚠️ Medical Disclaimer: This system provides screening assistance only and must not be used as a medical diagnostic system.

🛠 Tech Stack

FastAPI

Python 3.9+

Uvicorn

TensorFlow / Keras

ONNX Runtime

OpenCV, NumPy

📁 Backend Structure
backend/
│
├── 10_api/               # FastAPI routes & application logic
│   ├── main.py           # FastAPI app entry point
│   └── api.py            # API endpoints
│
├── 20_model_dump/        # Trained model artifacts
│   ├── *.h5              # TensorFlow/Keras models
│   └── *.onnx            # ONNX optimized models
│
├── 30_notebook/          # Model training & experimentation
│   └── *.ipynb           # Training and preprocessing notebooks
│
└── README.md             # Backend documentation

📦 Prerequisites

Make sure you have:

Python 3.9 or later

pip

virtual environment (recommended)

🧪 Setup & Installation
1️⃣ Create & Activate Virtual Environment
python -m venv venv


macOS / Linux

source venv/bin/activate


Windows

venv\Scripts\activate

2️⃣ Install Dependencies (Manual)

Since a requirements.txt file is not yet included, install dependencies manually:

pip install fastapi uvicorn tensorflow onnxruntime opencv-python numpy pillow


You may adjust this list depending on the model or API features you are using.

▶️ Running the Backend API

Navigate to the API folder and start the server:

cd 10_api
uvicorn main:app --reload


Backend will run at:

http://localhost:8000

📡 API Documentation

FastAPI automatically generates interactive docs:

Swagger UI

http://localhost:8000/docs


ReDoc

http://localhost:8000/redoc

🧠 Model Workflow

Model development & training occurs in:

30_notebook/


Trained models are exported as .h5 or .onnx

Production inference models are stored in:

20_model_dump/


APIs load models directly from 20_model_dump/

🔄 Updating Models

Train or update model in 30_notebook

Export trained model (.h5 or .onnx)

Move it to 20_model_dump/

Restart the backend server

🔗 Frontend Integration

Ensure frontend .env points to the backend:

VITE_BACKEND_URL=http://localhost:8000