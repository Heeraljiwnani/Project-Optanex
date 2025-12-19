from fastapi import FastAPI, UploadFile, File
import onnxruntime as ort
import numpy as np
import cv2
import io
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware


# === Load ONNX model ===
onnx_path = "resnet50_best.onnx"   # make sure you converted & saved it
session = ort.InferenceSession(onnx_path)

# Get input/output names from the model
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

IMG_SIZE = (224, 224)

# === Preprocessing (same as training) ===
def resize_image(img, size=(224, 224)):
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

def pad_to_square(img, color=(0, 0, 0)):
    h, w = img.shape[:2]
    size = max(h, w)
    delta_w = size - w
    delta_h = size - h
    top, bottom = delta_h // 2, delta_h - delta_h // 2
    left, right = delta_w // 2, delta_w - delta_w // 2
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

def apply_circular_crop(img):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    radius = min(center[0], center[1])
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, center, radius, (1,), thickness=-1)
    return cv2.bitwise_and(img, img, mask=mask)

def normalize_image(img):
    return img.astype(np.float32) / 255.0

def preprocess_image_bytes(image_bytes, size=(224, 224)):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = np.array(img)

    img = pad_to_square(img)
    img = resize_image(img, size)
    img = apply_circular_crop(img)
    img = apply_clahe(img)
    img = normalize_image(img)

    return np.expand_dims(img, axis=0).astype(np.float32)  # add batch dim

# === FastAPI App ===
app = FastAPI(title="ResNet50 Retina Classifier API (ONNX)")
# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or set specific domains ["https://example.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
def home():
    return {"message": "✅ Retina classifier ONNX API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img_array = preprocess_image_bytes(contents, size=IMG_SIZE)

    # Run ONNX inference
    preds = session.run([output_name], {input_name: img_array})[0]

    class_id = int(np.argmax(preds, axis=1)[0])
    confidence = float(np.max(preds))
    label_map = {
        0: "No DR",
        1: "Mild",
        2: "Moderate",
        3: "Severe",
        4: "Proliferative DR"
    }
    return {"class_name": label_map[class_id], "class_id": class_id, "confidence": confidence}
