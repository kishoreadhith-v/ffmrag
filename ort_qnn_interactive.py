# File_name : ort_qnn_interactive.py

import onnxruntime as ort
import numpy as np
import time
import io_utils
import os

def get_image_path():
    print("Choose an option:")
    print("1. Use a local image file")
    print("2. Use a URL to an image")
    print("3. Use default coffee cup image")
    
    choice = input("Enter your choice (1-3): ").strip()
    
    if choice == "1":
        img_path = input("Enter the path to your image file: ").strip()
        if not os.path.exists(img_path):
            print(f"Error: File '{img_path}' not found!")
            return None
        return img_path
    elif choice == "2":
        img_path = input("Enter the URL to your image: ").strip()
        return img_path
    elif choice == "3":
        return "https://raw.githubusercontent.com/quic/wos-ai/refs/heads/main/Artifacts/coffee_cup.jpg"
    else:
        print("Invalid choice!")
        return None

# Step1: Runtime and model initialization
print("Initializing ONNX Runtime with QNN EP...")
execution_provider_option = {
    "backend_path": "QnnHtp.dll",
    "enable_htp_fp16_precision": "1",
    "htp_performance_mode": "high_performance"
}

onnx_model_path = "./ImageClassifier.onnx"

session = ort.InferenceSession(
    onnx_model_path,
    providers=["QNNExecutionProvider"],
    provider_options=[execution_provider_option]
)

print("✓ Model loaded successfully!")

# Step2: Get image from user
img_path = get_image_path()
if img_path is None:
    print("Exiting...")
    exit(1)

print(f"Processing image: {img_path}")

# Preprocess the image
try:
    raw_img = io_utils.preprocess(img_path, target_size=(336, 336))
    print(f"✓ Image preprocessed successfully. Shape: {raw_img.shape}")
except Exception as e:
    print(f"✗ Error preprocessing image: {e}")
    exit(1)

# Model input and output names
outputs = session.get_outputs()[0].name
inputs = session.get_inputs()[0].name

# Step3: Model inference
print("Running inference...")
start_time = time.time()
prediction = session.run([outputs], {inputs: raw_img})
end_time = time.time()
execution_time = (end_time - start_time) * 1000

# Step4: Output postprocessing
print("\n" + "="*60)
print("PREDICTION RESULTS")
print("="*60)
io_utils.postprocess(prediction)
print(f"Execution Time: {execution_time:.2f} ms")
print("="*60)
