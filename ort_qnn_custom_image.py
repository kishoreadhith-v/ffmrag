# File_name : ort_qnn_custom_image.py

import onnxruntime as ort
import numpy as np
import time
import io_utils
import sys
import os

# Step1: Runtime and model initialization
# Set QNN Execution Provider options.
execution_provider_option = {
    "backend_path": "QnnHtp.dll",
    "enable_htp_fp16_precision": "1",
    "htp_performance_mode": "high_performance"
}

# Use your ONNX model file here
onnx_model_path = "./ImageClassifier.onnx"

session = ort.InferenceSession(
    onnx_model_path,
    providers=["QNNExecutionProvider"],
    provider_options=[execution_provider_option]
)

# Step2: Input/Output handling
# Check if user provided an image path as command line argument
if len(sys.argv) > 1:
    img_path = sys.argv[1]
    print(f"Using provided image: {img_path}")
    if not os.path.exists(img_path):
        print(f"Error: Image file '{img_path}' not found!")
        sys.exit(1)
else:
    # Default image if no argument provided
    img_path = "https://raw.githubusercontent.com/quic/wos-ai/refs/heads/main/Artifacts/coffee_cup.jpg"
    print("Using default coffee cup image")

# Preprocess the image
try:
    raw_img = io_utils.preprocess(img_path, target_size=(336, 336))  # ViT expects 336x336
    print(f"Image preprocessed successfully. Shape: {raw_img.shape}")
except Exception as e:
    print(f"Error preprocessing image: {e}")
    sys.exit(1)

# Model input and output names
outputs = session.get_outputs()[0].name
inputs = session.get_inputs()[0].name

print(f"Model input name: {inputs}")
print(f"Model output name: {outputs}")
print(f"Expected input shape: {session.get_inputs()[0].shape}")
print(f"Expected output shape: {session.get_outputs()[0].shape}")

# Step3: Model inferencing using preprocessed input.
print("Running inference...")
start_time = time.time()
prediction = session.run([outputs], {inputs: raw_img})
end_time = time.time()
execution_time = (end_time - start_time) * 1000

# Step4: Output postprocessing
print("\n" + "="*50)
print("PREDICTION RESULTS")
print("="*50)
io_utils.postprocess(prediction)
print(f"Execution Time: {execution_time:.2f} ms")
print("="*50)
