# File_name : ort_qnn_cpu.py

import onnxruntime as ort
import numpy as np
import time
import io_utils

# Step1: Runtime and model initialization
# Set QNN Execution Provider options for CPU backend
execution_provider_option = {
    "backend_path": "QnnCpu.dll",
    "htp_performance_mode": "high_performance"
}

# Use your ONNX model file here
onnx_model_path = "./ImageClassifier.onnx"

try:
    session = ort.InferenceSession(
        onnx_model_path,
        providers=["QNNExecutionProvider"],
        provider_options=[execution_provider_option]
    )
    print("QNN CPU provider loaded successfully")
except Exception as e:
    print(f"QNN provider failed: {e}")
    print("Falling back to CPU provider")
    session = ort.InferenceSession(
        onnx_model_path,
        providers=["CPUExecutionProvider"]
    )

# Step2: Input/Output handling, Generate raw input
img_path = "https://raw.githubusercontent.com/quic/wos-ai/refs/heads/main/Artifacts/coffee_cup.jpg"
raw_img = io_utils.preprocess(img_path)

# Model input and output names
outputs = session.get_outputs()[0].name
inputs = session.get_inputs()[0].name

print(f"Input name: {inputs}")
print(f"Output name: {outputs}")
print(f"Input shape: {session.get_inputs()[0].shape}")
print(f"Output shape: {session.get_outputs()[0].shape}")

# Step3: Model inferencing using preprocessed input.
start_time = time.time()
for i in range(10):
    prediction = session.run([outputs], {inputs: raw_img})
end_time = time.time()
execution_time = ((end_time - start_time) * 1000) / 10

# Step4: Output postprocessing
io_utils.postprocess(prediction)
print("Execution Time: ", execution_time, "ms")
