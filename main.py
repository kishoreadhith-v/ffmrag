import onnxruntime as ort
import os

# Try to use NPU (QNN HTP) first, then fallback to CPU
providers = []
provider_options = []

# QNN HTP (NPU) provider options
htp_provider = (
    'QNNExecutionProvider',
    {
        'backend_path': 'QnnHtp.dll',
        'enable_htp_fp16_precision': '1',
        'htp_performance_mode': 'high_performance'
    }
)

# CPU provider fallback
cpu_provider = ('CPUExecutionProvider', {})

onnx_model_path = 'ImageClassifier.onnx'

# Try to create session with NPU first
try:
    session = ort.InferenceSession(
        onnx_model_path,
        providers=[htp_provider[0], cpu_provider[0]],
        provider_options=[htp_provider[1], cpu_provider[1]]
    )
    actual_provider = session.get_providers()[0]
    print(f"Model is running on: {actual_provider}")
    if actual_provider == 'CPUExecutionProvider':
        print("Model is running on CPU. Attempting to switch to NPU...")
        # Try to re-initialize with only NPU
        try:
            session = ort.InferenceSession(
                onnx_model_path,
                providers=[htp_provider[0]],
                provider_options=[htp_provider[1]]
            )
            actual_provider = session.get_providers()[0]
            print(f"Switched to: {actual_provider}")
        except Exception as e:
            print(f"Failed to switch to NPU: {e}")
    else:
        print("Model is running on NPU (QNN HTP)")
except Exception as e:
    print(f"Failed to initialize with NPU: {e}")
    print("Falling back to CPU provider...")
    session = ort.InferenceSession(
        onnx_model_path,
        providers=[cpu_provider[0]],
        provider_options=[cpu_provider[1]]
    )
    print("Model is running on CPU.")
