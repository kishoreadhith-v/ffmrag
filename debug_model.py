# File_name : debug_model.py

import onnxruntime as ort
import numpy as np
from PIL import Image
import sys

def preprocess_different_ways(img_path):
    """Try different preprocessing approaches"""
    img = Image.open(img_path).convert('RGB')
    img = img.resize((336, 336), Image.BILINEAR)
    
    preprocessing_methods = {}
    
    # Method 1: Standard ImageNet normalization
    img_array = np.array(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    method1 = (img_array - mean) / std
    method1 = np.transpose(method1, (2, 0, 1))
    method1 = np.expand_dims(method1, axis=0)
    preprocessing_methods["ImageNet_norm"] = method1
    
    # Method 2: Just normalize to [0,1]
    img_array = np.array(img).astype(np.float32) / 255.0
    method2 = np.transpose(img_array, (2, 0, 1))
    method2 = np.expand_dims(method2, axis=0)
    preprocessing_methods["Simple_norm"] = method2
    
    # Method 3: Scale to [-1, 1]
    img_array = np.array(img).astype(np.float32) / 255.0
    method3 = (img_array * 2.0) - 1.0
    method3 = np.transpose(method3, (2, 0, 1))
    method3 = np.expand_dims(method3, axis=0)
    preprocessing_methods["MinusOne_to_One"] = method3
    
    # Method 4: No normalization, just scale to [0,255]
    img_array = np.array(img).astype(np.float32)
    method4 = np.transpose(img_array, (2, 0, 1))
    method4 = np.expand_dims(method4, axis=0)
    preprocessing_methods["Raw_0_255"] = method4
    
    return preprocessing_methods

def analyze_predictions(prediction, method_name):
    """Analyze raw prediction values"""
    pred = prediction[0]
    if isinstance(pred, list):
        pred = pred[0]
    pred = np.array(pred)
    
    logits = pred[0] if len(pred.shape) > 1 else pred
    
    print(f"\n--- {method_name} ---")
    print(f"Raw output shape: {logits.shape}")
    print(f"Raw output range: [{np.min(logits):.4f}, {np.max(logits):.4f}]")
    print(f"Raw output mean: {np.mean(logits):.4f}")
    print(f"Raw output std: {np.std(logits):.4f}")
    
    # Get top 5 raw values (before softmax)
    top5_indices = np.argsort(logits)[-5:][::-1]
    top5_values = logits[top5_indices]
    
    print("Top 5 raw logits:")
    for i, (idx, val) in enumerate(zip(top5_indices, top5_values)):
        print(f"  {i+1}. Index {idx}: {val:.4f}")
    
    # Apply softmax
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)
    top5_probs = probs[top5_indices]
    
    print("Top 5 probabilities:")
    for i, (idx, prob) in enumerate(zip(top5_indices, top5_probs)):
        print(f"  {i+1}. Index {idx}: {prob*100:.2f}%")

def main():
    if len(sys.argv) < 2:
        print("Usage: python debug_model.py <image_path>")
        return
    
    img_path = sys.argv[1]
    
    # Initialize model
    print("Loading model...")
    session = ort.InferenceSession(
        "./ImageClassifier.onnx",
        providers=["CPUExecutionProvider"]  # Use CPU for debugging
    )
    
    inputs = session.get_inputs()[0].name
    outputs = session.get_outputs()[0].name
    
    print(f"Model input shape: {session.get_inputs()[0].shape}")
    print(f"Model output shape: {session.get_outputs()[0].shape}")
    
    # Try different preprocessing methods
    preprocessing_methods = preprocess_different_ways(img_path)
    
    print("\n" + "="*60)
    print("TESTING DIFFERENT PREPROCESSING METHODS")
    print("="*60)
    
    for method_name, processed_img in preprocessing_methods.items():
        try:
            prediction = session.run([outputs], {inputs: processed_img})
            analyze_predictions(prediction, method_name)
        except Exception as e:
            print(f"\n--- {method_name} ---")
            print(f"ERROR: {e}")

if __name__ == "__main__":
    main()
