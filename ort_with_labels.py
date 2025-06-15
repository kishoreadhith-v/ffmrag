# File_name : ort_with_labels.py

import onnxruntime as ort
import numpy as np
import time
import sys
import os
from PIL import Image
import requests
import io

def get_imagenet_labels():
    """Download ImageNet class labels"""
    try:
        # Try ImageNet-21K labels first (since model has 10k classes)
        print("Trying to download ImageNet-21K labels...")
        url = "https://storage.googleapis.com/bit_models/imagenet21k_wordnet_lemmas.txt"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            labels = response.text.strip().split('\n')
            print(f"✓ Downloaded ImageNet-21K labels: {len(labels)} classes")
            return labels
    except Exception as e:
        print(f"Failed to download ImageNet-21K labels: {e}")
    
    try:
        # Fallback to standard ImageNet-1K labels
        print("Falling back to ImageNet-1K labels...")
        url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            labels = response.text.strip().split('\n')
            print(f"✓ Downloaded ImageNet-1K labels: {len(labels)} classes")
            # Pad with unknown classes for the remaining indices
            while len(labels) < 10000:
                labels.append(f"unknown_class_{len(labels)}")
            return labels
    except Exception as e:
        print(f"Failed to download ImageNet-1K labels: {e}")
    
    # Final fallback
    print("Using generic class names...")
    return [f"class_{i}" for i in range(10000)]

def preprocess_image(img_path, input_size=(336, 336)):
    """Preprocess image using torchvision-style transforms"""
    # ImageNet normalization values
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    # Load image
    if img_path.startswith('http'):
        response = requests.get(img_path)
        img = Image.open(io.BytesIO(response.content)).convert('RGB')
    else:
        img = Image.open(img_path).convert('RGB')
    
    # Resize image
    img = img.resize(input_size, Image.BILINEAR)
    
    # Convert to numpy array and normalize to [0, 1]
    img_array = np.array(img).astype(np.float32) / 255.0
    
    # Normalize with ImageNet mean and std
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    img_array = (img_array - mean) / std
    
    # Convert from HWC to CHW format
    img_array = np.transpose(img_array, (2, 0, 1))
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def postprocess_predictions(prediction, labels, top_k=5):
    """Process predictions and return top-k results with labels"""
    pred = prediction[0]
    if isinstance(pred, list):
        pred = pred[0]
    pred = np.array(pred)
    
    # Apply softmax to get probabilities
    logits = pred[0] if len(pred.shape) > 1 else pred
    exp_logits = np.exp(logits - np.max(logits))  # Subtract max for numerical stability
    probs = exp_logits / np.sum(exp_logits)
    
    # Get top-k indices
    top_indices = np.argsort(probs)[-top_k:][::-1]
    top_probs = probs[top_indices]
    
    results = []
    for idx, prob in zip(top_indices, top_probs):
        if idx < len(labels):
            label = labels[idx]
        else:
            label = f"class_{idx}"
        results.append((label, prob * 100))
    
    return results

def main():
    # Step1: Initialize model
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
    
    # Get input image path
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        print(f"Using provided image: {img_path}")
        if not os.path.exists(img_path):
            print(f"Error: Image file '{img_path}' not found!")
            return
    else:
        img_path = input("Enter path to your image (or press Enter for default): ").strip()
        if not img_path:
            img_path = "https://raw.githubusercontent.com/quic/wos-ai/refs/heads/main/Artifacts/coffee_cup.jpg"
    
    print(f"Processing image: {img_path}")
    
    # Get ImageNet labels
    print("Loading class labels...")
    labels = get_imagenet_labels()
    print(f"✓ Loaded {len(labels)} class labels")
    
    # Preprocess image
    try:
        input_tensor = preprocess_image(img_path, input_size=(336, 336))
        print(f"✓ Image preprocessed successfully. Shape: {input_tensor.shape}")
    except Exception as e:
        print(f"✗ Error preprocessing image: {e}")
        return
    
    # Get model input/output info
    inputs = session.get_inputs()[0].name
    outputs = session.get_outputs()[0].name
    
    print(f"Model expects input shape: {session.get_inputs()[0].shape}")
    print(f"Model output shape: {session.get_outputs()[0].shape}")
    
    # Run inference
    print("Running inference...")
    start_time = time.time()
    prediction = session.run([outputs], {inputs: input_tensor})
    end_time = time.time()
    execution_time = (end_time - start_time) * 1000
    
    # Process results
    print("\n" + "="*70)
    print("PREDICTION RESULTS")
    print("="*70)
    
    results = postprocess_predictions(prediction, labels, top_k=5)
    
    for i, (label, confidence) in enumerate(results, 1):
        print(f"{i}. {label}: {confidence:.2f}%")
    
    print(f"\nExecution Time: {execution_time:.2f} ms")
    print("="*70)

if __name__ == "__main__":
    main()
