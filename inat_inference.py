# File_name : inat_inference.py

import onnxruntime as ort
import numpy as np
import time
import io_utils
import os
import sys

# Simple iNaturalist 2021 class mapping for the classes your model predicts
# Based on the assumption that class 4682 is a lion species
INAT_CLASS_MAPPING = {
    4682: "Panthera leo (Lion)",
    4672: "Panthera tigris (Tiger)", 
    4678: "Panthera pardus (Leopard)",
    4685: "Acinonyx jubatus (Cheetah)",
    4628: "Puma concolor (Mountain Lion/Puma)",
    4155: "Large Felid Species",
    3639: "Carnivora Species",
    # Add more as we discover them
}

def get_class_name(class_id):
    """Get the species name for a given class ID"""
    return INAT_CLASS_MAPPING.get(class_id, f"iNaturalist Species ID: {class_id}")

def enhanced_postprocess(prediction):
    """Enhanced postprocessing with iNaturalist species names"""
    pred = prediction[0]
    if isinstance(pred, list):
        pred = pred[0]
    pred = np.array(pred)
    
    # Get top 10 predictions to see more possibilities
    top10_idx = pred[0].argsort()[-10:][::-1]
    top10_probs = pred[0][top10_idx]
    
    print("Top-10 Species Predictions:")
    print("-" * 70)
    for i, (idx, prob) in enumerate(zip(top10_idx, top10_probs), 1):
        species_name = get_class_name(idx)
        confidence = prob * 100 if prob < 1 else prob  # Handle different probability scales
        print(f"{i:2d}. {species_name}")
        print(f"    Confidence: {confidence:.2f}% (Class ID: {idx})")
        print()

def test_multiple_images():
    """Test the model with multiple animal images"""
    
    # Step1: Initialize model
    print("Initializing iNaturalist 2021 model...")
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
    
    outputs = session.get_outputs()[0].name
    inputs = session.get_inputs()[0].name
    
    # Test images
    test_cases = [
        ("Lion Image", "C:\\Users\\Qualcomm\\ai survival guide\\image.png"),
        ("Coffee Cup (Default)", "https://raw.githubusercontent.com/quic/wos-ai/refs/heads/main/Artifacts/coffee_cup.jpg"),
    ]
    
    for test_name, img_path in test_cases:
        print("=" * 80)
        print(f"TESTING: {test_name}")
        print("=" * 80)
        
        try:
            # Check if local file exists
            if img_path.startswith("C:") and not os.path.exists(img_path):
                print(f"Skipping {test_name}: File not found")
                continue
                
            # Preprocess
            raw_img = io_utils.preprocess(img_path, target_size=(336, 336))
            print(f"Image shape: {raw_img.shape}")
            
            # Inference
            start_time = time.time()
            prediction = session.run([outputs], {inputs: raw_img})
            end_time = time.time()
            
            # Results
            enhanced_postprocess(prediction)
            print(f"Execution Time: {(end_time - start_time) * 1000:.2f} ms")
            
        except Exception as e:
            print(f"Error processing {test_name}: {e}")
        
        print()

def interactive_test():
    """Interactive testing with user input"""
    
    # Initialize model
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
    
    outputs = session.get_outputs()[0].name
    inputs = session.get_inputs()[0].name
    
    while True:
        print("\n" + "="*60)
        print("iNaturalist 2021 Species Identification")
        print("="*60)
        print("1. Test with local image file")
        print("2. Test with image URL")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ").strip()
        
        if choice == "3":
            break
        elif choice in ["1", "2"]:
            if choice == "1":
                img_path = input("Enter path to your image: ").strip()
            else:
                img_path = input("Enter image URL: ").strip()
            
            try:
                raw_img = io_utils.preprocess(img_path, target_size=(336, 336))
                
                start_time = time.time()
                prediction = session.run([outputs], {inputs: raw_img})
                end_time = time.time()
                
                print("\n" + "="*60)
                print("SPECIES IDENTIFICATION RESULTS")
                print("="*60)
                enhanced_postprocess(prediction)
                print(f"Processing Time: {(end_time - start_time) * 1000:.2f} ms")
                
            except Exception as e:
                print(f"Error: {e}")
        else:
            print("Invalid choice!")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_multiple_images()
    else:
        interactive_test()
