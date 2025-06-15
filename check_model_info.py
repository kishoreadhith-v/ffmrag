# File_name : check_model_info.py

import onnx
import numpy as np

def analyze_onnx_model(model_path):
    """Analyze ONNX model to understand its structure"""
    try:
        model = onnx.load(model_path)
        
        print("="*60)
        print("MODEL ANALYSIS")
        print("="*60)
        
        # Get model metadata
        if hasattr(model, 'metadata_props'):
            print("Model Metadata:")
            for prop in model.metadata_props:
                print(f"  {prop.key}: {prop.value}")
            print()
        
        # Get input info
        print("Model Inputs:")
        for input_info in model.graph.input:
            print(f"  Name: {input_info.name}")
            print(f"  Type: {input_info.type}")
            if input_info.type.tensor_type.shape.dim:
                shape = [dim.dim_value if dim.dim_value > 0 else 'dynamic' for dim in input_info.type.tensor_type.shape.dim]
                print(f"  Shape: {shape}")
            print()
        
        # Get output info
        print("Model Outputs:")
        for output_info in model.graph.output:
            print(f"  Name: {output_info.name}")
            print(f"  Type: {output_info.type}")
            if output_info.type.tensor_type.shape.dim:
                shape = [dim.dim_value if dim.dim_value > 0 else 'dynamic' for dim in output_info.type.tensor_type.shape.dim]
                print(f"  Shape: {shape}")
            print()
        
        # Try to find any embedded label information
        print("Looking for embedded class information...")
        for node in model.graph.node:
            if 'class' in node.name.lower() or 'label' in node.name.lower():
                print(f"  Found node: {node.name}")
        
        # Check for any constant values that might be labels
        for initializer in model.graph.initializer:
            if 'class' in initializer.name.lower() or 'label' in initializer.name.lower():
                print(f"  Found initializer: {initializer.name}")
        
        print("="*60)
        
    except Exception as e:
        print(f"Error analyzing model: {e}")

if __name__ == "__main__":
    model_path = "./ImageClassifier.onnx"
    analyze_onnx_model(model_path)
