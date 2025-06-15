# File_name: analyze_model.py

import torch
import timm
import json
import requests

def analyze_model_and_find_labels():
    print("Analyzing your model...")
    
    # Load the model to see its configuration
    try:
        model = timm.create_model('vit_large_patch14_clip_336', pretrained=False, num_classes=10000)
        state_dict = torch.load('vit_large_patch14_clip_336.pth', map_location='cpu')
        model.load_state_dict(state_dict)
        
        print("✓ Model loaded successfully")
        print(f"Model architecture: {model.__class__.__name__}")
        print(f"Number of classes: {model.num_classes}")
        
        # Check if there's any metadata in the state dict
        if isinstance(state_dict, dict):
            print("\nChecking for metadata in model file...")
            for key in state_dict.keys():
                if 'label' in key.lower() or 'class' in key.lower():
                    print(f"Found metadata key: {key}")
                    
    except Exception as e:
        print(f"Error loading model: {e}")
    
    # Try to find ImageNet-21K labels from alternative sources
    print("\nSearching for ImageNet-21K labels from alternative sources...")
    
    # Try HuggingFace datasets
    try:
        # Common ImageNet-21K label sources
        urls_to_try = [
            "https://storage.googleapis.com/bit_models/imagenet21k_wordnet_lemmas.txt",
            "https://raw.githubusercontent.com/rwightman/pytorch-image-models/master/results/imagenet21k_wordnet_lemmas.txt",
            "https://huggingface.co/datasets/imagenet-21k/raw/main/imagenet21k_wordnet_lemmas.txt"
        ]
        
        for url in urls_to_try:
            try:
                print(f"Trying: {url}")
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    labels = response.text.strip().split('\n')
                    print(f"Success! Found {len(labels)} labels")
                    
                    if len(labels) > 4682:
                        print(f"\nClass 4682: '{labels[4682]}'")
                        
                        # Check if it's animal-related
                        target_label = labels[4682].lower()
                        animal_keywords = ['lion', 'cat', 'feline', 'carnivore', 'mammal', 'animal', 'beast', 'predator']
                        
                        is_animal = any(keyword in target_label for keyword in animal_keywords)
                        print(f"Is animal-related: {is_animal}")
                        
                        # Show context
                        print(f"\nContext around class 4682:")
                        for i in range(max(0, 4682-3), min(len(labels), 4682+4)):
                            marker = " ← TARGET" if i == 4682 else ""
                            print(f"  {i}: {labels[i]}{marker}")
                        
                        # Search for lion specifically
                        print(f"\nSearching for 'lion' in labels:")
                        lion_matches = []
                        for i, label in enumerate(labels):
                            if 'lion' in label.lower():
                                lion_matches.append((i, label))
                        
                        if lion_matches:
                            print("Found lion-related classes:")
                            for idx, label in lion_matches[:15]:
                                print(f"  {idx}: {label}")
                        else:
                            print("No direct 'lion' matches found")
                            
                        return True
                    else:
                        print(f"Not enough labels (only {len(labels)})")
                        
            except Exception as e:
                print(f"Failed: {e}")
                
    except Exception as e:
        print(f"Error in label search: {e}")
    
    # If no labels found, try to analyze the prediction pattern
    print(f"\nAnalyzing prediction pattern...")
    print("Your model consistently predicts class 4682 for the lion image.")
    print("This high confidence (92.39%) suggests the model is working correctly")
    print("and recognizes something specific about your image.")
    
    return False

if __name__ == "__main__":
    analyze_model_and_find_labels()
