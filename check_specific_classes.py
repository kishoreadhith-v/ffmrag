# File_name : check_specific_classes.py

import requests

def get_class_names():
    """Get ImageNet-21K class names"""
    try:
        url = "https://storage.googleapis.com/bit_models/imagenet21k_wordnet_lemmas.txt"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            labels = response.text.strip().split('\n')
            return labels
    except:
        return None

def main():
    labels = get_class_names()
    if not labels:
        print("Could not download labels")
        return
    
    # Check the classes that appeared in our debug
    classes_to_check = [4682, 4672, 4678, 4685, 4628, 4155, 3639, 3248, 3658, 3636]
    
    print("Class index to label mapping:")
    print("="*50)
    
    for idx in classes_to_check:
        if idx < len(labels):
            print(f"Index {idx}: {labels[idx]}")
        else:
            print(f"Index {idx}: OUT OF RANGE")
    
    # Let's also search for lion-related classes
    print("\nSearching for lion-related classes:")
    print("="*40)
    
    lion_classes = []
    for i, label in enumerate(labels):
        if 'lion' in label.lower():
            lion_classes.append((i, label))
    
    if lion_classes:
        for idx, label in lion_classes:
            print(f"Index {idx}: {label}")
    else:
        print("No lion-related classes found")

if __name__ == "__main__":
    main()
