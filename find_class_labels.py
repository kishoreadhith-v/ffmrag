# File_name: find_class_labels.py

import requests
import json

def search_class_4682():
    print("Searching for class 4682 in different datasets...")
    
    # Try different label sources
    label_sources = [
        {
            "name": "ImageNet-21K (Winter 2021)",
            "url": "https://raw.githubusercontent.com/google-research/vision_transformer/main/vit_jax/imagenet21k_wordnet_ids.txt"
        },
        {
            "name": "ImageNet-21K Labels", 
            "url": "https://raw.githubusercontent.com/google-research/vision_transformer/main/vit_jax/imagenet21k_wordnet_lemmas.txt"
        }
    ]
    
    for source in label_sources:
        try:
            print(f"\nTrying {source['name']}...")
            response = requests.get(source['url'])
            if response.status_code == 200:
                labels = response.text.strip().split('\n')
                print(f"Found {len(labels)} labels")
                
                if len(labels) > 4682:
                    print(f"Class 4682: {labels[4682]}")
                    
                    # Also check nearby classes for context
                    print("Nearby classes:")
                    for i in range(max(0, 4682-3), min(len(labels), 4682+4)):
                        marker = " <-- TARGET" if i == 4682 else ""
                        print(f"  Class {i}: {labels[i]}{marker}")
                        
                    # Search for lion-related entries
                    print(f"\nSearching for 'lion' in {source['name']}:")
                    lion_classes = []
                    for i, label in enumerate(labels):
                        if 'lion' in label.lower():
                            lion_classes.append((i, label))
                    
                    if lion_classes:
                        print("Found lion-related classes:")
                        for idx, label in lion_classes[:10]:  # Show first 10 matches
                            print(f"  Class {idx}: {label}")
                    else:
                        print("  No lion-related classes found")
                else:
                    print(f"  Not enough labels (only {len(labels)})")
            else:
                print(f"  Failed to fetch: HTTP {response.status_code}")
        except Exception as e:
            print(f"  Error: {e}")
    
    # Try some other common animal datasets
    print(f"\nTrying to find animal classification datasets...")
    
    # Search for common animal class patterns
    animal_keywords = ['lion', 'tiger', 'leopard', 'cat', 'feline', 'big cat', 'carnivore']
    
    print(f"\nNote: Your model predicted class 4682 with high confidence.")
    print("This suggests the model recognizes something specific in your lion image.")
    print("The class might be:")
    print("- A specific lion subspecies (African lion, Asiatic lion)")
    print("- A lion in a specific context (lion cub, lioness, male lion)")
    print("- A related big cat category")

if __name__ == "__main__":
    search_class_4682()
