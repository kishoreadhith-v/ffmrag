# File_name : inat_labels.py

import requests
import json
import numpy as np

def get_inat2021_labels():
    """
    Try to get iNaturalist 2021 labels from various sources
    """
    
    # Try different sources for iNaturalist 2021 labels
    urls = [
        "https://raw.githubusercontent.com/visipedia/inat_comp/master/2021/categories.json",
        "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/categories.json",
        "https://github.com/visipedia/inat_comp/raw/master/2021/categories.json"
    ]
    
    for url in urls:
        try:
            print(f"Trying to fetch labels from: {url}")
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                print(f"✓ Successfully loaded {len(data)} categories from iNaturalist 2021")
                return data
            else:
                print(f"✗ Failed with status code: {response.status_code}")
        except Exception as e:
            print(f"✗ Error: {e}")
            continue
    
    return None

def search_class_by_id(categories, class_id):
    """
    Search for a specific class ID in the categories
    """
    for category in categories:
        if category.get('id') == class_id:
            return category
    return None

def search_classes_by_name(categories, search_term):
    """
    Search for classes containing a specific term
    """
    results = []
    search_term = search_term.lower()
    for category in categories:
        name = category.get('name', '').lower()
        if search_term in name:
            results.append(category)
    return results

def main():
    print("Fetching iNaturalist 2021 labels...")
    categories = get_inat2021_labels()
    
    if categories is None:
        print("Could not fetch iNaturalist 2021 labels. Let me try to create a mapping based on common knowledge.")
        return
    
    # Look for the specific class ID that your model predicted
    target_class = 4682
    print(f"\nLooking for class ID {target_class}...")
    
    category = search_class_by_id(categories, target_class)
    if category:
        print(f"✓ Found class {target_class}: {category['name']}")
        if 'supercategory' in category:
            print(f"  Supercategory: {category['supercategory']}")
    else:
        print(f"✗ Class {target_class} not found in the dataset")
        print(f"Total categories available: {len(categories)}")
        print("First few categories:")
        for i, cat in enumerate(categories[:10]):
            print(f"  ID {cat.get('id', 'N/A')}: {cat.get('name', 'N/A')}")
    
    # Also search for lion-related categories
    print(f"\nSearching for lion-related categories...")
    lion_categories = search_classes_by_name(categories, 'lion')
    if lion_categories:
        print(f"Found {len(lion_categories)} lion-related categories:")
        for cat in lion_categories:
            print(f"  ID {cat.get('id', 'N/A')}: {cat.get('name', 'N/A')}")
    else:
        print("No lion-related categories found")
    
    # Search for other big cats
    for animal in ['tiger', 'leopard', 'cheetah', 'puma', 'jaguar']:
        results = search_classes_by_name(categories, animal)
        if results:
            print(f"\n{animal.title()} categories:")
            for cat in results[:5]:  # Show first 5 matches
                print(f"  ID {cat.get('id', 'N/A')}: {cat.get('name', 'N/A')}")

if __name__ == "__main__":
    main()
