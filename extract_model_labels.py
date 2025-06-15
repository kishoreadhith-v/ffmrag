# File_name : extract_model_labels.py

import timm
import torch
import json
import requests
import os

def check_timm_model_labels():
    """Check if timm model has built-in labels"""
    try:
        print("🔍 Checking timm model for built-in labels...")
        
        # Load the model info
        model_name = 'vit_large_patch14_clip_336'
        model = timm.create_model(model_name, pretrained=True, num_classes=10000)
        
        print(f"Model: {model_name}")
        print(f"Number of classes: {model.num_classes}")
        
        # Check model configuration
        if hasattr(model, 'default_cfg'):
            cfg = model.default_cfg
            print("\n📋 Model Configuration:")
            for key, value in cfg.items():
                print(f"  {key}: {value}")
                
                # Look for label files
                if 'label' in key.lower() or 'class' in key.lower():
                    print(f"  🎯 Found label-related config: {key} = {value}")
        
        # Check if there's a pretrained config
        if hasattr(model, 'pretrained_cfg'):
            pcfg = model.pretrained_cfg
            print("\n🏷️  Pretrained Configuration:")
            for key, value in pcfg.items():
                print(f"  {key}: {value}")
                
                if 'label' in key.lower() or 'class' in key.lower():
                    print(f"  🎯 Found label-related config: {key} = {value}")
        
        return model
        
    except Exception as e:
        print(f"❌ Error checking timm model: {e}")
        return None

def try_download_inat_labels():
    """Try to download iNaturalist 2021 official labels"""
    
    urls = [
        # Official iNaturalist 2021 competition labels
        "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/categories.json",
        "https://raw.githubusercontent.com/visipedia/inat_comp/master/2021/categories.json",
        # Hugging Face datasets
        "https://huggingface.co/datasets/inat2021/raw/main/categories.json",
        # Alternative academic sources
        "https://github.com/richardaecn/class-balanced-loss/raw/master/2021/iNaturalist2021_categories.json"
    ]
    
    for url in urls:
        try:
            print(f"\n🌐 Trying: {url}")
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if isinstance(data, list) and len(data) > 8000:  # iNaturalist 2021 has ~10k classes
                    print(f"✅ SUCCESS! Downloaded {len(data)} species")
                    
                    # Save the data
                    with open("inat2021_official_categories.json", "w") as f:
                        json.dump(data, f, indent=2)
                    
                    # Process into our format
                    species_mapping = {}
                    for item in data:
                        if isinstance(item, dict):
                            class_id = item.get('id')
                            name = item.get('name', '')
                            supercategory = item.get('supercategory', '')
                            
                            if class_id is not None:
                                # Create a nice species name
                                if supercategory and supercategory != name:
                                    species_name = f"{name} ({supercategory})"
                                else:
                                    species_name = name
                                    
                                species_mapping[int(class_id)] = species_name
                    
                    # Save our processed mapping
                    with open("inat2021_species_mapping.json", "w") as f:
                        json.dump(species_mapping, f, indent=2)
                    
                    print(f"📊 Processed {len(species_mapping)} species into mapping")
                    return species_mapping
                    
                else:
                    print(f"❌ Invalid data format or too few classes: {len(data) if isinstance(data, list) else 'not a list'}")
                    
            else:
                print(f"❌ HTTP {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            continue
    
    return None

def check_existing_model_files():
    """Check if there are any existing model files with labels"""
    
    # Check common locations for model metadata
    check_paths = [
        "./vit_large_patch14_clip_336.pth",
        "./ImageClassifier.onnx",
        "./categories.json",
        "./labels.txt",
        "./class_names.json"
    ]
    
    print("\n📁 Checking for existing model files...")
    
    for path in check_paths:
        if os.path.exists(path):
            print(f"✅ Found: {path}")
            
            # Try to extract any label information
            if path.endswith('.json'):
                try:
                    with open(path, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, dict) and len(data) > 100:
                            print(f"  📋 Contains {len(data)} entries - might be species mapping!")
                            return data
                        elif isinstance(data, list) and len(data) > 100:
                            print(f"  📋 Contains {len(data)} entries - might be species list!")
                            return {i: item for i, item in enumerate(data)}
                except Exception as e:
                    print(f"  ❌ Error reading {path}: {e}")
                    
            elif path.endswith('.txt'):
                try:
                    with open(path, 'r') as f:
                        lines = f.readlines()
                        if len(lines) > 100:
                            print(f"  📋 Contains {len(lines)} lines - might be species list!")
                            return {i: line.strip() for i, line in enumerate(lines)}
                except Exception as e:
                    print(f"  ❌ Error reading {path}: {e}")
        else:
            print(f"❌ Not found: {path}")
    
    return None

def main():
    print("🔬 Model Label Extractor for iNaturalist 2021")
    print("=" * 60)
    
    # Step 1: Check timm model configuration
    model = check_timm_model_labels()
    
    # Step 2: Check existing files
    existing_labels = check_existing_model_files()
    if existing_labels:
        print(f"\n🎉 Found existing labels with {len(existing_labels)} entries!")
        # Save as our species mapping
        with open("species_mapping_from_model.json", "w") as f:
            json.dump(existing_labels, f, indent=2)
        return existing_labels
    
    # Step 3: Try to download official labels
    downloaded_labels = try_download_inat_labels()
    if downloaded_labels:
        return downloaded_labels
    
    # Step 4: Check if we can extract from the model weights
    print("\n🔍 Checking model weights for embedded labels...")
    
    if model:
        # Sometimes labels are embedded in the model checkpoint
        try:
            # Load the original .pth file if available
            if os.path.exists("vit_large_patch14_clip_336.pth"):
                print("📦 Loading original model checkpoint...")
                checkpoint = torch.load("vit_large_patch14_clip_336.pth", map_location='cpu')
                
                print("🔍 Checkpoint keys:", list(checkpoint.keys()))
                
                # Look for label-related keys
                for key in checkpoint.keys():
                    if 'label' in key.lower() or 'class' in key.lower() or 'categories' in key.lower():
                        print(f"🎯 Found potential label key: {key}")
                        print(f"   Type: {type(checkpoint[key])}")
                        if hasattr(checkpoint[key], 'shape'):
                            print(f"   Shape: {checkpoint[key].shape}")
                        
        except Exception as e:
            print(f"❌ Error checking model checkpoint: {e}")
    
    print("\n❌ Could not extract built-in labels from model")
    print("💡 The model package might not include the species mapping")
    print("💡 Using our curated mapping instead")
    
    return None

if __name__ == "__main__":
    result = main()
    
    if result:
        print(f"\n✅ SUCCESS: Extracted {len(result)} species labels!")
        print("\n🔍 Sample species:")
        for i, (class_id, name) in enumerate(list(result.items())[:10]):
            print(f"  {class_id}: {name}")
        
        if len(result) > 10:
            print(f"  ... and {len(result) - 10} more species")
    else:
        print("\n⚠️  No built-in labels found in model package")
        print("💡 Will use curated species mapping")
