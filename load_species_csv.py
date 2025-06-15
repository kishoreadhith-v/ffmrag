# File_name : load_species_csv.py

import pandas as pd
import json
import os

def load_species_from_csv(csv_file_path):
    """
    Load species mapping from official iNaturalist 2021 CSV file
    Expected CSV format: class_id, class_name
    """
    try:
        print(f"🔍 Loading species data from {csv_file_path}...")
        
        # Read CSV file
        df = pd.read_csv(csv_file_path)
        
        # Check column names
        print(f"📊 CSV columns: {list(df.columns)}")
        print(f"📊 Total rows: {len(df)}")
        
        # Try different possible column name combinations
        class_id_col = None
        class_name_col = None
        
        # Look for class_id column
        for col in df.columns:
            if 'class_id' in col.lower() or 'id' in col.lower():
                class_id_col = col
                break
          # Look for class_name column  
        for col in df.columns:
            if 'class_name' in col.lower() or 'name' in col.lower() or 'species' in col.lower() or 'label' in col.lower():
                class_name_col = col
                break
        
        if class_id_col is None or class_name_col is None:
            print(f"❌ Could not find required columns")
            print(f"Available columns: {list(df.columns)}")
            print(f"Expected: class_id and class_name (or similar)")
            return None
        
        print(f"✅ Using columns: {class_id_col} -> {class_name_col}")
        
        # Create species mapping dictionary
        species_mapping = {}
        
        for index, row in df.iterrows():
            class_id = int(row[class_id_col])
            class_name = str(row[class_name_col]).strip()
            
            if pd.notna(class_name) and class_name:
                species_mapping[class_id] = class_name
        
        print(f"✅ Successfully loaded {len(species_mapping)} species")
        
        # Show some examples
        print(f"\n🔍 Sample species:")
        sample_count = 0
        for class_id, name in species_mapping.items():
            print(f"  {class_id}: {name}")
            sample_count += 1
            if sample_count >= 10:
                break
        
        if len(species_mapping) > 10:
            print(f"  ... and {len(species_mapping) - 10} more species")
        
        return species_mapping
        
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return None

def save_species_mapping(species_mapping, output_file="official_species_mapping.json"):
    """Save species mapping to JSON file"""
    try:
        with open(output_file, 'w') as f:
            json.dump(species_mapping, f, indent=2)
        
        print(f"💾 Saved species mapping to {output_file}")
        return True
        
    except Exception as e:
        print(f"❌ Error saving mapping: {e}")
        return False

def load_species_mapping(json_file="official_species_mapping.json"):
    """Load species mapping from JSON file"""
    try:
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                mapping = json.load(f)
                # Convert string keys to integers
                return {int(k): v for k, v in mapping.items()}
        else:
            print(f"❌ File {json_file} not found")
            return None
    except Exception as e:
        print(f"❌ Error loading mapping: {e}")
        return None

def main():
    print("🔬 Official iNaturalist 2021 Species CSV Loader")
    print("=" * 60)
    
    # Look for CSV file
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    
    if not csv_files:
        print("❌ No CSV files found in current directory")
        print("💡 Please place your iNaturalist 2021 CSV file here")
        print("💡 Expected format: class_id, class_name")
        return None
    
    print(f"📁 Found CSV files: {csv_files}")
    
    # Use the first CSV file found, or prompt user
    if len(csv_files) == 1:
        csv_file = csv_files[0]
        print(f"📊 Using: {csv_file}")
    else:
        print("Multiple CSV files found:")
        for i, file in enumerate(csv_files):
            print(f"  {i+1}. {file}")
        
        try:
            choice = int(input("Enter choice (1-{}): ".format(len(csv_files)))) - 1
            csv_file = csv_files[choice]
        except:
            print("Using first file as default")
            csv_file = csv_files[0]
    
    # Load species data
    species_mapping = load_species_from_csv(csv_file)
    
    if species_mapping:
        # Save to JSON for easy loading
        save_species_mapping(species_mapping)
        
        print(f"\n📊 SUMMARY:")
        print(f"Total species loaded: {len(species_mapping)}")
        print(f"Saved to: official_species_mapping.json")
        print(f"\n✅ Ready to use in your analyzers!")
        
        return species_mapping
    else:
        print("❌ Failed to load species data")
        return None

if __name__ == "__main__":
    result = main()
