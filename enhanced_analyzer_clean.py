# File_name : enhanced_analyzer_fixed.py

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
import onnxruntime as ort
import os
import time
import json

def load_species_mapping():
    """Load species mapping from official JSON file"""
    mapping_file = "official_species_mapping.json"
    
    if not os.path.exists(mapping_file):
        print("❌ Official species mapping file not found!")
        print("Please run 'load_species_csv.py' with your official CSV file first.")
        return {}
    
    try:
        with open(mapping_file, "r") as f:
            mapping = json.load(f)
            print(f"✅ Loaded official iNaturalist 2021 mapping with {len(mapping)} species")
            return {int(k): v for k, v in mapping.items()}
    except Exception as e:
        print(f"❌ Error loading official species mapping: {e}")
        return {}

# Load the species mapping at startup
SPECIES_NAMES = load_species_mapping()
print(f"✅ Loaded {len(SPECIES_NAMES)} species names")

def get_taxonomic_info(class_id):
    """Get taxonomic information and confidence interpretation"""
    species_name = SPECIES_NAMES.get(class_id, f"Species #{class_id} (Scientific name not loaded)")
    
    # Determine taxonomic group based on species name or class ID patterns
    name_lower = species_name.lower()
    
    if any(term in name_lower for term in ['lion', 'tiger', 'leopard', 'cheetah', 'cat', 'felidae']):
        group = "🦁 FELIDAE (Cat Family)"
        accuracy_note = "High accuracy expected for big cats"
    elif any(term in name_lower for term in ['hyena', 'wolf', 'dog', 'carnivore']):
        group = "🐺 CARNIVORA (Carnivores)"
        accuracy_note = "Good accuracy for carnivorous mammals"
    elif any(term in name_lower for term in ['bird', 'eagle', 'aves']):
        group = "🐦 AVES (Birds)"
        accuracy_note = "Bird classification - high diversity"
    elif any(term in name_lower for term in ['plant', 'rose', 'tree', 'flower', 'plantae']):
        group = "🌿 PLANTAE (Plants)"
        accuracy_note = "Plant classification - very high diversity"
    elif any(term in name_lower for term in ['insect', 'beetle', 'butterfly', 'ant', 'bee']):
        group = "🐛 INSECTA (Insects)"
        accuracy_note = "Insect classification - extremely high diversity"
    elif any(term in name_lower for term in ['fish', 'marine']):
        group = "🐟 AQUATIC (Marine/Freshwater)"
        accuracy_note = "Aquatic species classification"
    elif any(term in name_lower for term in ['mammal', 'primate', 'deer']):
        group = "🦌 MAMMALIA (Mammals)"
        accuracy_note = "Mammalian species classification"
    elif any(term in name_lower for term in ['reptile', 'snake', 'lizard']):
        group = "🦎 REPTILIA (Reptiles)"
        accuracy_note = "Reptile classification"
    elif any(term in name_lower for term in ['amphibian']):
        group = "🐸 AMPHIBIA (Amphibians)"
        accuracy_note = "Amphibian classification"
    else:
        group = "❓ UNKNOWN TAXONOMY"
        accuracy_note = "Taxonomic group not determined - may need species database update"
    
    return species_name, group, accuracy_note

def interpret_confidence(confidence, total_classes=10000):
    """Interpret confidence score in context"""
    if confidence > 50:
        return "🟢 VERY HIGH - Excellent match"
    elif confidence > 20:
        return "🟡 HIGH - Strong match"
    elif confidence > 10:
        return "🟠 MODERATE - Good match"
    elif confidence > 5:
        return "🔵 LOW - Possible match"
    else:
        return "⚪ VERY LOW - Uncertain"

class EnhancedAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("🔬 Enhanced Species Analyzer - iNaturalist 2021")
        self.root.geometry("1000x800")
        self.root.configure(bg='#1e1e1e')
        
        # Load model
        self.session = None
        self.load_model()
        
        # Create GUI
        self.create_widgets()
        
    def load_model(self):
        """Load ONNX model"""
        try:
            print("Loading iNaturalist 2021 model...")
            self.session = ort.InferenceSession("./ImageClassifier.onnx", providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            print("✓ Model loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Model loading failed: {e}")
            
    def create_widgets(self):
        """Create enhanced GUI"""
        # Main container
        main_frame = tk.Frame(self.root, bg='#1e1e1e')
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Title with database status
        title = tk.Label(main_frame, text="🔬 Enhanced Species Analyzer", 
                        font=("Arial", 24, "bold"), bg='#1e1e1e', fg='#00ff88')
        title.pack(pady=5)
        
        # Database status
        db_status = tk.Label(main_frame, text=f"📊 Database: {len(SPECIES_NAMES)} species loaded", 
                            font=("Arial", 12), bg='#1e1e1e', fg='#888888')
        db_status.pack(pady=2)
        
        # Upload section
        upload_frame = tk.Frame(main_frame, bg='#2d2d2d', relief='raised', bd=2)
        upload_frame.pack(fill='x', pady=10)
        
        tk.Label(upload_frame, text="📁 Image Upload", font=("Arial", 16, "bold"), 
                bg='#2d2d2d', fg='#ffffff').pack(pady=5)
        
        btn_frame = tk.Frame(upload_frame, bg='#2d2d2d')
        btn_frame.pack(pady=10)
        
        tk.Button(btn_frame, text="🔍 Browse & Analyze", command=self.browse_and_analyze,
                 font=("Arial", 12, "bold"), bg='#0066cc', fg='white', 
                 padx=20, pady=8).pack(side='left', padx=10)
        
        tk.Button(btn_frame, text="📊 Detailed Analysis", command=self.detailed_analysis,
                 font=("Arial", 12, "bold"), bg='#cc6600', fg='white',
                 padx=20, pady=8).pack(side='left', padx=10)
        
        tk.Button(btn_frame, text="🔄 Update Species DB", command=self.update_species_db,
                 font=("Arial", 12, "bold"), bg='#9933cc', fg='white',
                 padx=20, pady=8).pack(side='left', padx=10)
        
        # Content area
        content_frame = tk.Frame(main_frame, bg='#1e1e1e')
        content_frame.pack(fill='both', expand=True, pady=10)
        
        # Image display
        image_frame = tk.LabelFrame(content_frame, text="🖼️ Selected Image", 
                                   font=("Arial", 14, "bold"), bg='#2d2d2d', fg='#ffffff')
        image_frame.pack(side='left', fill='y', padx=(0, 10))
        
        self.image_label = tk.Label(image_frame, text="No image selected", 
                                   bg='#2d2d2d', fg='#888888', 
                                   width=30, height=20)
        self.image_label.pack(padx=10, pady=10)
        
        # Results display
        results_frame = tk.LabelFrame(content_frame, text="📊 Analysis Results", 
                                     font=("Arial", 14, "bold"), bg='#2d2d2d', fg='#ffffff')
        results_frame.pack(side='right', fill='both', expand=True)
        
        # Results text with scrollbar
        text_frame = tk.Frame(results_frame, bg='#2d2d2d')
        text_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.results_text = tk.Text(text_frame, bg='#1e1e1e', fg='#ffffff', 
                                   font=("Consolas", 11), wrap='word', state='disabled')
        scrollbar = tk.Scrollbar(text_frame, orient='vertical', command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        self.results_text.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Initial message
        self.update_results("Welcome to Enhanced Species Analyzer!\n")
        self.update_results("🔬 Ready to analyze iNaturalist 2021 species\n")
        self.update_results(f"📊 Database loaded: {len(SPECIES_NAMES)} species\n\n")
        self.update_results("Click 'Browse & Analyze' to start!\n")
        
    def browse_and_analyze(self):
        """Browse for image and analyze"""
        file_path = filedialog.askopenfilename(
            title="Select an image for species analysis", 
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")]
        )
        if file_path:
            self.analyze_image(file_path)
            
    def detailed_analysis(self):
        """Detailed analysis with taxonomic info"""
        file_path = filedialog.askopenfilename(
            title="Select image for detailed species analysis", 
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")]
        )
        if file_path:
            self.analyze_image(file_path, detailed=True)
            
    def update_species_db(self):
        """Try to update the species database"""
        try:
            self.update_results("🔄 Updating species database...\n")
            self.root.update()
            
            # Try to create comprehensive mapping using iNaturalist data
            try:
                import subprocess
                result = subprocess.run(["python", "find_inat_labels.py"], 
                                       capture_output=True, text=True, cwd=".")
                
                if result.returncode == 0:
                    self.update_results("✅ iNaturalist 2021 comprehensive database created!\n")
                else:
                    self.update_results("⚠️  Using existing species database\n")
                    
            except Exception as e:
                self.update_results(f"⚠️  iNaturalist update failed: {e}\n")
                
            # Reload the species mapping
            global SPECIES_NAMES
            SPECIES_NAMES = load_species_mapping()
            
            self.update_results(f"📊 Database updated: {len(SPECIES_NAMES)} species loaded\n\n")
            self.update_results("✅ Species database now loaded with comprehensive iNaturalist 2021 mapping!\n")
                
        except Exception as e:
            self.update_results(f"❌ Species database update failed: {e}\n")
            self.update_results(f"Current database size: {len(SPECIES_NAMES)} species\n")
    
    def analyze_image(self, image_path, detailed=False):
        """Analyze image with optional detailed output"""
        try:
            # Show progress
            self.update_results("🔄 Processing image...\n")
            self.root.update()
            
            # Display image
            self.display_image(image_path)
            
            # Preprocess and predict
            start_time = time.time()
            preprocessed = self.preprocess_image(image_path)
            prediction = self.session.run([self.output_name], {self.input_name: preprocessed})
            end_time = time.time()
            
            # Analyze results
            self.display_analysis(image_path, prediction[0], end_time - start_time, detailed)
            
        except Exception as e:
            self.update_results(f"❌ Analysis failed: {e}\n")
            
    def preprocess_image(self, image_path):
        """Preprocess image for model"""
        image = Image.open(image_path).convert('RGB').resize((336, 336))
        img_array = np.array(image, dtype=np.float32) / 255.0
        
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_array = (img_array - mean) / std
        
        return np.expand_dims(np.transpose(img_array, (2, 0, 1)), 0).astype(np.float32)
        
    def display_image(self, image_path):
        """Display selected image"""
        try:
            image = Image.open(image_path)
            image.thumbnail((300, 300), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            self.image_label.configure(image=photo, text="")
            self.image_label.image = photo
        except Exception as e:
            self.image_label.configure(text=f"Image load error: {e}", image="")
            
    def display_analysis(self, image_path, prediction, processing_time, detailed):
        """Display comprehensive analysis results"""
        filename = os.path.basename(image_path)
        pred_array = np.array(prediction[0])
        top_indices = pred_array.argsort()[-10:][::-1]
        top_probs = pred_array[top_indices]
        
        # Build results
        results = f"📊 ANALYSIS: {filename}\n"
        results += f"⏱️  Time: {processing_time*1000:.1f}ms\n"
        results += "=" * 50 + "\n\n"
        
        if detailed:
            results += "🔬 DETAILED TAXONOMIC ANALYSIS\n\n"
        else:
            results += "🏆 TOP PREDICTIONS\n\n"
            
        for i, (idx, prob) in enumerate(zip(top_indices, top_probs), 1):
            species_name, taxonomic_group, accuracy_note = get_taxonomic_info(idx)
            confidence = prob * 100 if prob < 1 else prob
            confidence_level = interpret_confidence(confidence)
            
            if i <= 3:
                medal = ["🥇", "🥈", "🥉"][i-1]
            else:
                medal = f"{i:2d}."
                
            results += f"{medal} {species_name}\n"
            results += f"    Confidence: {confidence:.2f}% {confidence_level}\n"
            
            if detailed:
                results += f"    Taxonomy: {taxonomic_group}\n"
                results += f"    Note: {accuracy_note}\n"
                
            results += f"    Class ID: {idx}\n\n"
            
        if detailed:
            results += "\n💡 INTERPRETATION:\n"
            top_confidence = top_probs[0] * 100 if top_probs[0] < 1 else top_probs[0]
            if top_confidence > 15:
                results += "✅ High confidence identification\n"
            elif top_confidence > 8:
                results += "⚠️  Moderate confidence - likely correct\n"
            else:
                results += "❓ Low confidence - multiple possibilities\n"
                
            results += f"📈 With 10,000 possible species, {top_confidence:.1f}% is significant\n"
            
        self.update_results(results)
        
    def update_results(self, text):
        """Update results display"""
        self.results_text.config(state='normal')
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, text)
        self.results_text.config(state='disabled')
        self.results_text.see(1.0)

def main():
    root = tk.Tk()
    app = EnhancedAnalyzer(root)
    print("🚀 Enhanced Species Analyzer started!")
    root.mainloop()

if __name__ == "__main__":
    main()
