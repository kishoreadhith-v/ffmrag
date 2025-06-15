# File_name : upload_analyzer.py

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
print(f"🔬 Upload analyzer loaded with {len(SPECIES_NAMES)} species")

class SpeciesAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("iNaturalist 2021 - Species Upload Analyzer")
        self.root.geometry("800x700")
        self.root.configure(bg='#2e3440')
        
        # Initialize model
        self.session = None
        self.input_name = None
        self.output_name = None
        self.load_model()
        
        # Create GUI
        self.create_widgets()
        
    def load_model(self):
        """Load the ONNX model"""
        try:
            print("Loading iNaturalist 2021 model...")
            self.session = ort.InferenceSession("./ImageClassifier.onnx", providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            print("✓ Model loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")
            
    def create_widgets(self):
        """Create the GUI widgets"""
        # Title
        title_label = tk.Label(self.root, text="🔬 iNaturalist Species Analyzer", 
                              font=("Arial", 20, "bold"), 
                              bg='#2e3440', fg='#88c0d0')
        title_label.pack(pady=20)
        
        # Upload button
        upload_frame = tk.Frame(self.root, bg='#2e3440')
        upload_frame.pack(pady=10)
        
        upload_btn = tk.Button(upload_frame, text="📁 Upload Image", 
                              command=self.upload_image,
                              font=("Arial", 14, "bold"),
                              bg='#5e81ac', fg='white',
                              padx=30, pady=10,
                              cursor="hand2")
        upload_btn.pack(side=tk.LEFT, padx=10)
        
        # Browse button
        browse_btn = tk.Button(upload_frame, text="🔍 Browse Files", 
                              command=self.browse_files,
                              font=("Arial", 14, "bold"),
                              bg='#81a1c1', fg='white',
                              padx=30, pady=10,
                              cursor="hand2")
        browse_btn.pack(side=tk.LEFT, padx=10)
        
        # Image display area
        self.image_frame = tk.Frame(self.root, bg='#3b4252', relief='sunken', bd=2)
        self.image_frame.pack(pady=20, padx=20, fill='both', expand=True)
        
        # Image label
        self.image_label = tk.Label(self.image_frame, 
                                   text="No image selected\n\nClick 'Upload Image' or 'Browse Files' to start",
                                   font=("Arial", 14),
                                   bg='#3b4252', fg='#d8dee9',
                                   justify='center')
        self.image_label.pack(expand=True)
        
        # Results area
        self.results_frame = tk.Frame(self.root, bg='#2e3440')
        self.results_frame.pack(pady=10, padx=20, fill='x')
        
        # Results title
        self.results_title = tk.Label(self.results_frame, text="🎯 Analysis Results", 
                                     font=("Arial", 16, "bold"),
                                     bg='#2e3440', fg='#a3be8c')
        self.results_title.pack()
        
        # Results text
        self.results_text = tk.Text(self.results_frame, height=8, width=80,
                                   font=("Consolas", 11),
                                   bg='#3b4252', fg='#d8dee9',
                                   wrap='word', state='disabled')
        self.results_text.pack(pady=10)
        
        # Scrollbar for results
        scrollbar = ttk.Scrollbar(self.results_frame, orient="vertical", command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
    def upload_image(self):
        """Handle drag and drop or upload"""
        # For now, this will open file dialog (drag-drop is complex in tkinter)
        self.browse_files()
        
    def browse_files(self):
        """Open file browser to select image"""
        file_types = [
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.gif"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("PNG files", "*.png"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Select an image to analyze",
            filetypes=file_types,
            initialdir=os.getcwd()
        )
        
        if filename:
            self.analyze_image(filename)
            
    def preprocess_image(self, image_path):
        """Preprocess image for model input"""
        try:
            # Load and convert to RGB
            image = Image.open(image_path).convert('RGB')
            
            # Resize to model input size
            image = image.resize((336, 336))
            
            # Convert to numpy array and normalize
            img_array = np.array(image, dtype=np.float32) / 255.0
            
            # ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_array = (img_array - mean) / std
            
            # Convert to CHW format and add batch dimension
            img_array = np.transpose(img_array, (2, 0, 1)).astype(np.float32)
            img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
            
            return img_array
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to preprocess image: {e}")
            return None
            
    def run_inference(self, preprocessed_image):
        """Run model inference"""
        try:
            if self.session is None:
                return None
                
            # Run inference
            prediction = self.session.run([self.output_name], {self.input_name: preprocessed_image})
            return prediction[0]
            
        except Exception as e:
            messagebox.showerror("Error", f"Inference failed: {e}")
            return None
            
    def get_species_name(self, class_id):
        """Get species name for class ID"""
        return SPECIES_NAMES.get(class_id, f"UNKNOWN SPECIES #{class_id}")
        
    def analyze_image(self, image_path):
        """Complete image analysis pipeline"""
        try:
            # Update UI to show loading
            self.results_text.config(state='normal')
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "🔄 Analyzing image...\n")
            self.results_text.config(state='disabled')
            self.root.update()
            
            # Display image
            self.display_image(image_path)
            
            # Preprocess
            start_time = time.time()
            preprocessed = self.preprocess_image(image_path)
            if preprocessed is None:
                return
                
            # Run inference
            prediction = self.run_inference(preprocessed)
            if prediction is None:
                return
                
            end_time = time.time()
            processing_time = (end_time - start_time) * 1000
            
            # Get top predictions
            pred_array = np.array(prediction[0])
            top_10_indices = pred_array.argsort()[-10:][::-1]
            top_10_probs = pred_array[top_10_indices]
            
            # Display results
            self.display_results(image_path, top_10_indices, top_10_probs, processing_time)
            
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {e}")
            
    def display_image(self, image_path):
        """Display the selected image"""
        try:
            # Load and resize image for display
            image = Image.open(image_path)
            
            # Calculate display size (max 400x400)
            max_size = 400
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image)
            
            # Update label
            self.image_label.configure(image=photo, text="")
            self.image_label.image = photo  # Keep a reference
            
        except Exception as e:
            self.image_label.configure(text=f"Error loading image:\n{e}", image="")
            
    def display_results(self, image_path, top_indices, top_probs, processing_time):
        """Display analysis results"""
        # Enable text widget
        self.results_text.config(state='normal')
        self.results_text.delete(1.0, tk.END)
        
        # Add header
        filename = os.path.basename(image_path)
        header = f"📊 Analysis Results for: {filename}\n"
        header += f"⏱️  Processing Time: {processing_time:.1f} ms\n"
        header += "=" * 70 + "\n\n"
        self.results_text.insert(tk.END, header)
        
        # Add top predictions
        self.results_text.insert(tk.END, "🏆 Top 10 Species Predictions:\n\n")
        
        for i, (class_id, prob) in enumerate(zip(top_indices, top_probs), 1):
            species_name = self.get_species_name(class_id)
            confidence = prob * 100 if prob < 1 else prob  # Handle different scales
            
            # Add emoji based on ranking
            if i == 1:
                emoji = "🥇"
            elif i == 2:
                emoji = "🥈"
            elif i == 3:
                emoji = "🥉"
            else:
                emoji = f"{i:2d}."
            
            result_line = f"{emoji} {species_name}\n"
            result_line += f"    Confidence: {confidence:.2f}% | Class ID: {class_id}\n\n"
            
            self.results_text.insert(tk.END, result_line)
        
        # Add footer
        footer = "=" * 70 + "\n"
        footer += "💡 Tip: Try uploading different images to see how the model performs!\n"
        self.results_text.insert(tk.END, footer)
        
        # Disable text widget and scroll to top
        self.results_text.config(state='disabled')
        self.results_text.see(1.0)

def main():
    """Main function to run the GUI application"""
    root = tk.Tk()
    app = SpeciesAnalyzer(root)
    
    print("🚀 iNaturalist Species Analyzer GUI started!")
    print("Upload images to analyze species...")
    
    root.mainloop()

if __name__ == "__main__":
    main()
