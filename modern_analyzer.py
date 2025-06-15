# File_name : modern_analyzer.py

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
import onnxruntime as ort
import os
import time
import json
import cv2
import threading
from queue import Queue

def load_species_mapping():
    """Load spec                  # Get results and apply softmax for proper probabilities
            logits = prediction[0]
            # Apply softmax to convert logits to probabilities
            exp_logits = np.exp(logits - np.max(logits))  # Subtract max for numerical stability
            probabilities = exp_logits / np.sum(exp_logits)
            
            top_indices = np.argsort(probabilities)[::-1]   # Get results and apply softmax for proper probabilities
            logits = prediction[0]
            # Apply softmax to convert logits to probabilities
            exp_logits = np.exp(logits - np.max(logits))  # Subtract max for numerical stability
            probabilities = exp_logits / np.sum(exp_logits)
            
            top_indices = np.argsort(probabilities)[::-1] mapping from official JSON file"""
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
    species_name = SPECIES_NAMES.get(class_id, f"Species #{class_id} (Name not found)")
    
    # Determine taxonomic group based on species name
    name_lower = species_name.lower()
    
    if any(term in name_lower for term in ['panthera', 'felis', 'lynx', 'puma', 'acinonyx']):
        group = "🦁 FELIDAE (Cats)"
        accuracy_note = "High accuracy for big cats and felines"
    elif any(term in name_lower for term in ['canis', 'vulpes', 'ursus', 'carnivora']):
        group = "🐺 CARNIVORA (Carnivores)"
        accuracy_note = "Good accuracy for carnivorous mammals"
    elif any(term in name_lower for term in ['corvus', 'trogon', 'aves', 'bird', 'eagle', 'hawk', 'owl']):
        group = "🐦 AVES (Birds)"
        accuracy_note = "Bird classification - high diversity"
    elif any(term in name_lower for term in ['crocodylus', 'reptile', 'snake', 'lizard', 'gecko']):
        group = "🦎 REPTILIA (Reptiles)"
        accuracy_note = "Reptile classification"
    elif any(term in name_lower for term in ['oncorhynchus', 'paralabrax', 'fish', 'shark', 'salmon']):
        group = "🐟 PISCES (Fish)"
        accuracy_note = "Marine life classification"
    elif any(term in name_lower for term in ['smilisca', 'amphibian', 'frog', 'toad']):
        group = "🐸 AMPHIBIA (Amphibians)"
        accuracy_note = "Amphibian classification"
    elif any(term in name_lower for term in ['plantae', 'plant', 'tree', 'flower']):
        group = "🌿 PLANTAE (Plants)"
        accuracy_note = "Plant classification - extremely high diversity"
    elif any(term in name_lower for term in ['insecta', 'butterfly', 'beetle', 'ant']):
        group = "� INSECTA (Insects)"
        accuracy_note = "Insect classification - vast diversity"
    else:
        group = "🔬 SPECIES"
        accuracy_note = "Species identified from iNaturalist 2021 dataset"
    
    return species_name, group, accuracy_note

def interpret_confidence(confidence):
    """Interpret confidence score"""
    if confidence > 80:
        return "🟢 VERY HIGH", "#27ae60"
    elif confidence > 50:
        return "🟡 HIGH", "#f39c12"
    elif confidence > 20:
        return "🟠 MODERATE", "#e67e22"
    elif confidence > 5:
        return "🔵 LOW", "#3498db"
    else:
        return "⚪ VERY LOW", "#95a5a6"

class ModernAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("🔬 iNaturalist 2021 Species Analyzer")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f8f9fa')
        
        # Model variables
        self.session = None
        self.input_name = None
        self.output_name = None
        
        # Camera variables
        self.camera = None
        self.camera_running = False
        self.camera_thread = None
        self.frame_queue = Queue(maxsize=2)
        self.prediction_queue = Queue(maxsize=5)
        self.current_frame = None
        self.last_prediction_time = 0
        self.prediction_interval = 2.0  # Predict every 2 seconds
        
        # Load model
        self.load_model()
        
        # Create modern UI
        self.create_modern_ui()
        
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
            
    def create_modern_ui(self):
        """Create modern, clean UI"""
        # Header
        header = tk.Frame(self.root, bg='#2c3e50', height=80)
        header.pack(fill='x')
        header.pack_propagate(False)
        
        title = tk.Label(header, text="🔬 iNaturalist 2021 Species Analyzer", 
                        font=("Segoe UI", 24, "bold"), bg='#2c3e50', fg='white')
        title.pack(pady=15)
        
        subtitle = tk.Label(header, text=f"AI-powered species identification • {len(SPECIES_NAMES):,} species database", 
                           font=("Segoe UI", 11), bg='#2c3e50', fg='#bdc3c7')
        subtitle.pack()
        
        # Main content
        main_container = tk.Frame(self.root, bg='#f8f9fa')
        main_container.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Left panel
        left_panel = tk.Frame(main_container, bg='white', relief='flat', bd=0)
        left_panel.pack(side='left', fill='both', expand=True, padx=(0, 15))
        
        # Controls section
        controls_frame = tk.Frame(left_panel, bg='white', padx=25, pady=25)
        controls_frame.pack(fill='x')
        
        tk.Label(controls_frame, text="Select Image", font=("Segoe UI", 16, "bold"), 
                bg='white', fg='#2c3e50').pack(anchor='w')
        
        tk.Label(controls_frame, text="Choose an image to identify the species using AI", 
                font=("Segoe UI", 10), bg='white', fg='#7f8c8d').pack(anchor='w', pady=(5, 20))
          # Buttons
        btn_container = tk.Frame(controls_frame, bg='white')
        btn_container.pack(fill='x')
        
        # Row 1: File operations
        file_row = tk.Frame(btn_container, bg='white')
        file_row.pack(fill='x', pady=(0, 10))
        
        self.browse_btn = tk.Button(file_row, text="📷 Browse & Analyze", 
                                   command=self.browse_and_analyze,
                                   font=("Segoe UI", 11, "bold"), bg='#3498db', fg='white',
                                   relief='flat', cursor='hand2', padx=20, pady=10)
        self.browse_btn.pack(side='left', padx=(0, 10))
        
        self.detail_btn = tk.Button(file_row, text="🔍 Detailed Analysis", 
                                   command=self.detailed_analysis,
                                   font=("Segoe UI", 11), bg='#95a5a6', fg='white',
                                   relief='flat', cursor='hand2', padx=20, pady=10)
        self.detail_btn.pack(side='left')
        
        # Row 2: Camera operations
        camera_row = tk.Frame(btn_container, bg='white')
        camera_row.pack(fill='x')
        
        self.camera_btn = tk.Button(camera_row, text="📹 Start Live Camera", 
                                   command=self.toggle_camera,
                                   font=("Segoe UI", 11, "bold"), bg='#e74c3c', fg='white',
                                   relief='flat', cursor='hand2', padx=20, pady=10)
        self.camera_btn.pack(side='left', padx=(0, 10))
        
        self.capture_btn = tk.Button(camera_row, text="📸 Capture & Analyze", 
                                    command=self.capture_and_analyze,
                                    font=("Segoe UI", 11), bg='#f39c12', fg='white',
                                    relief='flat', cursor='hand2', padx=20, pady=10,
                                    state='disabled')
        self.capture_btn.pack(side='left')
        
        # Separator
        separator = tk.Frame(left_panel, bg='#ecf0f1', height=1)
        separator.pack(fill='x', padx=25, pady=15)
        
        # Image preview
        preview_frame = tk.Frame(left_panel, bg='white', padx=25, pady=25)
        preview_frame.pack(fill='both', expand=True)
        
        tk.Label(preview_frame, text="Image Preview", font=("Segoe UI", 16, "bold"), 
                bg='white', fg='#2c3e50').pack(anchor='w')
          # Image display area
        self.image_container = tk.Frame(preview_frame, bg='#ecf0f1', relief='flat', bd=1)
        self.image_container.pack(fill='both', expand=True, pady=(15, 0))
        
        self.image_label = tk.Label(self.image_container, text="📷\\n\\nNo image or camera feed\\n\\nUse 'Browse & Analyze' for files\\nor 'Start Live Camera' for real-time detection", 
                                   font=("Segoe UI", 12), bg='#ecf0f1', fg='#95a5a6')
        self.image_label.pack(expand=True)
        
        # Right panel
        right_panel = tk.Frame(main_container, bg='white', relief='flat', bd=0, width=450)
        right_panel.pack(side='right', fill='both')
        right_panel.pack_propagate(False)
        
        # Results section
        results_frame = tk.Frame(right_panel, bg='white', padx=25, pady=25)
        results_frame.pack(fill='both', expand=True)
        
        tk.Label(results_frame, text="Analysis Results", font=("Segoe UI", 16, "bold"), 
                bg='white', fg='#2c3e50').pack(anchor='w')
        
        # Results text area
        text_container = tk.Frame(results_frame, bg='white')
        text_container.pack(fill='both', expand=True, pady=(15, 0))
        
        self.results_text = tk.Text(text_container, font=("Consolas", 10), 
                                   bg='#f8f9fa', fg='#2c3e50', relief='flat', bd=0,
                                   padx=20, pady=20, wrap='word', state='disabled')
        
        scrollbar = tk.Scrollbar(text_container, orient='vertical', command=self.results_text.yview,
                                bg='#ecf0f1', width=15)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        self.results_text.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Status bar
        status_bar = tk.Frame(self.root, bg='#34495e', height=35)
        status_bar.pack(fill='x', side='bottom')
        status_bar.pack_propagate(False)
        
        self.status_label = tk.Label(status_bar, text="Ready • Model loaded successfully", 
                                    font=("Segoe UI", 10), bg='#34495e', fg='white')
        self.status_label.pack(pady=8)
          # Welcome message
        self.update_results("🎉 Welcome to the iNaturalist 2021 Species Analyzer!\\n\\n")
        self.update_results("✨ This advanced AI can identify species from images and live camera feed.\\n\\n")
        self.update_results(f"📊 Database: {len(SPECIES_NAMES):,} species loaded from official iNaturalist 2021 dataset\\n\\n")
        self.update_results("🚀 OPTIONS:\\n")
        self.update_results("   📷 Browse & Analyze - Upload image files\\n")
        self.update_results("   🔍 Detailed Analysis - Comprehensive species info\\n")
        self.update_results("   📹 Start Live Camera - Real-time species detection\\n")
        self.update_results("   📸 Capture & Analyze - Detailed analysis of live frame\\n\\n")
        self.update_results("🎯 Ready to identify species! Choose an option above.\\n\\n")
        
    def update_results(self, text):
        """Update results text area"""
        self.results_text.config(state='normal')
        self.results_text.insert('end', text)
        self.results_text.config(state='disabled')
        self.results_text.see('end')
        
    def update_status(self, text):
        """Update status bar"""
        self.status_label.config(text=text)
        
    def browse_and_analyze(self):
        """Browse for image and analyze"""
        file_path = filedialog.askopenfilename(
            title="Select Image for Species Analysis",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")]
        )
        if file_path:
            self.analyze_image(file_path)
            
    def detailed_analysis(self):
        """Perform detailed analysis"""
        file_path = filedialog.askopenfilename(
            title="Select Image for Detailed Species Analysis",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")]
        )
        if file_path:
            self.analyze_image(file_path, detailed=True)
            
    def display_image(self, image_path):
        """Display selected image"""
        try:
            # Open and resize image
            img = Image.open(image_path)
            
            # Calculate size to fit in display area
            display_size = (400, 300)
            img.thumbnail(display_size, Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(img)
            
            # Update image label
            self.image_label.configure(image=photo, text="", bg='white')
            self.image_label.image = photo  # Keep a reference
            
        except Exception as e:
            self.image_label.configure(text=f"Error loading image:\\n{str(e)}", 
                                      bg='#ecf0f1', image='')
            
    def analyze_image(self, image_path, detailed=False):
        """Analyze image and show results"""
        try:
            self.update_status("Analyzing image...")
            self.update_results(f"\\n{'='*50}\\n")
            self.update_results(f"🔍 ANALYZING: {os.path.basename(image_path)}\\n")
            self.update_results(f"{'='*50}\\n\\n")
            
            # Display image
            self.display_image(image_path)
            
            # Process image
            start_time = time.time()
            input_data = self.preprocess_image(image_path)
            prediction = self.session.run([self.output_name], {self.input_name: input_data})
            processing_time = time.time() - start_time
            
            # Get results
            probabilities = prediction[0][0]
            top_indices = np.argsort(probabilities)[::-1]
              # Show top predictions
            self.update_results("🏆 TOP SPECIES PREDICTIONS:\\n\\n")
            
            for i, idx in enumerate(top_indices[:5]):
                confidence = probabilities[idx] * 100
                species_name, group, note = get_taxonomic_info(idx)
                conf_label, conf_color = interpret_confidence(confidence)
                
                # Format species name nicely
                species_display = species_name
                if len(species_display) > 50:
                    species_display = species_display[:47] + "..."
                
                self.update_results(f"{i+1}. {species_display}\\n")
                self.update_results(f"   📊 Confidence: {confidence:.2f}% {conf_label}\\n")
                self.update_results(f"   🏷️  Group: {group}\\n")
                
                if detailed:
                    self.update_results(f"   🆔 Class ID: {idx}\\n")
                    self.update_results(f"   📝 {note}\\n")
                
                self.update_results("\\n")
            
            # Processing info
            self.update_results(f"⚡ Processing time: {processing_time:.3f} seconds\\n")
            self.update_results(f"🧠 Model: iNaturalist 2021 (10,000 classes)\\n\\n")
            
            if detailed:
                self.update_results("📊 DETAILED STATISTICS:\\n\\n")
                self.update_results(f"   Total classes: {len(probabilities):,}\\n")
                self.update_results(f"   Highest confidence: {np.max(probabilities)*100:.2f}%\\n")
                self.update_results(f"   Top 5 average: {np.mean(probabilities[top_indices[:5]])*100:.2f}%\\n")
                self.update_results(f"   Analysis time: {time.strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")
            
            self.update_status(f"Analysis complete • Top match: {confidence:.1f}% confidence")
            
        except Exception as e:
            self.update_results(f"\\n❌ Analysis failed: {str(e)}\\n\\n")
            self.update_status("Analysis failed")
            
    def preprocess_image(self, image_path):
        """Preprocess image for model"""
        # Load and resize image
        img = Image.open(image_path).convert('RGB')
        img = img.resize((336, 336), Image.Resampling.BILINEAR)
        
        # Convert to numpy array and normalize
        img_array = np.array(img).astype(np.float32) / 255.0
        
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        img_array = (img_array - mean) / std
        
        # Transpose to CHW format and add batch dimension
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array

    def toggle_camera(self):
        """Toggle camera on/off"""
        if not self.camera_running:
            self.start_camera()
        else:
            self.stop_camera()
            
    def start_camera(self):
        """Start camera feed"""
        try:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                messagebox.showerror("Error", "Could not access camera")
                return
                
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            self.camera_running = True
            self.camera_btn.config(text="⏹️ Stop Camera", bg='#27ae60')
            self.capture_btn.config(state='normal')
            
            # Start camera thread
            self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
            self.camera_thread.start()
            
            # Start prediction thread
            self.prediction_thread = threading.Thread(target=self.prediction_loop, daemon=True)
            self.prediction_thread.start()
            
            self.update_status("Camera started • Live species detection active")
            self.update_results("\\n📹 LIVE CAMERA STARTED\\n")
            self.update_results("🔍 Real-time species detection every 2 seconds\\n")
            self.update_results("📸 Click 'Capture & Analyze' for detailed analysis\\n\\n")
            
        except Exception as e:
            messagebox.showerror("Camera Error", f"Failed to start camera: {e}")
            
    def stop_camera(self):
        """Stop camera feed"""
        self.camera_running = False
        
        if self.camera:
            self.camera.release()
            self.camera = None
            
        self.camera_btn.config(text="📹 Start Live Camera", bg='#e74c3c')
        self.capture_btn.config(state='disabled')
        
        # Reset image display
        self.image_label.configure(text="📷\\n\\nCamera stopped\\n\\nClick 'Start Live Camera' to resume", 
                                  bg='#ecf0f1', image='')
        
        self.update_status("Camera stopped")
        self.update_results("\\n📹 Camera feed stopped\\n\\n")
        
    def camera_loop(self):
        """Main camera loop"""
        while self.camera_running and self.camera:
            try:
                ret, frame = self.camera.read()
                if not ret:
                    break
                    
                # Store current frame
                self.current_frame = frame.copy()
                
                # Convert frame for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                
                # Resize for display
                display_size = (400, 300)
                frame_pil.thumbnail(display_size, Image.Resampling.LANCZOS)
                
                # Convert to PhotoImage
                photo = ImageTk.PhotoImage(frame_pil)
                
                # Update display in main thread
                self.root.after(0, self.update_camera_display, photo)
                
                # Add frame to queue for prediction (if not full)
                current_time = time.time()
                if current_time - self.last_prediction_time >= self.prediction_interval:
                    if not self.frame_queue.full():
                        self.frame_queue.put(frame.copy())
                        self.last_prediction_time = current_time
                
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                print(f"Camera loop error: {e}")
                break
                
    def update_camera_display(self, photo):
        """Update camera display (called from main thread)"""
        if self.camera_running:
            self.image_label.configure(image=photo, text="", bg='white')
            self.image_label.image = photo
            
    def prediction_loop(self):
        """Background prediction loop"""
        while self.camera_running:
            try:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get()
                    
                    # Make prediction
                    prediction = self.predict_frame(frame)
                    if prediction:
                        # Update results in main thread
                        self.root.after(0, self.update_live_results, prediction)
                        
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Prediction loop error: {e}")
                time.sleep(1)
                
    def predict_frame(self, frame):
        """Make prediction on camera frame"""
        try:
            # Preprocess frame
            input_data = self.preprocess_frame(frame)
            prediction = self.session.run([self.output_name], {self.input_name: input_data})
            
            # Get predictions and apply softmax to convert logits to probabilities
            logits = prediction[0][0]
            # Apply softmax to get proper probabilities
            exp_logits = np.exp(logits - np.max(logits))  # Subtract max for numerical stability
            probabilities = exp_logits / np.sum(exp_logits)
            
            top_idx = np.argmax(probabilities)
            confidence = probabilities[top_idx] * 100
            
            if confidence > 1:  # Only show if confidence > 1%
                species_name, group, note = get_taxonomic_info(top_idx)
                return {
                    'species': species_name,
                    'confidence': confidence,
                    'group': group,
                    'class_id': top_idx
                }
            return None
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return None
            
    def update_live_results(self, prediction):
        """Update live prediction results"""
        timestamp = time.strftime("%H:%M:%S")
        conf_label, conf_color = interpret_confidence(prediction['confidence'])
        
        # Format the species name nicely
        species_display = prediction['species']
        if len(species_display) > 40:
            species_display = species_display[:37] + "..."
        
        self.update_results(f"[{timestamp}] {prediction['group']}\\n")
        self.update_results(f"          🔍 {species_display}\\n")
        self.update_results(f"          📊 {prediction['confidence']:.1f}% {conf_label}\\n\\n")
        
    def capture_and_analyze(self):
        """Capture current frame and do detailed analysis"""
        if self.current_frame is not None:
            self.update_results("\\n📸 CAPTURED FRAME FOR DETAILED ANALYSIS\\n")
            self.update_results("="*50 + "\\n\\n")
            
            # Analyze captured frame
            self.analyze_frame(self.current_frame, detailed=True)
        else:
            messagebox.showwarning("No Frame", "No camera frame available to capture")
            
    def preprocess_frame(self, frame):
        """Preprocess camera frame for model"""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image and resize
        img = Image.fromarray(frame_rgb)
        img = img.resize((336, 336), Image.Resampling.BILINEAR)
        
        # Convert to numpy array and normalize
        img_array = np.array(img).astype(np.float32) / 255.0
        
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        img_array = (img_array - mean) / std
        
        # Transpose to CHW format and add batch dimension
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
        
    def analyze_frame(self, frame, detailed=False):
        """Analyze camera frame with detailed results"""
        try:
            self.update_status("Analyzing captured frame...")
            
            # Process frame
            start_time = time.time()
            input_data = self.preprocess_frame(frame)
            prediction = self.session.run([self.output_name], {self.input_name: input_data})
            processing_time = time.time() - start_time
            
            # Get results
            probabilities = prediction[0][0]
            top_indices = np.argsort(probabilities)[::-1]
              # Show top predictions
            self.update_results("🏆 TOP PREDICTIONS FROM CAMERA:\\n\\n")
            
            for i, idx in enumerate(top_indices[:5]):
                confidence = probabilities[idx] * 100
                species_name, group, note = get_taxonomic_info(idx)
                conf_label, conf_color = interpret_confidence(confidence)
                
                # Format species name nicely
                species_display = species_name
                if len(species_display) > 50:
                    species_display = species_display[:47] + "..."
                
                self.update_results(f"{i+1}. {species_display}\\n")
                self.update_results(f"   📊 Confidence: {confidence:.2f}% {conf_label}\\n")
                self.update_results(f"   🏷️  Group: {group}\\n")
                
                if detailed:
                    self.update_results(f"   🆔 Class ID: {idx}\\n")
                    self.update_results(f"   📝 {note}\\n")
                
                self.update_results("\\n")
            
            # Processing info
            self.update_results(f"⚡ Processing time: {processing_time:.3f} seconds\\n")
            self.update_results(f"📹 Source: Live camera capture\\n\\n")
            
            self.update_status(f"Frame analyzed • Top match: {probabilities[top_indices[0]]*100:.1f}% confidence")
            
        except Exception as e:
            self.update_results(f"\\n❌ Frame analysis failed: {str(e)}\\n\\n")
            self.update_status("Frame analysis failed")
    
    def on_closing(self):
        """Handle window closing"""
        if self.camera_running:
            self.stop_camera()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = ModernAnalyzer(root)
    
    # Handle window closing
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    print("🚀 Modern Species Analyzer with Live Camera started!")
    root.mainloop()

if __name__ == "__main__":
    main()
