from flask import Flask, request, jsonify, Response, stream_with_context
import numpy as np
from PIL import Image
import onnxruntime as ort
import io
import os
import base64
import json
import requests
from modern_analyzer import get_taxonomic_info
from flask_cors import CORS
import time
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

app = Flask(__name__)

# Configure CORS properly with a single configuration
CORS(app, 
     resources={r"/*": {
         "origins": ["http://localhost:3000"],  # Allow only the React dev server
         "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
         "allow_headers": ["Content-Type", "Authorization"],
         "supports_credentials": True
     }}
)

# Initialize chatbot

# Load ONNX model (CPU for simplicity)
MODEL_PATH = 'ImageClassifier.onnx'
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Qualcomm API configuration
QUALCOMM_API_URL = "http://localhost:3001/api/v1/workspace/qualcomm-edge-ai-hackathon/chat"
QUALCOMM_API_KEY = "45B2REQ-B1TMYCN-H49WB34-HXYX5AJ"

# Create a session with retry strategy
def create_retry_session(retries=5, backoff_factor=1.0):
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[500, 502, 503, 504, 408, 429],
        allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "TRACE"]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def get_species_description(species_name):
    """Get detailed description of the species from Qualcomm API with retry logic"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {QUALCOMM_API_KEY}"
    }
    
    request_data = {
        "message": f"""What is {species_name}? Provide a detailed description in the following format:

# {species_name}

## Common Names
[List common names with bullet points]

## Physical Characteristics
[List characteristics with bullet points]

## Habitat
[List habitat information with bullet points]

## Behavior
[List behavioral traits with bullet points]

## Interesting Facts
[List interesting facts with bullet points]

Keep each section concise but informative.""",
        "mode": "chat",
        "sessionId": f"session-{int(time.time())}",
        "attachments": []
    }
    
    try:
        print(f"Sending request for species: {species_name}")
        
        # Create a new session for each request
        session = create_retry_session()
        
        # Make the request with increased timeout
        response = session.post(
            QUALCOMM_API_URL,
            headers=headers,
            json=request_data,
            timeout=120  # Increased timeout to 2 minutes
        )
        
        # Close the session
        session.close()
        
        if response.status_code == 200:
            try:
                data = response.json()
                if isinstance(data, dict):
                    if data.get('error'):
                        print(f"API returned error: {data['error']}")
                        return f"Unable to fetch description: {data['error']}"
                    
                    if 'textResponse' in data and data['textResponse']:
                        # Clean up and format the response
                        description = data['textResponse']
                        
                        # Ensure proper Markdown formatting
                        description = description.replace('**', '#')  # Replace bold with headers
                        description = description.replace(':[', ':')  # Clean up any weird formatting
                        description = description.replace('[', '')
                        description = description.replace(']', '')
                        
                        # Ensure proper spacing between sections
                        sections = ['# Common Names', '# Physical Characteristics', '# Habitat', '# Behavior', '# Interesting Facts']
                        for section in sections:
                            description = description.replace(section, f"\n{section}\n")
                        
                        # Clean up any multiple newlines
                        while '\n\n\n' in description:
                            description = description.replace('\n\n\n', '\n\n')
                        
                        return description.strip()
                    else:
                        print("No text response in API response")
                        return "No description available at the moment. Please try again later."
                        
                print(f"Unexpected response format: {data}")
                return "Description format not recognized in the response."
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                return "Error parsing API response."
        else:
            error_msg = f"API Error: Status {response.status_code}"
            if response.text:
                try:
                    error_data = response.json()
                    if error_data.get('error'):
                        error_msg += f", {error_data['error']}"
                except:
                    error_msg += f", Response: {response.text[:200]}"
            print(error_msg)
            return "Unable to fetch description at the moment. Please try again later."
    except requests.exceptions.Timeout:
        print("Request timed out")
        return "Request timed out. Please try again later."
    except requests.exceptions.ConnectionError as e:
        print(f"Connection error: {str(e)}")
        return "Unable to connect to the description service. Please try again later."
    except Exception as e:
        print(f"Error in get_species_description: {str(e)}")
        return f"Error fetching description: {str(e)}"

def preprocess_image_file(file):
    image = Image.open(file).convert('RGB')
    image = image.resize((336, 336))
    img_array = np.array(image, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std
    img_array = np.transpose(img_array, (2, 0, 1)).astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    return img_array

def preprocess_base64_image(base64_string):
    if 'base64,' in base64_string:
        base64_string = base64_string.split('base64,')[1]
    image_bytes = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize((336, 336))
    img_array = np.array(image, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std
    img_array = np.transpose(img_array, (2, 0, 1)).astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    return img_array

def get_predictions(input_data):
    prediction = session.run([output_name], {input_name: input_data})
    probabilities = prediction[0][0]
    top_indices = np.argsort(probabilities)[::-1][:5]
    results = []
    
    for idx in top_indices:
        confidence = float(probabilities[idx]) * 100  # Convert to percentage
        species_name, group, note = get_taxonomic_info(idx)
        
        results.append({
            'class_id': int(idx),
            'species': species_name,
            'confidence': confidence,
            'group': group,
            'note': note
        })
    
    return results

@app.route('/', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({
        'status': 'OK',
        'message': 'Species Analysis API is running',
        'endpoints': ['/analyze', '/analyze-frame', '/chat']
    })

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    try:
        # Get initial results
        input_data = preprocess_image_file(file)
        results = get_predictions(input_data)
        
        # Send immediate response with results
        if not results:
            return jsonify({'error': 'No species identified'}), 400
            
        # Start description fetch in background
        top_species = results[0]['species']
        print(f"Getting description for: {top_species}")
        
        def generate():
            try:
                # Send initial results immediately
                yield json.dumps({
                    'type': 'results',
                    'data': results
                }) + '\n'
                
                # Get and send description
                description = get_species_description(top_species)
                yield json.dumps({
                    'type': 'description',
                    'data': description
                }) + '\n'
                
            except Exception as e:
                yield json.dumps({
                    'type': 'error',
                    'data': str(e)
                }) + '\n'
        
        return Response(stream_with_context(generate()), mimetype='application/x-ndjson')
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze-frame', methods=['POST'])
def analyze_frame():
    try:
        data = request.get_json()
        if not data or 'frame' not in data:
            return jsonify({'error': 'No frame data provided'}), 400
        
        # Get initial results
        input_data = preprocess_base64_image(data['frame'])
        results = get_predictions(input_data)
        
        # Send immediate response with results
        if not results:
            return jsonify({'error': 'No species identified'}), 400
            
        # Start description fetch in background
        top_species = results[0]['species']
        print(f"Getting description for: {top_species}")
        
        def generate():
            try:
                # Send initial results immediately
                yield json.dumps({
                    'type': 'results',
                    'data': results
                }) + '\n'
                
                # Get and send description
                description = get_species_description(top_species)
                yield json.dumps({
                    'type': 'description',
                    'data': description
                }) + '\n'
                
            except Exception as e:
                yield json.dumps({
                    'type': 'error',
                    'data': str(e)
                }) + '\n'
        
        return Response(stream_with_context(generate()), mimetype='application/x-ndjson')
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'Message is required'}), 400
        
        message = data['message']
        
        # Call the blocking chat function
        response = chatbot.blocking_chat(message)
        
        return jsonify({
            'response': response,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 