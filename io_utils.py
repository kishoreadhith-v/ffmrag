import numpy as np
from PIL import Image
import requests
import io

# ImageNet mean and std for normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Preprocess function: downloads image, resizes, normalizes, and returns numpy array

def preprocess(img_path, target_size=(224, 224)):
    if img_path.startswith('http'):
        response = requests.get(img_path)
        img = Image.open(io.BytesIO(response.content)).convert('RGB')
    else:
        img = Image.open(img_path).convert('RGB')
    img = img.resize(target_size)
    img_np = np.array(img).astype(np.float32) / 255.0
    
    # Ensure mean and std are float32
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    
    img_np = (img_np - mean) / std
    img_np = np.transpose(img_np, (2, 0, 1))  # HWC to CHW
    img_np = np.expand_dims(img_np, axis=0)   # Add batch dimension
    return img_np.astype(np.float32)

# Postprocess function: prints top-5 predictions

def postprocess(prediction):
    pred = prediction[0]
    if isinstance(pred, list):
        pred = pred[0]
    pred = np.array(pred)
    top5_idx = pred[0].argsort()[-5:][::-1]
    print("Top-5 class indices:", top5_idx)
    print("Top-5 probabilities:", pred[0][top5_idx])
