import json
import base64
import numpy as np
import onnxruntime as ort
from io import BytesIO
from PIL import Image

# Load the model
session = ort.InferenceSession("hair_classifier_empty.onnx")
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

def prepare_image(img, target_size=(200, 200)):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def preprocess_image(img):
    # Convert to numpy array
    img_array = np.array(img)
    
    # Convert to float32 and normalize to [0, 1]
    img_array = img_array.astype('float32') / 255.0
    
    # Normalize using ImageNet mean and std
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std
    
    # Transpose from (H, W, C) to (C, H, W)
    img_array = img_array.transpose(2, 0, 1)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def lambda_handler(event, context):
    try:
        # Decode the base64-encoded image
        image_bytes = base64.b64decode(event['image'])
        img = Image.open(BytesIO(image_bytes))
        
        # Prepare and preprocess the image
        img = prepare_image(img)
        img_array = preprocess_image(img)
        
        # Run inference
        outputs = session.run([output_name], {input_name: img_array})
        output = outputs[0][0][0]
        
        # Apply sigmoid to get probability
        probability = 1 / (1 + np.exp(-output))
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'prediction': float(output),
                'probability': float(probability)
            })
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            })
        }

# For local testing
if __name__ == "__main__":
    from urllib import request
    
    def download_image(url):
        with request.urlopen(url) as resp:
            buffer = resp.read()
        return base64.b64encode(buffer).decode('utf-8')
    
    # Test with the provided image
    image_url = "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"
    event = {'image': download_image(image_url)}
    result = lambda_handler(event, None)
    print(result)