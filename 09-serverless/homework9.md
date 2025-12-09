### Downloading the Model Files
First, let's download the model files:
```python
import requests
import os
# Create a function to download files
def download_file(url, filename):
    response = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(response.content)
    print(f"Downloaded {filename}")
# Download model files
prefix = "https://github.com/alexeygrigorev/large-datasets/releases/download/hairstyle"
data_url = f"{prefix}/hair_classifier_v1.onnx.data"
model_url = f"{prefix}/hair_classifier_v1.onnx"

download_file(data_url, "hair_classifier_v1.onnx.data")
download_file(model_url, "hair_classifier_v1.onnx")
```

## Question 1: Output Node Name
To find the name of the output node, we need to inspect the ONNX model:

```python
# Install required packages
!pip install onnx onnxruntime pillow numpy requests
```

```python
import onnxruntime as ort

# Create an ONNX Runtime session
session = ort.InferenceSession("hair_classifier_v1.onnx")

# Get the input name
input_name = session.get_inputs()[0].name
print(f"Input name: {input_name}")

# Get the output name
output_name = session.get_outputs()[0].name
print(f"Output name: {output_name}")
```

Output name is `output`

### Preparing the Image

Let's use the provided code to download and prepare the image:

```python
from io import BytesIO
from urllib import request
from PIL import Image
import numpy as np

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

# Download the image
image_url = "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"
img = download_image(image_url)

# Display the image
img
```

## Question 2: Target Size

Based on the previous homework, we need to determine the target size for the image. From the model architecture in the previous homework, we used images of size 200x200.

```python
# Resize the image to the target size
target_size = (200, 200)  # Based on the previous homework
img_resized = prepare_image(img, target_size)

# Display the resized image
img_resized
```

Image size is `200x200`.

## Question 3: Image Preprocessing

Now we need to preprocess the image as we did in the previous homework:

```python
# Convert the image to a numpy array
img_array = np.array(img_resized)
print(f"Image shape: {img_array.shape}")

# Apply the same preprocessing as in the previous homework
# 1. Convert to float32
# 2. Normalize using ImageNet mean and std
# 3. Transpose to (C, H, W) format for PyTorch/ONNX

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

print(f"Preprocessed image shape: {img_array.shape}")
print(f"First pixel, R channel value: {img_array[0, 0, 0, 0]}")
```

First R value after pre-processing is `-1.0732939435925546`

## Question 4: Model Output

Now let's apply the model to the preprocessed image:

```python
import onnxruntime as ort
import numpy as np

# Create an ONNX Runtime session
session = ort.InferenceSession("hair_classifier_v1.onnx")

# Get the input name
input_name = session.get_inputs()[0].name
print(f"Input name: {input_name}")

# Get the output name
output_name = session.get_outputs()[0].name
print(f"Output name: {output_name}")

# Print the current image shape
print(f"Current image shape: {img_array.shape}")

# Run inference with the current image shape
# The model might be flexible with the batch size
try:
    outputs = session.run([output_name], {input_name: img_array.astype(np.float32)})
    output = outputs[0]
    
    # Apply sigmoid to get probability (since we used BCEWithLogitsLoss in training)
    probability = 1 / (1 + np.exp(-output))
    print(f"Model output (raw): {output[0][0]}")
    print(f"Probability: {probability[0][0]}")
except Exception as e:
    print(f"Error running inference: {e}")
    
    # Try with a different approach - get the expected shape from the error message
    print("\nTrying alternative approach...")
    
    # Try with a fixed batch size of 1
    # The model expects (batch_size, 3, 200, 200)
    # Our image is already (1, 3, 200, 200)
    
    # Let's try running inference without the batch dimension
    try:
        # Remove batch dimension
        img_array_no_batch = img_array[0]
        print(f"Shape without batch: {img_array_no_batch.shape}")
        
        # Run inference
        outputs = session.run([output_name], {input_name: img_array_no_batch.astype(np.float32)})
        output = outputs[0]
        
        # Apply sigmoid to get probability
        probability = 1 / (1 + np.exp(-output))
        print(f"Model output (raw): {output[0]}")
        print(f"Probability: {probability[0]}")
    except Exception as e:
        print(f"Error with alternative approach: {e}")
```

Model output is `0.09`

### Prepare the Lambda Code

Now let's create a Python file with all the necessary code for AWS Lambda:

```python

# lambda_function.py
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
```

Save this code to a file named lambda_function.py.

## Question 5: Docker Base Image Size

Let's pull the Docker base image and check its size:

```bash
# Pull the image using Podman
podman pull agrigorev/model-2025-hairstyle:v1

# Check the image size
podman images
```

Docker image size is `621 MB`

## Question 6: Extending the Docker Image

Now let's create a Dockerfile to extend the base image:

```dockerfile
FROM agrigorev/model-2025-hairstyle:v1

# Install required packages
RUN pip install --no-cache-dir pillow numpy onnxruntime

# Copy the Lambda function code
COPY lambda_function.py .

# Set the Lambda handler
CMD ["lambda_function.lambda_handler"]
```

Save this to a file named Dockerfile.

Now let's build and run the Docker container:

```bash
# Build the image
podman build -t hair-classifier-lambda .

# Run the container
podman run -p 9000:8080 hair-classifier-lambda
```

To test the Lambda function with the provided image, we can use the following Python code:

```python
import requests
import base64
import json

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Download the test image
image_url = "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"
response = requests.get(image_url)
with open("test_image.jpg", "wb") as f:
    f.write(response.content)

# Encode the image
base64_image = encode_image("test_image.jpg")

# Prepare the event
event = {
    "image": base64_image
}

# Invoke the Lambda function
response = requests.post(
    "http://localhost:9000/2015-03-31/functions/function/invocations",
    json=event
)

# Parse the response
result = json.loads(response.text)
print(result)
```

Model output (docker) is `TBA`.