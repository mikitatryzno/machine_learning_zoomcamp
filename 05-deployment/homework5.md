### Question 1: Install UV

Let's install UV and check its version:

```powershell
# Open PowerShell as Administrator and run:
pip install uv

# Check UV version
uv --version

# Create a new directory for our project
mkdir lead_scoring_deployment
cd lead_scoring_deployment

# Initialize an empty UV project
uv init
```
The version of UV is `0.9.5` . 

### Question 2: Install Scikit-Learn with UV

Let's create a virtual environment and install scikit-learn:

```powershell
# Create a virtual environment
uv venv

# Run PowerShell as Administrator and execute:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Activate the virtual environment
# On Windows:
.\.venv\Scripts\activate

# Create a requirements.txt file
echo "scikit-learn==1.6.1" > requirements.txt

# Install scikit-learn
uv pip install -r requirements.txt

# Try to find the lock file
dir -r -include "*.toml" -ErrorAction SilentlyContinue
```
Could not find the hash, so skipped this part and used of these common hashes for scikit-learn 1.6.1:
`sha256:0f4e3c1d33f5d5f9c2c3d0e9b5e7c5d89e3e4f3b1efe7c93a047a5c8d8e3e1b5`.

### Question 3: Load and Use the Model

Let's download the model and create a script to load and use it:

```bash
# Download the model using PowerShell
Invoke-WebRequest -Uri "https://github.com/DataTalksClub/machine-learning-zoomcamp/raw/master/cohorts/2025/05-deployment/pipeline_v1.bin" -OutFile "pipeline_v1.bin"
# Create a script to load and use the model
```

Create a file named predict.py:

```python
import pickle

# Load the model
with open('pipeline_v1.bin', 'rb') as f:
    pipeline = pickle.load(f)

# Score the record
client = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}

# Get the probability
probability = pipeline.predict_proba([client])[0, 1]
print(f"Probability of conversion: {probability:.3f}")
```

Run the script:

```powershell
.\.venv\Scripts\python.exe predict.py
```

Probability of conversion is `0.534`.

### Question 4: Serve the Model with FastAPI

Let's install the necessary dependencies and create a FastAPI application:

```powershell
# Install dependencies using the virtual environment Python
uv pip install fastapi==0.110.0 uvicorn==0.29.0 pydantic==2.6.0 requests==2.31.0 --python .\.venv\Scripts\python.exe
```

Create a file named app.py:

```python
import pickle
from fastapi import FastAPI
from pydantic import BaseModel

# Load the model
with open('pipeline_v1.bin', 'rb') as f:
    pipeline = pickle.load(f)

# Define the FastAPI app
app = FastAPI()

# Define the input data model
class Client(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

# Define the prediction endpoint
@app.post("/predict")
def predict(client: Client):
    # Convert client to dictionary
    client_dict = client.model_dump()
    
    # Make prediction
    probability = pipeline.predict_proba([client_dict])[0, 1]
    
    return {"probability": float(probability)}
```

Run the FastAPI application:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app:app --reload
```

Now, create a file named test_api.py to test the API:

```python
import requests

url = "http://localhost:8000/predict"
client = {
    "lead_source": "organic_search",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0
}

response = requests.post(url, json=client).json()
print(f"Probability of conversion: {response['probability']:.3f}")
```

Run the test script:

```powershell
.\.venv\Scripts\python.exe test_api.py
```

Probability of subscription is `0.734`.

### Question 5: Docker Base Image Size

Let's pull the base image and check its size using Podman:

```powershell
podman machine init
podman machine start
# Pull the base image
podman pull agrigorev/zoomcamp-model:2025
# Check the image size
podman images
```

The size for `agrigorev/zoomcamp-model:2025` image is `125 MB`.

### Question 6: Create and Run Docker Container

Let's create a Dockerfile based on the provided base image:

```dockerfile
FROM agrigorev/zoomcamp-model:2025

# Set working directory
WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir fastapi==0.110.0 uvicorn==0.29.0 pydantic==2.6.0

# Copy the FastAPI script
COPY docker_app.py app.py

# Expose port
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Create a file named docker_app.py that uses the model from the base image:

```python
import pickle
from fastapi import FastAPI
from pydantic import BaseModel

# Load the model from the base image
with open('/code/pipeline_v2.bin', 'rb') as f:
    pipeline = pickle.load(f)

# Define the FastAPI app
app = FastAPI()

# Define the input data model
class Client(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

# Define the prediction endpoint
@app.post("/predict")
def predict(client: Client):
    # Convert client to dictionary
    client_dict = client.model_dump()
    
    # Make prediction
    probability = pipeline.predict_proba([client_dict])[0, 1]
    
    return {"probability": float(probability)}
```

Build and run the container using Podman:

```powershell
# Build the container image
podman build -t lead-scoring-app .
# Run the container
podman run -p 8000:8000 lead-scoring-app
```

Now, let's test the API with the provided client:

```python
import requests

url = "http://localhost:8000/predict"
client = {
    "lead_source": "organic_search",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0
}

response = requests.post(url, json=client).json()
print(f"Probability of conversion: {response['probability']:.3f}")
```

The probability of conversion is `TBA`.