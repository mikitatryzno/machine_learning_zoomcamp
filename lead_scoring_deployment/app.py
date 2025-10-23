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