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