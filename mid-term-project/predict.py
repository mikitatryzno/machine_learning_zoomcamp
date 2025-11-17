#!/usr/bin/env python
# Recipe Rating Prediction - Flask API for Serving Predictions

import pickle
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Download NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

# Load the trained model
try:
    with open('model.pkl', 'rb') as f:
        model_components = pickle.load(f)
        tfidf = model_components['tfidf']
        model = model_components['model']
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    tfidf = None
    model = None

# Text preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove HTML tags and special characters
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = nltk.word_tokenize(text)
    
    # Remove stopwords and lemmatize
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    cleaned_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    
    return ' '.join(cleaned_tokens)

@app.route('/')
def home():
    return """
    <h1>Recipe Rating Prediction API</h1>
    <p>Send a POST request to /predict with the following JSON structure:</p>
    <pre>
    {
        "text": "This recipe was amazing! I loved how easy it was to make.",
        "user_reputation": 5,
        "thumbs_up": 3,
        "thumbs_down": 0
    }
    </pre>
    """

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or tfidf is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    # Get data from request
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({'error': 'Please provide review text'}), 400
    
    # Extract features
    text = data.get('text', '')
    user_reputation = data.get('user_reputation', 1)
    thumbs_up = data.get('thumbs_up', 0)
    thumbs_down = data.get('thumbs_down', 0)
    
    # Preprocess text
    processed_text = preprocess_text(text)
    
    # Transform text using TF-IDF
    text_features = tfidf.transform([processed_text]).toarray()
    
    # Create metadata features
    interaction_score = thumbs_up - thumbs_down
    has_replies = 0  # Default to 0 for prediction
    
    meta_features = np.array([[user_reputation, thumbs_up, thumbs_down, interaction_score, has_replies]])
    
    # Combine features
    features = np.hstack((text_features, meta_features))
    
    # Make prediction
    prediction = model.predict(features)[0]
    
    # Get prediction probabilities
    probabilities = model.predict_proba(features)[0]
    confidence = max(probabilities)
    
    return jsonify({
        'predicted_rating': int(prediction),
        'confidence': float(confidence)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)