#!/usr/bin/env python
# Recipe Rating Prediction - Model Training Script

import pandas as pd
import numpy as np
import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

def main():
    print("Recipe Rating Prediction - Training Model")
    
    # Download NLTK resources
    print("Downloading NLTK resources...")
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
    
    # Load data
    print("Loading data...")
    try:
        # Try to load from UCI ML repo
        from ucimlrepo import fetch_ucirepo
        recipe_reviews = fetch_ucirepo(id=911)
        df = recipe_reviews.data.features
        print("Data loaded from UCI ML repository.")
    except:
        # Fallback to local file
        df = pd.read_csv('data/recipe_reviews.csv')
        print("Data loaded from local file.")
    
    # Data preprocessing
    print("Preprocessing data...")
    
    # Filter out rows with missing text
    df = df.dropna(subset=['text'])
    
    # Filter out rows where stars = 0 (no rating)
    # Create a copy to avoid SettingWithCopyWarning
    df_with_ratings = df[df['stars'] > 0].copy()
    
    print(f"Dataset after filtering: {df_with_ratings.shape}")
    
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
    
    # Apply text preprocessing
    print("Processing text data...")
    df_with_ratings.loc[:, 'processed_text'] = df_with_ratings['text'].apply(preprocess_text)
    
    # Feature engineering
    print("Engineering features...")
    df_with_ratings.loc[:, 'interaction_score'] = df_with_ratings['thumbs_up'] - df_with_ratings['thumbs_down']
    df_with_ratings.loc[:, 'has_replies'] = (df_with_ratings['reply_count'] > 0).astype(int)
    
    # Extract features for model training
    print("Preparing features for model...")
    
    # Split the data
    print("Splitting data into train and test sets...")
    X_text = df_with_ratings['processed_text']
    X_meta = df_with_ratings[['user_reputation', 'thumbs_up', 'thumbs_down', 'interaction_score', 'has_replies']]
    y = df_with_ratings['stars']
    
    X_train_text, X_test_text, X_train_meta, X_test_meta, y_train, y_test = train_test_split(
        X_text, X_meta, y, test_size=0.2, random_state=42
    )
    
    # Process text features
    print("Vectorizing text data...")
    tfidf = TfidfVectorizer(max_features=1000)
    X_train_text_tfidf = tfidf.fit_transform(X_train_text)
    X_test_text_tfidf = tfidf.transform(X_test_text)
    
    # Convert sparse matrix to dense for concatenation
    X_train_text_dense = X_train_text_tfidf.toarray()
    X_test_text_dense = X_test_text_tfidf.toarray()
    
    # Combine text features with metadata
    print("Combining features...")
    X_train = np.hstack((X_train_text_dense, X_train_meta.values))
    X_test = np.hstack((X_test_text_dense, X_test_meta.values))
    
    # Train the model
    print("Training model...")
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate on test set
    print("Evaluating model on test set...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Create a pipeline for saving
    print("Creating pipeline for prediction...")
    pipeline = Pipeline([
        ('tfidf', tfidf),
        ('classifier', model)
    ])
    
    # Save the model components for prediction
    print("Saving model components...")
    with open('model.pkl', 'wb') as f:
        pickle.dump({
            'tfidf': tfidf,
            'model': model
        }, f)
    
    print("Model saved as model.pkl")
    print("Training completed successfully!")

if __name__ == "__main__":
    main()