#!/usr/bin/env python
# Script to download the Recipe Reviews dataset

import os
import pandas as pd

def main():
    print("Downloading Recipe Reviews dataset...")
    
    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
    
    try:
        # Try to download from UCI ML repo
        from ucimlrepo import fetch_ucirepo
        recipe_reviews = fetch_ucirepo(id=911)
        
        # Get the features dataframe
        df = recipe_reviews.data.features
        
        # Save to CSV
        df.to_csv('data/recipe_reviews.csv', index=False)
        print("Dataset downloaded and saved to data/recipe_reviews.csv")
        
    except Exception as e:
        print(f"Error downloading from UCI ML repo: {e}")
        print("Please download the dataset manually from:")
        print("https://archive.ics.uci.edu/dataset/911/recipe+reviews+and+user+feedback")
        print("and save it as data/recipe_reviews.csv")

if __name__ == "__main__":
    main()