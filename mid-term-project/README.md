### Project Description

This project builds a machine learning model to predict recipe ratings (1-5 stars) based on user reviews and other features. Many users leave detailed comments about recipes without explicitly providing a star rating. Our solution analyzes review text and user interaction metrics to predict what rating a user would give, which can help recipe websites automatically assign ratings to reviews without explicit ratings.

The model can be used to:
Automatically assign ratings when users leave comments without explicit ratings
Identify discrepancies between review sentiment and assigned ratings
Improve recipe recommendation systems
Understand what factors contribute most to positive or negative recipe reviews

### Dataset

The project uses the "Recipe Reviews and User Feedback Dataset" from UCI Machine Learning Repository, which contains:

Recipe information (name, ranking, code)
User details (ID, name, reputation)
Review content (text, timestamps, reply count)
User interaction metrics (thumbs up/down)
Star ratings (1-5 scale, with 0 indicating no rating)


### How to Run the Project

## Prerequisites

Python 3.8+

pip or pipenv

## Setup Instructions

1. Clone the repository:

```bash
git clone https://github.com/yourusername/recipe-rating-prediction.git
cd recipe-rating-prediction
```

2. Set up the environment:

```bash
# Using pipenv
pipenv install

# Or using pip
pip install -r requirements.txt
```

3. Download the dataset:

```bash
python download_data.py
```

4. Train the model:

```bash
python train.py
```

5. Start the prediction service:

```bash
python predict.py
```

6. Access the API at http://localhost:5000 

### Using Docker/Podman

1. Build the container:

```bash
# Using Docker
docker build -t recipe-rating-predictor .

# Using Podman
podman build -t recipe-rating-predictor .
```

2. Run the container:

```bash
# Using Docker
docker run -p 5000:5000 recipe-rating-predictor

# Using Podman
podman run -p 5000:5000 recipe-rating-predictor
```

3. Access the API at http://localhost:5000


### API Usage

Endpoint: /predict

Method: POST

Request Body:

```json
{
  "text": "This recipe was amazing! I loved how easy it was to make and the flavors were perfect.",
  "user_reputation": 5,
  "thumbs_up": 3,
  "thumbs_down": 0
}
```

Response:

```json
{
  "predicted_rating": 5,
  "confidence": 0.92
}
```

### Model Information

The final model is a gradient boosting classifier that combines text features extracted using TF-IDF vectorization with user interaction metrics. The model was trained on 80% of the dataset and evaluated on the remaining 20%, achieving an accuracy of approximately 85%.

### Project Structure

```
recipe-rating-prediction/
├── README.md
├── notebook.ipynb          # Exploratory data analysis and model development
├── train.py                # Script to train the final model
├── predict.py              # Flask API for serving predictions
├── download_data.py        # Script to download the dataset
├── requirements.txt        # Python dependencies
├── Pipfile                 # Pipenv dependencies
├── Pipfile.lock            # Pipenv lock file
├── Dockerfile              # Docker/Podman container definition
└── data/                   # Data directory
    └── recipe_reviews.csv  # Dataset file
```