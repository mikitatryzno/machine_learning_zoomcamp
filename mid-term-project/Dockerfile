FROM python:3.9-slim

WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files
COPY download_data.py .
COPY train.py .
COPY predict.py .
COPY model.pkl* ./

# Download NLTK data
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')"

# Download the dataset if model.pkl doesn't exist
RUN if [ ! -f model.pkl ]; then python download_data.py && python train.py; fi

# Expose port
EXPOSE 5000

# Run the API server
CMD ["python", "predict.py"]