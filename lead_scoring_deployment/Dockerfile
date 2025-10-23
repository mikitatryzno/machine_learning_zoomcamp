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