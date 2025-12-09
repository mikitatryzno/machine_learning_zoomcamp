FROM agrigorev/model-2025-hairstyle:v1

# Install required packages
RUN pip install --no-cache-dir pillow numpy onnxruntime

# Copy the Lambda function code
COPY lambda_function.py .

# Set the Lambda handler
CMD ["lambda_function.lambda_handler"]