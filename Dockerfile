# Use a lightweight Python image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy requirements and install them FIRST (saves build time)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your models, scalers, and code into the container
COPY . .

# Expose the port FastAPI runs on
EXPOSE 8000

# Start the server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]