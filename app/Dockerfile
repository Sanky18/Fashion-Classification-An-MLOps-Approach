# Use the official Python image
FROM python:3.9

RUN apt-get update && apt-get install -y libhdf5-dev pkg-config

# Set the working directory in the container
WORKDIR /app

# Copy only the specific files you want
COPY fashion_mnist_model.h5 /app
COPY . /app

# Install dependencies from requirements.txt

RUN pip install --no-cache-dir -r requirement.txt uvicorn python-multipart prometheus-fastapi-instrumentator psutil


# Expose the port the app runs on
EXPOSE 8000

# Command to run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]