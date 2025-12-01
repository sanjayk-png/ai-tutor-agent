# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

# Install system dependencies (needed for FAISS and PDF tools)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install the specific libraries your new code needs
RUN pip install --no-cache-dir fastapi uvicorn python-multipart google-generativeai PyPDF2 pillow faiss-cpu numpy

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Run the new application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]