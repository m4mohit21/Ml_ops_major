# Use the official Python image as the base
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Install MLflow and additional dependencies
RUN pip install mlflow

# Expose the port for MLflow UI and the app (adjust as needed)
EXPOSE 5000 5001

# Copy the application files into the container
COPY . .

# Set environment variables (e.g., for MLflow tracking)
ENV MLFLOW_TRACKING_URI=http://mlflow-server:5000

# Set the command to run your Python script (main entry point)
CMD ["python", "iris_experiment.py"]
