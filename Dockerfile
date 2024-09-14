# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install git
RUN apt-get update && apt-get install -y git

# Clone the project repo into the container
RUN git clone https://github.com/EllaN12/Customer_Churn_Analysis-.git /app

# Change directory to the cloned project
WORKDIR /app

# Install the required Python packages
RUN pip install -r requirements.txt

# Expose the port (modify if your app uses a different port)
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]
