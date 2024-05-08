# Use Python 3.10 base image
FROM python:3.10

# Set working directory inside the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire Modal_Data directory into the container
COPY Modal_Data /app/Modal_Data

RUN pip freeze

# Expose the port the app runs on
EXPOSE 8000

# Set environment variable to ensure Python outputs are sent straight to terminal without buffering
# ENV PYTHONUNBUFFERED=1

# Command to run the Flask application
# ENTRYPOINT [ “python” ]
# CMD ["./Modal_Data/app.py"]
