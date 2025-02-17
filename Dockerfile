# Use the official Python image
FROM python:3.10

# Set the working directory
WORKDIR /app

# Copy all files into the container
COPY . /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8080 (Google Cloud Run default)
EXPOSE 8080

# Run the app
CMD ["python", "app.py"]
