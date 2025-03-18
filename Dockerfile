# Use official Python image
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir flask numpy pandas scikit-learn pickle5

# Expose port
EXPOSE 5000

# Start Flask server
CMD ["python", "app.py"]

