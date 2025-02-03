# Base image
FROM python:3.12.8

# Set the working directory
WORKDIR /app

# Copy files here
COPY . /app

# Install dependencies here
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Set environment variables here
ENV PYTHONBUFFERED=1

# Expose ports here if you need to
# EXPOSE 8000

# Command to run the application (Modify as per your script entry point)
CMD ["python", "ray_rllib_trainer.py"]