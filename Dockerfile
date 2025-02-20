# Base Image for GPU container support
FROM pytorch/conda-cuda:latest

# Set the working directory
WORKDIR /app

# Copy requirements first for dependency caching
COPY requirements.txt .

# Install dependencies using python
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt gputil

# Copy the rest of the application code
COPY . .

# Command to run the application
CMD ["python", "ray_rllib_trainer.py"]