FROM python:3.12.8

WORKDIR /app

# Copy requirements first for dependency caching
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

CMD ["python", "ray_rllib_trainer.py"]
