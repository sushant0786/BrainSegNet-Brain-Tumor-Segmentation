# 1. Start from a tiny Python image
FROM python:3.11-slim

# 2. Keep logs unbuffered and pick a working folder
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# 3. Install Python dependencies first 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of your source code
COPY . .

# 5. Expose the port FastAPI will listen on
EXPOSE 8000

# 6. Launch the API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
