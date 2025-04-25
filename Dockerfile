FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker caching
COPY requirements.txt .

# Install Python dependencies and verify streamlit is properly installed
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir streamlit && \
    which streamlit

# Install spaCy model
RUN python -m spacy download en_core_web_lg

# Copy the application code and data
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

# Use absolute path to streamlit to ensure it's found
CMD ["python", "-m", "streamlit", "run", "app.py"]