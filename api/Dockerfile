# api/Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# # Install deps
# COPY requirements.txt /app/requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt \
#     && pip install --no-cache-dir uvicorn

# # App files
# COPY main.py /app/main.py
# COPY sentiment_model.pkl /app/sentiment_model.pkl
# # Not strictly required in the container, but handy for ad-hoc testing
# COPY test_data.json /app/test_data.json

# # Logs directory (mounted at runtime as a named volume too)
# RUN mkdir -p /app/logs

# EXPOSE 8000

# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# Install dependencies (pin scikit-learn to the version used to train)
COPY api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt \
    && pip install --no-cache-dir uvicorn

# Copy application code and assets
COPY main.py /app/main.py
COPY evaluate.py /app/evaluate.py
COPY test_data.json /app/test_data.json
COPY sentiment_model.pkl /app/sentiment_model.pkl

# Logs directory (also mounted as a named volume at runtime)
RUN mkdir -p /app/logs

EXPOSE 8000

# Run API
CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8000"]