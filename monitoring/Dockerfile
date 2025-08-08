# monitoring/Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# # Install dashboard deps
# COPY requirements.txt /app/requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt \
#     && pip install --no-cache-dir streamlit scikit-learn matplotlib

# # App files
# COPY monitoring/ /app/monitoring/

# # Ensure logs dir exists (will be mounted as a named volume)
# RUN mkdir -p /app/logs

# EXPOSE 8501

# CMD ["streamlit", "run", "monitoring/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Install dashboard deps
COPY monitoring/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the dashboard code (and any local CSV you ship)
COPY monitoring/ /app/monitoring/

# Ensure logs dir exists (mounted as a named volume at /app/logs)
RUN mkdir -p /app/logs

EXPOSE 8501

# Run Streamlit
CMD ["streamlit","run","monitoring/streamlit_app.py","--server.port=8501","--server.address=0.0.0.0"]