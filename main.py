# # main.py
# # Alex Holyk

# from fastapi import FastAPI, HTTPException, status
# from pydantic import BaseModel
# import joblib
# # import random

# app = FastAPI(title="Sentiment Analysis API")

# # Load the trained model
# file_name = 'sentiment_model.pkl'
# try:
#     model = joblib.load(file_name)
#     print("Model loaded successfully.")
# except FileNotFoundError:
#     print(f"Error: Model file {file_name} not found.")
#     model = None

# # Define the input data model
# class PredictionInput(BaseModel):
#     text: str

# # # Health check endpoint
# # @app.get("/health")
# # def health_check():
# #     """ 
# #     This endpoint is used to verify that the API server is running and responsive.
# #     """
# #     return {"status": "ok"}

# # Predict sentiment endpoint
# @app.post("/predict")
# def predict(input_data: PredictionInput):
#     """
#     Takes a text input and returns a sentiment prediction.
#     """
#     if model is None:
#         raise HTTPException(
#             status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
#             detail="Model is not loaded. Cannot make predictions."
#         )
    
#     prediction = model.predict([input_data.text])
#     return {"sentiment": prediction[0]}














# main.py
# Alex Holyk

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import Literal
from datetime import datetime, timezone
import os
import json
import joblib
from threading import Lock

app = FastAPI(title="Sentiment Analysis API")

# ---- Model loading ----
MODEL_FILE = "sentiment_model.pkl"
try:
    model = joblib.load(MODEL_FILE)
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file {MODEL_FILE} not found.")
    model = None

# ---- Logging setup ----
LOG_DIR = "logs"            # folder at project root
LOG_FILE = "prediction_logs.json"  # newline-delimited JSON (one log/event per line)
_os_lock = Lock()

def _ensure_log_dir():
    os.makedirs(LOG_DIR, exist_ok=True)

def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()

def _log_prediction(request_text: str, predicted_sentiment: str, true_label: str) -> None:
    """
    Append a single-line JSON object to logs/prediction_logs.json.
    """
    _ensure_log_dir()
    record = {
        "timestamp": _utc_timestamp(),
        "request_text": request_text,
        "predicted_sentiment": predicted_sentiment,
        "true_label": true_label,
    }
    line = json.dumps(record, ensure_ascii=False)
    # Use a lock to avoid interleaving writes under concurrency
    with _os_lock:
        with open(os.path.join(LOG_DIR, LOG_FILE), "a", encoding="utf-8") as f:
            f.write(line + "\n")

# ---- Request schema ----
class PredictionInput(BaseModel):
    text: str = Field(..., description="Raw text to classify")
    # In lieu of a frontend feedback form, client must provide true label in the request (e.g., via Postman).
    true_label: Literal["positive", "negative", "neutral"] = Field(
        ..., description="User-provided ground truth label for this text"
    )

# ---- Endpoint ----
@app.post("/predict")
def predict(input_data: PredictionInput):
    """
    Takes text and a user-provided true_label, returns model prediction,
    and logs {timestamp, request_text, predicted_sentiment, true_label}
    to logs/prediction_logs.json (one JSON object per line).
    """

    print(f"[BOOT] Using PredictionInput fields: text, true_label")
    
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Cannot make predictions."
        )

    # Model expects an iterable of texts
    prediction = model.predict([input_data.text])
    predicted = str(prediction[0])

    # Log the event (always)
    try:
        _log_prediction(
            request_text=input_data.text,
            predicted_sentiment=predicted,
            true_label=input_data.true_label,
        )
    except Exception as e:
        # If logging fails, still return a prediction
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to write prediction log: {e}"
        )

    return {"sentiment": predicted}


# Can now run in terminal: uvicorn main:app --reload
