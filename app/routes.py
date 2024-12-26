import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fastapi import APIRouter, HTTPException
from model.predict import ModelPredictor
from model.monitor import monitor_prediction_time

router = APIRouter()
predictor = ModelPredictor("model/next_word_svm.pkl")

@router.post("/predict/")
@monitor_prediction_time
def predict(context: str):
    try:
        result = predictor.predict(context)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
