from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os

app = FastAPI(
    title="AI Smart Re-Triage API",
    description="Prototype API for ER AI Smart Re-Triage Assistant",
    version="1.0.0"
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
model = joblib.load(MODEL_PATH)

class PatientInput(BaseModel):
    age: int
    initial_triage_level: int
    systolic_bp: float
    diastolic_bp: float
    pulse: float
    rr: float
    spo2: float
    temp: float
    wait_minutes: float
    repeated_measure_count: int
    delta_pulse: float
    delta_bp: float

def map_color(recommendation: str) -> str:
    if recommendation == "retriage_now":
        return "red"
    elif recommendation == "observe_closely":
        return "yellow"
    return "green"

@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "AI Smart Re-Triage API is running"
    }

@app.post("/predict")
def predict(data: PatientInput):
    input_df = pd.DataFrame([{
        "age": data.age,
        "initial_triage_level": data.initial_triage_level,
        "systolic_bp": data.systolic_bp,
        "diastolic_bp": data.diastolic_bp,
        "pulse": data.pulse,
        "rr": data.rr,
        "spo2": data.spo2,
        "temp": data.temp,
        "wait_minutes": data.wait_minutes,
        "repeated_measure_count": data.repeated_measure_count,
        "delta_pulse": data.delta_pulse,
        "delta_bp": data.delta_bp
    }])

    prediction = model.predict(input_df)[0]

    probabilities = model.predict_proba(input_df)[0]
    class_names = model.classes_

    prob_dict = {
        class_names[i]: float(probabilities[i])
        for i in range(len(class_names))
    }

    return {
        "recommendation": prediction,
        "status_color": map_color(prediction),
        "probabilities": prob_dict
    }
