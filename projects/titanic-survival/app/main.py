# FastAPI allows applications to interact with a Python machine learning model through HTTP requests.
# It exposes a machine learning model through HTTP endpoints so that other applications can send data and receive predictions.

import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

app = FastAPI() # creating a FastAPI application

model = joblib.load("model.pkl")

class Passenger(BaseModel): # declaring the data structure
    pclass: int
    sex: str
    age: float
    sibsp: int
    parch: int
    fare: float
    embarked: str

@app.get("/")   # defining an endpoint / get: HTTP method that brings in the information / "/": address
def home():
    return {"message": "Titanic ML API is running"}

@app.post("/predict")
def predict(passenger: Passenger):

    input_data = pd.DataFrame([
        passenger.model_dump() # converting Pydantic object to Python dictionary
    ])

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    survival_probability = probabilities[1]

    return {
        "prediction": int(prediction),
        "survived": bool(prediction),
        "survival_probability": float(survival_probability)
    }
