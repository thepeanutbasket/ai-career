# FastAPI allows applications to interact with a Python machine learning model through HTTP requests.
# It exposes a machine learning model through HTTP endpoints so that other applications can send data and receive predictions.

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI() # creating a FastAPI application

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

    return {
        "message": "Passenger data received",
        "passenger": passenger
    }

