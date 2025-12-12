from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()


pipeline = joblib.load("employee_attrition.pkl")
columns = joblib.load("columns.pkl")

@app.get("/")
def home():
    return {"message": "FastAPI running"}

@app.post("/predict")
def predict(data: dict):

    
    df = pd.DataFrame([data], columns=columns)
    pred = int(pipeline.predict(df)[0])
    result = "Yes" if pred == 1 else "No"

    return {"Attrition": result}
