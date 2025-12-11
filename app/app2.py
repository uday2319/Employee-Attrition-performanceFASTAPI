from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()
pipeline = joblib.load("employee_attrition.pkl")  
columns = joblib.load("columns.pkl")

def process_input(data):
    x = np.zeros(len(columns))
    for i, col in enumerate(columns):
        val = data.get(col, 0)
        try:
            x[i] = float(val)
        except:
            x[i] = 0
    return x.reshape(1, -1)

@app.get("/")
def home():
    return {"message": "FastAPI running"}

@app.post("/predict")
def predict(data: dict):
    x = process_input(data)
    pred = int(pipeline.predict(x)[0])


    result = "Yes" if pred == 1 else "No"

    return {"Attrition": result}
