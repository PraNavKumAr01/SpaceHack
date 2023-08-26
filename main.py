from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import pickle

app = FastAPI()

# Define the expected JSON input structure using Pydantic
class InputData(BaseModel):
    OrbitalPeriod: float
    PlanetaryRadius: float
    EquilibriumTemperature: float
    InsolationFlux: float
    StellarSurfaceGravity: float

loaded_model = joblib.load('modelmain.joblib')

@app.post("/predict")
async def predict(data: InputData):
    try:
        # Convert JSON data to a numpy array
        x = np.array([data.OrbitalPeriod, 
                      data.PlanetaryRadius,
                      data.EquilibriumTemperature,
                      data.InsolationFlux, 
                      data.StellarSurfaceGravity]
                    )

        columns = ['OrbitalPeriod', 'PlanetaryRadius',
                   'EquilibriumTemperature', 'InsolationFlux', 'StellarSurfaceGravity']

        df = pd.DataFrame([x], columns=columns)

        # Get predictions from the predictor
        y_pred = loaded_model.predict(df)

        # Convert predicted value to boolean (True or False)
        result = bool(y_pred)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
