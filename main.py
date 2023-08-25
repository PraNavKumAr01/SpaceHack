from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from autogluon.tabular import TabularDataset, TabularPredictor

app = FastAPI()

# Define the expected JSON input structure using Pydantic
class InputData(BaseModel):
    orbitalPeriod: float
    planetaryRadius: float
    eqlbmTemperature: float
    insolationFlux: float
    surfaceGravity: float

save_path = 'agModels-predictClass3'

predictor = TabularPredictor.load(save_path)

@app.post("/predict/")
async def predict(data: InputData):
    try:
        # Convert JSON data to a numpy array
        x = np.array([data.orbitalPeriod, 
                      data.planetaryRadius,
                      data.eqlbmTemperature,
                      data.insolationFlux, 
                      data.surfaceGravity]
                    )

        columns = ['OrbitalPeriod[days', 'PlanetaryRadius[Earthradii',
                   'EquilibriumTemperature[K', 'InsolationFlux[Earthflux', 'StellarSurfaceGravity[log10(cm/s**2)']

        df = pd.DataFrame([x], columns=columns)

        # Create a TabularDataset
        test_data = TabularDataset(df)

        # Get predictions from the predictor
        y_pred = predictor.predict(test_data)

        predicted_value = y_pred.iloc[0]

        # Convert predicted value to boolean (True or False)
        result = bool(predicted_value)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
