from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class IrisData(BaseModel):
    data: List[float]

@app.post("/predict")
async def get_prediction(iris_data: IrisData):
    prediction = predict(iris_data.data)
    return {"prediction": int(prediction[0])}
