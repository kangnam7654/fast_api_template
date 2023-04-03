from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


model = 'Machine Learning Model'

# API 엔드포인트에 입력 데이터 형식을 정의합니다.
class InputData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


# API 엔드포인트를 작성합니다.
@app.post("/predict")
async def predict(input_data: InputData):
    # 입력 데이터를 모델에 전달하여 예측 결과를 반환합니다.
    input_vector = [input_data.sepal_length, input_data.sepal_width, input_data.petal_length, input_data.petal_width]
    prediction = model.predict([input_vector])[0]
    return {"prediction": prediction}

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/hello")
async def hello():
    return {"test": "YAS!"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)