import numpy as np
import pickle
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 데이터를 로드하고 훈련합니다.
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=42)
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X_train, y_train)

# 모델을 저장합니다.
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# 모델을 로드하고 예측 함수를 정의합니다.
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

def predict(data):
    return model.predict(np.array(data).reshape(1, -1))

