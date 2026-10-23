"""糖尿病预测 - FastAPI 推理服务（ONNX Runtime）"""
from contextlib import asynccontextmanager

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = "model/model.onnx"
# 边界约定（与 train.py / export_onnx.py 一致）：这些列里的 0 表示"没测到"，要换成 NaN
ZERO_IS_MISSING = [1, 2, 3, 4, 5]


class PatientFeatures(BaseModel):
    """一次预测请求。字段顺序必须与训练时的特征列顺序一致。"""
    pregnancies: float
    glucose: float
    blood_pressure: float
    skin_thickness: float
    insulin: float
    bmi: float
    diabetes_pedigree: float
    age: float
    # 阈值做成请求参数：业务方（比如医生）可以不改代码就调整"宁可误报"的程度
    threshold: float = Field(0.5, ge=0.0, le=1.0, description="判定为患病的概率阈值")


class Prediction(BaseModel):
    probability: float
    prediction: int
    threshold: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 服务启动时加载一次 ONNX（推理全程不依赖 sklearn）
    app.state.session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    app.state.input_name = app.state.session.get_inputs()[0].name
    yield


app = FastAPI(title="糖尿病预测服务", lifespan=lifespan)


@app.post("/predict", response_model=Prediction)
def predict(req: PatientFeatures):
    # 1) 按训练时的列序组装成 (1, 8) 的 float32 张量
    x = np.array([[
        req.pregnancies, req.glucose, req.blood_pressure, req.skin_thickness,
        req.insulin, req.bmi, req.diabetes_pedigree, req.age,
    ]], dtype=np.float32)

    # 2) 边界约定：0 → NaN。必须在喂进 ONNX 之前做，
    #    否则 0 会被当成真实测量值，静默算出错误的概率
    x[:, ZERO_IS_MISSING] = np.where(x[:, ZERO_IS_MISSING] == 0, np.nan, x[:, ZERO_IS_MISSING])

    # 3) 推理：取出正类（患病）概率
    proba = float(app.state.session.run(None, {app.state.input_name: x})[1][0, 1])

    # 4) 用请求里的阈值判定（默认 0.5）
    return Prediction(
        probability=proba,
        prediction=int(proba >= req.threshold),
        threshold=req.threshold,
    )
