"""糖尿病预测 - FastAPI 推理服务（ONNX Runtime）

工程细节：
- 启动时加载一次模型，并记录模型指纹（sha256）与加载时间，/health 可查
- 每个请求带 X-Request-ID 与 X-Process-Time 响应头，并写一行日志
- 模型加载失败：启动即报错（不带着坏模型对外服务）
- 推理异常：返回干净的 500，不把堆栈甩给调用方
"""
import hashlib
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("diabetes-api")

MODEL_PATH = "model/model.onnx"
# 边界约定（与 train.py / export_onnx.py 一致）：这些列里的 0 表示"没测到"，要换成 NaN
ZERO_IS_MISSING = [1, 2, 3, 4, 5]


class PatientFeatures(BaseModel):
    """一次预测请求。字段顺序必须与训练时的特征列顺序一致。

    每列都加了范围校验：类型合法但数值荒谬的输入（如 -999）会被挡在门口，
    避免它一路静默算出一个错误概率。
    注意 ge=0 是刻意的——这些列里的 0 表示"没测到"，是合法输入。
    """
    pregnancies: float = Field(ge=0.0, le=20, description="怀孕次数")
    glucose: float = Field(ge=0.0, le=600, description="血糖值（mg/dL）")
    blood_pressure: float = Field(ge=0.0, le=250, description="舒张压（mm Hg）")
    skin_thickness: float = Field(ge=0.0, le=100, description="皮褶厚度（mm）")
    insulin: float = Field(ge=0.0, le=1000, description="胰岛素（μU/mL）")
    bmi: float = Field(ge=0.0, le=100, description="体质指数")
    diabetes_pedigree: float = Field(ge=0.0, le=10, description="糖尿病家族遗传函数")
    age: float = Field(ge=0.0, le=120, description="年龄（岁）")
    # 阈值做成请求参数：业务方（比如医生）可以不改代码就调整"宁可误报"的程度
    threshold: float = Field(0.5, ge=0.0, le=1.0, description="判定为患病的概率阈值")


class Prediction(BaseModel):
    probability: float
    prediction: int
    threshold: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时加载一次。加载失败就让启动失败——绝不能带着坏模型对外服务
    try:
        app.state.session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    except Exception:
        logger.exception("模型加载失败: %s", MODEL_PATH)
        raise

    app.state.input_name = app.state.session.get_inputs()[0].name
    with open(MODEL_PATH, "rb") as f:
        app.state.model_version = hashlib.sha256(f.read()).hexdigest()[:12]
    app.state.loaded_at = datetime.now().isoformat(timespec="seconds")

    logger.info("模型已加载: %s (version=%s)", MODEL_PATH, app.state.model_version)
    yield
    logger.info("服务正在停止")


app = FastAPI(title="糖尿病预测服务", lifespan=lifespan)


@app.middleware("http")
async def add_request_headers(request: Request, call_next):
    """给每个响应加上请求 ID 与耗时，并记录一行访问日志。"""
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    start = time.perf_counter()
    response = await call_next(request)
    cost_ms = (time.perf_counter() - start) * 1000

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = f"{cost_ms:.1f}ms"
    logger.info(
        "%s %s -> %d rid=%s %.1fms",
        request.method, request.url.path, response.status_code, request_id, cost_ms,
    )
    return response


@app.get("/health")
def health():
    """健康检查：顺便暴露当前加载的模型版本，避免"静默跑旧模型"。"""
    stat = os.stat(MODEL_PATH)
    return {
        "status": "ok",
        "model": MODEL_PATH,
        "model_version": app.state.model_version,
        "model_modified": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        "loaded_at": app.state.loaded_at,
    }


@app.post("/predict", response_model=Prediction)
def predict(req: PatientFeatures):
    t0 = time.perf_counter()

    # 1) 按训练时的列序组装成 (1, 8) 的 float32 张量
    x = np.array([[
        req.pregnancies, req.glucose, req.blood_pressure, req.skin_thickness,
        req.insulin, req.bmi, req.diabetes_pedigree, req.age,
    ]], dtype=np.float32)

    # 2) 边界约定：0 → NaN。必须在喂进 ONNX 之前做，
    #    否则 0 会被当成真实测量值，静默算出错误的概率
    x[:, ZERO_IS_MISSING] = np.where(x[:, ZERO_IS_MISSING] == 0, np.nan, x[:, ZERO_IS_MISSING])

    # 3) 推理：取出正类（患病）概率
    try:
        proba = float(app.state.session.run(None, {app.state.input_name: x})[1][0, 1])
    except Exception:
        # 记录完整堆栈给运维，但只把一句人话返回给调用方
        logger.exception("推理失败")
        raise HTTPException(status_code=500, detail="模型推理失败，请稍后重试")

    # 4) 用请求里的阈值判定（默认 0.5）
    prediction = int(proba >= req.threshold)
    logger.info(
        "predict proba=%.6f threshold=%.2f prediction=%d %.1fms",
        proba, req.threshold, prediction, (time.perf_counter() - t0) * 1000,
    )

    return Prediction(
        probability=proba,
        prediction=prediction,
        threshold=req.threshold,
    )
