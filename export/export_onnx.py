"""糖尿病预测 - ONNX 导出 + 一致性验证"""
import os
import numpy as np
import pandas as pd
import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort

PKL = "model/model.pkl"
ONNX = "model/model.onnx"
DATA = "data/diabetes.csv"
COLS = ["pregnancies", "glucose", "blood_pressure", "skin_thickness",
        "insulin", "bmi", "diabetes_pedigree", "age", "outcome"]
# 与 train.py 一致的边界约定：这些列里的 0 表示"没测到"，需要换成 NaN
ZERO_IS_MISSING = [1, 2, 3, 4, 5]

# 1. 加载 M2 训练好的整条流水线
model = joblib.load(PKL)
print(f"✅ 已加载 {PKL}")

# 2. 整条流水线 → ONNX（输入 8 个 float 特征）
#    zipmap=False：概率输出为矩阵，而不是 [{0:0.3, 1:0.7}] 这种 list[dict]
onnx_model = convert_sklearn(
    model,
    initial_types=[("input", FloatTensorType([None, 8]))],
    options={"zipmap": False},
)
os.makedirs("model", exist_ok=True)
with open(ONNX, "wb") as f:
    f.write(onnx_model.SerializeToString())
print(f"✅ 已导出 {ONNX}")

# 3. 用 ONNX Runtime 推理（全程不经过 sklearn）
sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name
print(f"ONNX 输入名: {input_name}, 形状: {sess.get_inputs()[0].shape}")

# 4. 一致性验证：同一批数据，pkl 与 onnx 的概率必须一致
df = pd.read_csv(DATA, names=COLS)
X_sample = df.drop("outcome", axis=1).head(5).to_numpy(dtype=float)

# 边界约定：先做 0→NaN，模型才知道哪些是缺失值（pkl 和 onnx 用同一份数据）
X_sample[:, ZERO_IS_MISSING] = np.where(
    X_sample[:, ZERO_IS_MISSING] == 0, np.nan, X_sample[:, ZERO_IS_MISSING]
)
X_sample = X_sample.astype(np.float32)

pkl_proba = model.predict_proba(X_sample)[:, 1]
onnx_proba = sess.run(None, {input_name: X_sample})[1][:, 1]

diff = float(np.abs(pkl_proba - onnx_proba).max())
print("pkl  概率:", np.round(pkl_proba, 6))
print("onnx 概率:", np.round(onnx_proba, 6))
print(f"✅ 最大误差: {diff:.2e}  (要求 < 1e-5)")
print("✅ 一致性验证通过" if diff < 1e-5 else "❌ 误差过大，导出有问题")
