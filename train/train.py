"""糖尿病预测 - 训练脚本"""
import os, urllib.request
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"
DATA = "data/diabetes.csv"
MODEL = "model/model.pkl"
# 用位置索引（而非列名）：ONNX 只认位置，索引版 Pipeline 才能被 skl2onnx 正确转换
# 对应 X 的列序：glucose=1, blood_pressure=2, skin_thickness=3, insulin=4, bmi=5
ZERO_IS_MISSING = [1, 2, 3, 4, 5]

# 1. 下载数据（没有才下）
os.makedirs("data", exist_ok=True)
if not os.path.exists(DATA):
    urllib.request.urlretrieve(URL, DATA)
    print("✅ 数据已下载到", DATA)

# 2. 读取数据
cols = ["pregnancies","glucose","blood_pressure","skin_thickness",
        "insulin","bmi","diabetes_pedigree","age","outcome"]
df = pd.read_csv(DATA, names=cols)
print("数据形状:", df.shape)

# 3. 特征 / 标签
#    "0=没测到"是业务约定，ONNX 只懂 NaN 这种数值语义；
#    所以在进 Pipeline 之前先把 0 换成 NaN（这一步留在模型边界之外）
X = df.drop("outcome", axis=1).to_numpy(dtype=float)
X[:, ZERO_IS_MISSING] = np.where(X[:, ZERO_IS_MISSING] == 0, np.nan, X[:, ZERO_IS_MISSING])
y = df["outcome"].to_numpy()

# 4. 划分训练/测试集（分层抽样，保持正负样本比例）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. 组装流水线：填补(NaN→中位数) → 标准化 → 逻辑回归
#    SimpleImputer 用默认的 missing_values=np.nan —— skl2onnx 只认它；0 已在上面换成 NaN
model = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000)),
])
model.fit(X_train, y_train)

# 7. 评估：准确率 + AUC（不平衡数据下 AUC 更诚实）
acc = accuracy_score(y_test, model.predict(X_test))
auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
print(f"✅ 测试集准确率: {acc:.4f}")
print(f"✅ 测试集 AUC:   {auc:.4f}")

# 8. 保存整条流水线（含填补+标准化，推理时才能复现同样的预处理）
os.makedirs("model", exist_ok=True)
joblib.dump(model, MODEL)
print(f"✅ 模型已保存到 {MODEL}")