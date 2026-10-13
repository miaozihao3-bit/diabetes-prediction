"""糖尿病预测 - 训练脚本"""
import os, urllib.request
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"
DATA = "data/diabetes.csv"
MODEL = "model/model.pkl"
ZERO_IS_MISSING = ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"]

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
X = df.drop("outcome", axis=1)
y = df["outcome"]

# 4. 划分训练/测试集（分层抽样，保持正负样本比例）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. 预处理：只对"0 其实是缺失"的 5 列填补中位数，其余列原样通过
preprocess = ColumnTransformer([
    ("impute", SimpleImputer(missing_values=0, strategy="median"), ZERO_IS_MISSING),
], remainder="passthrough")

# 6. 组装流水线：填补 → 标准化 → 逻辑回归
model = Pipeline([
    ("preprocess", preprocess),
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