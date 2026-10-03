"""糖尿病预测 - 训练脚本"""
import os, urllib.request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"
DATA = "data/diabetes.csv"

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

# 4. 划分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. 训练模型
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 6. 评估
acc = accuracy_score(y_test, model.predict(X_test))
print(f"✅ 测试集准确率: {acc:.4f}")