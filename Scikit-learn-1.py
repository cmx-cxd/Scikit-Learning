# -*- coding: utf-8 -*-
# Scikit-learn-1.py
"""
### 实验题目
#### 题目1：线性回归预测糖尿病进展（回归问题）
**任务描述**：  
使用Scikit-learn内置的糖尿病数据集，建立一个线性回归模型来预测糖尿病进展指标。  
**步骤**：  
1. 加载糖尿病数据集（`load_diabetes`）。
2. 将数据集分为训练集和测试集（80%训练，20%测试）。
3. 训练线性回归模型。
4. 在测试集上进行预测。
5. 计算模型在测试集上的均方误差（MSE）和决定系数（R²）。
6. 可视化预测结果：绘制实际值与预测值的散点图，残差分析图，以及特征与目标值关系图。
"""

from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# 设置中文字体支持 - 使用更通用的方法
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']  # 多种中文字体备选
plt.rcParams['axes.unicode_minus'] = False    # 正确显示负号

fch = load_diabetes()
x = fch.data
y = fch.target
feature_names = fch.feature_names

x_train , x_test , y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

reg = LinearRegression()
reg.fit(x_train, y_train)

y_pred = reg.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("=" * 60)
print("线性回归模型结果")
print("=" * 60)
print(f"模型系数 (coef_): {reg.coef_}")
print(f"模型截距 (intercept_): {reg.intercept_}")
print(f"均方误差 (MSE): {mse:.4f}")
print(f"决定系数 (R²): {r2:.4f}")

# 打印特征重要性
print("\n特征重要性:")
for i, (name, coef) in enumerate(zip(feature_names, reg.coef_)):
    print(f"{name}: {coef:.4f}")

# 6. 可视化预测结果
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', lw=2)
plt.xlabel("实际值")
plt.ylabel("预测值")
plt.title("糖尿病进展预测: 实际值 vs 预测值")
plt.grid(alpha=0.3)
# 使用R^2而不是R²避免特殊字符问题
plt.text(0.05, 0.95, f"R^2 = {r2:.4f}\nMSE = {mse:.4f}", 
         transform=plt.gca().transAxes, 
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
plt.show()

# 附加：残差图分析
plt.figure(figsize=(10, 6))
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel("预测值")
plt.ylabel("残差 (实际值 - 预测值)")
plt.title("残差分析图")
plt.grid(alpha=0.3)
plt.show()

# 修正：特征与目标值关系图 - 创建足够的子图
n_features = len(feature_names)
n_cols = 3
n_rows = (n_features + n_cols - 1) // n_cols  # 计算需要的行数

plt.figure(figsize=(15, 5*n_rows))
for i, feature in enumerate(feature_names):
    plt.subplot(n_rows, n_cols, i+1)
    plt.scatter(x_test[:, i], y_test, alpha=0.3, label='实际值')
    plt.scatter(x_test[:, i], y_pred, alpha=0.3, label='预测值', color='orange')
    plt.xlabel(feature)
    plt.ylabel("目标值")
    plt.legend()
plt.tight_layout()
plt.suptitle("特征与目标值关系", y=1.02)
plt.show()