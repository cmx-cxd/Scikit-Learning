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

from sklearn.datasets import load_diabetes                  # sklearn内置的糖尿病数据集
from sklearn.linear_model import LinearRegression           # 线性回归算法模型
from sklearn.model_selection import train_test_split        # 训练集和测试集分割函数
from sklearn.metrics import mean_squared_error, r2_score    # 模型性能评估函数
import matplotlib.pyplot as plt
import numpy as np
import os 

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']  # 多种中文字体备选
plt.rcParams['axes.unicode_minus'] = False    # 正确显示负号

print("=" * 60)
print("实验一：线性回归预测糖尿病进展")
print("=" * 60)

# 加载数据
print("正在加载糖尿病数据集...")
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target
feature_names = diabetes.feature_names
"""
    :Attribute Information:
      - age     age in years
      - sex
      - bmi     body mass index
      - bp      average blood pressure
      - s1      tc, total serum cholesterol
      - s2      ldl, low-density lipoproteins
      - s3      hdl, high-density lipoproteins
      - s4      tch, total cholesterol / HDL
      - s5      ltg, possibly log of serum triglycerides level
      - s6      glu, blood sugar level
    :属性信息:
      - age     年龄
      - sex     性
      - bmi     身体质量指数
      - bp      平均血压
      - s1      tc, 血清总胆固醇
      - s2      ldl, 低密度脂蛋白
      - s3      hdl, 高密度脂蛋白
      - s4      tch, 总胆固醇/高密度脂蛋白
      - s5      ltg, 可能是血清甘油三酯水平的对数
      - s6      glu, 血糖水平
"""

# 创建中英文特征名称映射字典
feature_name_mapping = {
    'age': '年龄',
    'sex': '性别',
    'bmi': '身体质量指数',
    'bp': '平均血压',
    's1': '血清总胆固醇',
    's2': '低密度脂蛋白',
    's3': '高密度脂蛋白',
    's4': '总胆固醇/高密度脂蛋白',
    's5': '血清甘油三酯对数',
    's6': '血糖水平'
}

# 创建中文特征名称列表
chinese_feature_names = [feature_name_mapping[name] for name in feature_names]

print(f"数据集形状: {X.shape}")
print(f"特征数量: {len(feature_names)}")
print(f"特征名称(英文): {', '.join(feature_names)}")
print(f"特征名称(中文): {', '.join(chinese_feature_names)}")

# 划分训练集和测试集
X_train , X_test , y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
) # 参数test_size设为0.2，即测试集占比为20%

print(f"训练集样本数: {X_train.shape[0]}")
print(f"测试集样本数: {X_test.shape[0]}")

# 创建并训练模型
print("训练线性回归模型中...")
reg = LinearRegression()
reg.fit(X_train, y_train)

print("训练完成！")

# 预测
y_pred = reg.predict(X_test)

# 评估模型
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("=" * 60)
print("线性回归模型性能评估")
print("=" * 60)
print(f"模型系数 (coef_): {reg.coef_}")
print(f"模型截距 (intercept_): {reg.intercept_}")
print(f"均方误差 (MSE): {mse:.4f}")
print(f"决定系数 (R²): {r2:.4f}")

# 打印特征重要性（使用中英文对照）
print("\n特征重要性:")
for i, (en_name, cn_name, coef) in enumerate(zip(feature_names, chinese_feature_names, reg.coef_)):
    print(f"{en_name}({cn_name}): {coef:.4f}")

# 输出特征重要性排序（使用中文名称）
print("\n特征重要性排序（按绝对值降序）:")
sorted_indices = np.argsort(np.abs(reg.coef_))[::-1] # 按系数绝对值降序排列
for i, idx in enumerate(sorted_indices):
    print(f"{i+1}. {chinese_feature_names[idx]}({feature_names[idx]}): {reg.coef_[idx]:.4f}")

print("=" * 60)

# 可视化预测结果
plt.figure(figsize=(15, 10))

# 1. 实际值 vs 预测值散点图
plt.subplot(2, 2, 1)
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

# 2. 残差图分析
plt.subplot(2, 2, 2)
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel("预测值")
plt.ylabel("残差 (实际值 - 预测值)")
plt.title("残差分析图")
plt.grid(alpha=0.3)

# 3. 特征与目标值关系图 (选取两个最重要特征)
top_features = np.argsort(np.abs(reg.coef_))[-2:]
for i, feature_idx in enumerate(top_features):
    plt.subplot(2, 2, 3 + i)
    plt.scatter(X_test[:, feature_idx], y_test, alpha=0.5, label='实际值')
    plt.scatter(X_test[:, feature_idx], y_pred, alpha=0.5, label='预测值')
    # 使用中文特征名称
    plt.xlabel(chinese_feature_names[feature_idx])
    plt.ylabel("目标值")
    plt.legend()
    plt.title(f"{chinese_feature_names[feature_idx]}与目标值关系")

plt.tight_layout()

# 4. 保存结果为png图片
save_dir = "img"
filename = "diabetes_prediction_results.png"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, filename)
plt.savefig(save_path, dpi=300, bbox_inches='tight')

# 5. 显示可视化结果
plt.show()

print(f"可视化结果已保存为 '{save_path}'")
print("=" * 60)
print("实验一顺利完成!")
print("=" * 60)