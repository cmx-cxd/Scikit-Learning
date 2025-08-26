# -*- coding: utf-8 -*-
# Scikit-learn-2.py
"""
#### 题目2: 随机森林分类鸢尾花（分类问题）
**任务描述**: 
使用鸢尾花数据集(Iris)训练一个随机森林分类器。  
**步骤**:  
1. 加载鸢尾花数据集。  
2. 划分训练集和测试集(80%训练, 20%测试)。  
3. 训练随机森林分类器(设置n_estimators=100)。  
4. 在测试集上进行预测。  
5. 使用混淆矩阵和分类报告(包括精确率、召回率、F1分数)评估模型。  
6. 绘制ROC曲线(注意: 这是一个多分类问题, ROC曲线需要为每个类别分别绘制)。  
"""

from sklearn.datasets import load_iris                  # 鸢尾花数据集
from sklearn.model_selection import train_test_split    # 训练集和测试集划分器
from sklearn.ensemble import RandomForestClassifier     # 随机森林分类器
from sklearn.metrics import ConfusionMatrixDisplay      # 混淆矩阵
from sklearn.metrics import classification_report       # 分类报告
from sklearn.metrics import roc_curve, auc              # ROC曲线
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("实验二：随机森林分类鸢尾花")
print("=" * 60)

# 1.加载鸢尾花数据集
print("正在加载鸢尾花数据集...")
datasets_iris = load_iris()
X = datasets_iris.data
y = datasets_iris.target
class_names_en = datasets_iris.target_names
"""
    class_names_en =[
        np.str_('setosa'),      # 山鸢尾
        np.str_('versicolor'),  # 变色鸢尾
        np.str_('virginica')    # 维吉尼亚鸢尾
    ]
"""
feature_names_en = datasets_iris.feature_names
"""
    feature_names_en = [
        "sepal length (cm)",    # 萼片长度（厘米）
        "sepal width (cm)",     # 萼片宽度（厘米）
        "petal length (cm)",    # 花瓣长度（厘米）
        "petal width (cm)",     # 花瓣宽度（厘米）
    ]
"""

# 创建中英文映射字典
class_name_mapping = {
    'setosa': '山鸢尾',
    'versicolor': '变色鸢尾',
    'virginica': '维吉尼亚鸢尾'
}

feature_name_mapping = {
    'sepal length (cm)': '萼片长度（厘米）',
    'sepal width (cm)': '萼片宽度（厘米）',
    'petal length (cm)': '花瓣长度（厘米）',
    'petal width (cm)': '花瓣宽度（厘米）'
}

# 创建中文名称列表
class_names_zh = [class_name_mapping[name] for name in class_names_en]
feature_names_zh = [feature_name_mapping[name] for name in feature_names_en]

print(f"数据集形状: {X.shape}")
print(f"类别数量: {len(class_names_en)}")
print(f"类别名称(英文): {', '.join(class_names_en)}")
print(f"类别名称(中文): {', '.join(class_names_zh)}")
print(f"特征数量: {len(feature_names_en)}")
print(f"特征名称(英文): {', '.join(feature_names_en)}")
print(f"特征名称(中文): {', '.join(feature_names_zh)}")

# 2.划分训练集和测试集（80%训练，20%测试）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

print(f"训练集样本数: {X_train.shape[0]}")
print(f"测试集样本数: {X_test.shape[0]}")

# 3.训练随机森林分类器，设置n_estimators=100
print("训练随机森林分类器中...")
clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
clf.fit(X_train, y_train)
print("训练完成！")

# 4.在测试集上进行预测
y_pred = clf.predict(X_test)
# 获取预测概率
y_pred_proba = clf.predict_proba(X_test)
print("预测结果:", y_pred)
print("预测概率:", y_pred_proba)

# 5.评估模型
accuracy = np.mean(y_pred == y_test)
print("=" * 60)
print("模型性能评估")
print("=" * 60)
print(f"准确率: {accuracy:.4f}")

# 5.1 使用混淆矩阵评估模型
print("\n混淆矩阵:")
fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay.from_estimator(
    clf, X_test, y_test,
    display_labels=class_names_zh,   # 使用中文类别名称
    cmap=plt.cm.Blues,
    ax=ax
)
disp.ax_.set_title("鸢尾花分类混淆矩阵")
plt.tight_layout()

# 保存混淆矩阵图片
save_dir = "img"
os.makedirs(save_dir, exist_ok=True)
confusion_matrix_path = os.path.join(save_dir, "iris_confusion_matrix.png")
plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"混淆矩阵已保存为 '{confusion_matrix_path}'")

# 5.2 使用分类报告评估模型
print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=class_names_zh))   # 使用中文类别名称

# 6. 绘制ROC曲线（多分类）
# 将标签二值化
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
n_classes = y_test_bin.shape[1]

# 计算每个类的ROC曲线和AUC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# 绘制所有ROC曲线
plt.figure(figsize=(10, 8))
colors = ['blue', 'red', 'green']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (AUC = {1:0.2f})'
             ''.format(class_names_zh[i], roc_auc[i]))   # 使用中文类别名称

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假正率 (False Positive Rate)')
plt.ylabel('真正率 (True Positive Rate)')
plt.title('多类别ROC曲线')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()

# 保存ROC曲线图片
roc_curve_path = os.path.join(save_dir, "iris_roc_curve.png")
plt.savefig(roc_curve_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"ROC曲线已保存为 '{roc_curve_path}'")

print("=" * 60)
print("实验二顺利完成!")
print("=" * 60)