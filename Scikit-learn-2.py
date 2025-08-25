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
import matplotlib.pyplot as plt
from sklearn import metrics, svm
from sklearn.metrics import ConfusionMatrixDisplay      # 混淆矩阵
from sklearn.metrics import classification_report       # 分类报告
from sklearn.metrics import roc_curve, auc              # ROC曲线
from sklearn.preprocessing import label_binarize
import numpy as np

# 1.加载鸢尾花数据集
datasets_iris = load_iris()
X = datasets_iris.data
y = datasets_iris.target
class_names = datasets_iris.target_names
feature_names = datasets_iris.feature_names

# 2.划分训练集和测试集（80%训练，20%测试）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# 3.训练随机森林分类器，设置n_estimators=100
clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
clf.fit(X_train, y_train)

# 4.在测试集上预测
y_pred = clf.predict(X_test)
print("预测结果:", y_pred)

# 5.1 使用混淆矩阵评估模型
np.set_printoptions(precision=2)
titles_options = [
    ("Confusion matrix, without normalization", None),
    ("Normalized confusion matrix", "true",)
]
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
for i, (title, normalize) in enumerate(titles_options):
    disp = ConfusionMatrixDisplay.from_estimator(
        clf,
        X_test,
        y_test,
        display_labels=class_names,
        cmap=plt.cm.Blues,
        normalize=normalize,
        ax=axs[i]  # 指定子图位置
    )
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.tight_layout()
plt.show()

# 5.2 使用分类报告评估模型
print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=class_names))

# 6. 绘制ROC曲线（多分类）
# 获取预测概率
y_score = clf.predict_proba(X_test)

# 将标签二值化
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
n_classes = y_test_bin.shape[1]

# 计算每个类的ROC曲线和AUC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# 绘制所有ROC曲线
plt.figure(figsize=(8, 6))
colors = ['blue', 'red', 'green']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(class_names[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC Curve')
plt.legend(loc="lower right")
plt.show()