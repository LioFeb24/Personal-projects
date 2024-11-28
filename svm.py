from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# 加载数据集
digits = load_digits()
X_data = np.array(digits.data)
y_data = np.array(digits.target)

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=1)


# 自定义 SVM 多分类类
class SVM:
    def __init__(self, learning_rate=0.01, lambda_param=0.01, n_iters=1000):
        self.learning_rate = learning_rate  # 学习率
        self.lambda_param = lambda_param  # 正则化参数
        self.n_iters = n_iters  # 迭代次数
        self.models = []  # 存储多个 SVM 模型

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # 为每一个类别训练一个 SVM
        for c in range(10):
            y_binary = np.where(y == c, 1, -1)  # 将类别c标记为1，其他类别标记为-1
            w = np.zeros(n_features)  # 权重初始化
            b = 0  # 偏置初始化
            # 训练过程
            for _ in range(self.n_iters):
                for idx, x_i in enumerate(X):
                    if y_binary[idx] * (np.dot(x_i, w) - b) >= 1:
                        w -= self.learning_rate * (2 * self.lambda_param * w)  # 正则化项
                    else:
                        w -= self.learning_rate * (2 * self.lambda_param * w - np.dot(x_i, y_binary[idx]))
                        b -= self.learning_rate * y_binary[idx]

                        # 存储模型 (权重和偏置)
            self.models.append((w, b))

    def predict(self, X):
        # 进行多分类预测
        predictions = np.zeros((X.shape[0], len(self.models)))
        for idx, (w, b) in enumerate(self.models):
            predictions[:, idx] = np.dot(X, w) - b  # 记录每个模型的预测结果

        return np.argmax(predictions, axis=1)  # 返回具有最高预测值的类别


# 初始化 SVM 多分类模型
svm_model = SVM()

# 训练模型
svm_model.fit(x_train, y_train)

# 预测测试集
y_pred = svm_model.predict(x_test)
l = len(y_pred)
for i in range(l):
    print(f'预测值:{y_pred[i]}\n真实值:{y_test[i]}\n')
# 评估准确率
accuracy = accuracy_score(y_test, y_pred)
print("模型准确率: {:.2f}%".format(accuracy * 100))

# 可视化部分测试样本的预测结果
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(x_test[i].reshape(8, 8), cmap='gray')
    ax.set_title(f"pred: {y_pred[i]}, infact: {y_test[i]}")
    ax.axis('off')
plt.show()