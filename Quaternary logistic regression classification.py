from sklearn.datasets import load_iris
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split  
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  

# 数据加载  
iris = load_iris()
X_data = iris.data[iris.target != 2]
y_data = iris.target[iris.target != 2]
x_train, x_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2)  

class linear:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = np.zeros(4)  # 初始化权重为四个特征
        self.bias = 0
        self.loss_l = []  # 用于记录损失值

    def sigmoid(self, z):
        """Sigmoid激活函数"""
        return 1 / (1 + np.exp(-z))

    def fit(self, x, y):
        """训练模型，使用梯度下降法计算权重和偏置"""
        n_samples = len(y)

        for iteration in range(self.n_iterations):
            # 计算预测值（概率）
            z = np.dot(x, self.weights) + self.bias  # 线性组合
            y_predicted = self.sigmoid(z)  # Sigmoid 激活函数
            
            # 计算梯度
            gradient_weights = np.dot(x.T, (y_predicted - y)) / n_samples  # 权重的梯度
            bias_gradient = np.sum(y_predicted - y) / n_samples  # 偏置的梯度

            # 更新权重和偏置
            self.weights -= self.learning_rate * gradient_weights  # 更新权重
            self.bias -= self.learning_rate * bias_gradient  # 更新偏置

            # 计算交叉熵损失并记录
            loss = -np.mean(y * np.log(y_predicted) + (1 - y) * np.log(1 - y_predicted))  # 交叉熵损失
            self.loss_l.append(loss)  # 记录损失

            # 每100次迭代，展示一次损失函数曲线
            if (iteration + 1) % 100 == 0:
                self.plot_loss_curve(iteration + 1)
            plt.show()  # 显示最终图形

    def predict(self, x):
        """进行预测，输出概率值"""
        z = np.dot(x, self.weights) + self.bias
        return self.sigmoid(z)  # 返回概率值

    def predict_class(self, x,):
        """基于阈值进行分类预测"""
        p = self.predict(x)
        if p >= 0.5:
            c = 1
        else:
            c = 0
        return c

    def plot_loss_curve(self, iteration):
        """绘制损失函数曲线"""
        plt.plot(range(1, iteration + 1), self.loss_l, label='Loss函数')
        plt.xlabel('迭代次数')
        plt.ylabel('Loss')
        plt.savefig(f'迭代次数{iteration}.png')


model = linear()  
model.fit(x=x_train, y=y_train)  
print('w:',model.weights,'b:',model.bias,'loss:',model.loss_l)
l = len(y_test)
score = 0
for i in range(l):
    概率 = model.predict(x_test[i])
    print(f'特征为:\n{x_test[i]}\n类别为{y_test[i]}\ny帽:\n{概率}')
    if 概率 >= 0.5:
        c = 1
    else:
        c = 0
    print(f'判断分类结果:{c}\n')

    if c == y_test[i]:
        score += 1
print('得分',score/l*100,'%')
    