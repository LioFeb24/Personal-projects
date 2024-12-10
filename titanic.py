import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
plt.rcParams['font.family'] = 'SimHei'
# 读取数据
data = pd.read_csv('./train.csv')

# 数据清洗
data_clear = data.loc[:, ['Survived', 'Age', 'Sex', 'Pclass', 'Fare', 'Embarked']]
data_clear['Sex'] = data_clear['Sex'].replace({'male': 10, 'female': -10})
data_clear['Embarked'] = data_clear['Embarked'].replace({'S': -10, 'C': 0, 'Q': 10})
data_clear['Age'].fillna(data_clear['Age'].median(), inplace=True)
data_clear['Embarked'].fillna(data_clear['Embarked'].median(), inplace=True)

id = np.array(data['PassengerId']).astype(int)

# 分离特征和标签
data_no_survived = data_clear.drop(['Survived'], axis=1)
X_data = np.array(data_no_survived)
y_data = np.array(data_clear['Survived'])

# 对 'Age' 和 'Fare' 特征进行标准化
scaler = StandardScaler()
X_data[:, 1] = scaler.fit_transform(X_data[:, 1].reshape(-1, 1)).reshape(-1)  # 'Age' 列
X_data[:, 4] = scaler.fit_transform(X_data[:, 4].reshape(-1, 1)).reshape(-1)  # 'Fare' 列

del data
# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2)

class LinearModel:
    def __init__(self, learning_rate=0.001, n_iterations=20000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = np.zeros(X_data.shape[1])
        self.bias = 0
        self.loss_l = []

        # Adam 优化器参数
        self.m = np.zeros(X_data.shape[1])
        self.v = np.zeros(X_data.shape[1])
        self.bias_m = 0
        self.bias_v = 0
        self.t = 0

    def sigmoid(self, z):
        z = np.clip(z, -709, 709)
        return 1 / (1 + np.exp(-z))

    def fit(self, x, y):
        n_samples = len(y)

        for iteration in range(self.n_iterations):
            self.t += 1  #增加时间步长
            z = np.dot(x, self.weights) + self.bias
            y_predicted = self.sigmoid(z)

            if iteration % 100 == 0:
                print(f'步数{iteration}:  -- w: {self.weights}, b: {self.bias}')

            gradient_weights = np.dot(x.T, (y_predicted - y)) / n_samples
            bias_gradient = np.sum(y_predicted - y) / n_samples

            # 更新Adam
            self.m = 0.9 * self.m + 0.1 * gradient_weights
            self.v = 0.999 * self.v + 0.001 * gradient_weights ** 2
            self.bias_m = 0.9 * self.bias_m + 0.1 * bias_gradient
            self.bias_v = 0.999 * self.bias_v + 0.001 * bias_gradient ** 2

            m_hat = self.m / (1 - 0.9 ** self.t)  # 改正 Bias
            v_hat = self.v / (1 - 0.999 ** self.t)
            bias_m_hat = self.bias_m / (1 - 0.9 ** self.t)
            bias_v_hat = self.bias_v / (1 - 0.999 ** self.t)

            self.weights -= self.learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)  # 更新权重
            self.bias -= self.learning_rate * bias_m_hat / (np.sqrt(bias_v_hat) + 1e-8)  # 更新截距
            # 记录 loss
            loss = -np.mean(y * np.log(y_predicted + 1e-15) + (1 - y) * np.log(1 - y_predicted + 1e-15))
            self.loss_l.append(loss)


    def predict(self, x):
        z = np.dot(x, self.weights) + self.bias
        return self.sigmoid(z)

    def predict_class(self, x):
        p = self.predict(x)
        return (p >= 0.5).astype(int)

# 创建并训练模型
model = LinearModel()
model.fit(x=X_data, y=y_data)

# 载入测试集
evaluate = pd.read_csv('./evaluate.csv')
data_test = evaluate.loc[:, ['Age', 'Sex', 'Pclass', 'Fare', 'Embarked']]
data_test['Sex'] = data_test['Sex'].replace({'male': 10, 'female': -10})
data_test['Embarked'] = data_test['Embarked'].replace({'S': -10, 'C': 0, 'Q': 10})
data_test['Age'].fillna(data_test['Age'].median(), inplace=True)
data_test['Embarked'].fillna(data_test['Embarked'].median(), inplace=True)
data_test_id = np.array(evaluate['PassengerId']).astype(int)
X_data_test = np.array(data_test)
def test_evaluation(x_test,id):
    '''分类evaluate.csv'''
    output = []
    l = len(x_test)
    for i in range(l):
        c = model.predict_class(x_test[i])
        print(id[i],c)
        output.append(c)
    csv = pd.DataFrame(zip(id, output), columns=['PassengerId', 'Survived'])
    path = 'evaluate验证结果.csv'
    csv.to_csv(path, index=False)
test_evaluation(X_data_test,data_test_id)

#以下为测试用
def test(X_test,y_test):
    '''测试阶段'''
    output = []
    y_data = y_test
    l = len(y_data)
    score = 0
    X_data = X_test
    for i in range(l):
        p = model.predict(X_data[i].reshape(1, -1))
        print(f'特征为:\n{X_data[i]}\n类别为{y_data[i]}\n趋向1的概率:\n{p[0]}')
        c = model.predict_class(X_data[i].reshape(1, -1))[0]
        output.append(c)
        print(f'判断分类结果:{c}\n')
        if c == y_data[i]:
            score += 1
        s = round(score / l *100,2)
    print(f'得分{s}')
    print('w:', model.weights, 'b:', model.bias, 'loss:', model.loss_l[-1])
test(X_test=x_test,y_test=y_test)

#模型关系可视化：
# 使用训练集进行预测
y_train_pred = model.predict(x_train)
y_train_pred_class = np.round(y_train_pred)  # 将概率转化为类别

# 可视化训练损失
plt.figure(figsize=(10, 6))
plt.plot(model.loss_l)
plt.title('训练损失变化情况')
plt.xlabel('迭代次数（每100次）')
plt.ylabel('损失')
plt.grid()
plt.show()

# 绘制特征重要性
# 特征对分类结果的影响(权重)
feature_names = ['Age', 'Sex', 'Pclass', 'Fare', 'Embarked']
coefs = model.weights

plt.figure(figsize=(10, 6))
plt.barh(feature_names, coefs)
plt.title('特征重要性（回归系数）')
plt.xlabel('回归系数')
plt.ylabel('特征')
plt.grid()
plt.show()