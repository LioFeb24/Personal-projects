import numpy as np  
from pandas import read_csv
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import mean_squared_error, r2_score  

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  

boston = read_csv('BostonHousing.csv')

def clear(data)-> None:
    columns = data.columns.tolist()
    for column in columns:
        if not column == 'chas':
            # 计算数据的四分位数
            Q1_tot = boston[column].quantile(0.25)
            Q3_tot = boston[column].quantile(0.75)
            IQR_tot = Q3_tot - Q1_tot
            # 根据四分位数的范围识别异常值
            outliers = ((boston[column] < (Q1_tot - 1.5 * IQR_tot)) | (boston[column] > (Q3_tot + 1.5 * IQR_tot)))
            # 用平均值替换异常值
            median = boston[column].median()
            boston.loc[outliers, column] = median
clear(boston)

data_x = np.array(boston['rm'])
data_y = np.array(boston['medv'])
#划分测试集和训练集
x_train,x_test,y_train,y_test = train_test_split(data_x,data_y,test_size=0.2)
#print('x_train(房间数量):\n',x_train,"\nx_test(房间数量):\n",x_test,"\ny_train(房子价格):\n",y_train,"\ny_test(房子价格):\n",y_test)

# 梯度下降法实现的线性回归模型  
class line_model:  
    def __init__(self, learning_rate=0.01, n_iterations=1000):  
        self.learning_rate = learning_rate  
        self.n_iterations = n_iterations  
        self.weight = 0  
        self.bias = 0  

    def fit(self, x, y):  
        """训练模型，使用梯度下降法计算权重和偏置"""  
        n_samples = len(y)  
        for _ in range(self.n_iterations):  #下降1000次 步幅为 0.01
            y_predicted = self.predict(x)  
            # 计算梯度  
            # 初始化 weight 和 bias
            weight = []  
            bias = 0  
            # 计算 weight 和 bias 的具体实现  
            for i in range(n_samples):  

                x_i = x[i]  
                y_i = y[i]  
                y_predicted_i = y_predicted[i]  

                # 计算 weight
                gradient = -2 / n_samples * (x_i * (y_i - y_predicted_i))  
                weight.append(gradient)  

                # 计算 bias
                bias += y_i - y_predicted_i  


            # 最后计算 bias  
            bias = -2 / n_samples * bias  
            
            # 更新参数  
            self.weight -= self.learning_rate * sum(weight)  # 更新权重  
            self.bias -= self.learning_rate * bias  # 更新偏置  

    def predict(self, x):  
        """定义预测函数"""  
        d = []
        l = len(x)
        for i in range(l):
            d.append(self.weight * x[i] + self.bias)
        return d

model = line_model()  
model.fit(x=x_train, y=y_train)  

# 预测训练集和测试集的值  
y_train_pred = model.predict(x_train)  
y_test_pred = model.predict(x_test)  

# 计算训练集和测试集的得分  
mse_train = mean_squared_error(y_train, y_train_pred)  # 训练集的均方误差  
mse_test = mean_squared_error(y_test, y_test_pred)      # 测试集的均方误差  
r2_train = r2_score(y_train, y_train_pred)              # 训练集的 R² 得分  
r2_test = r2_score(y_test, y_test_pred)                # 测试集的 R² 得分  

# 输出得分  
print(f"训练集 MSE: {mse_train:.4f}")  
print(f"测试集 MSE: {mse_test:.4f}")  
print(f"训练集 R²: {r2_train:.4f}")  
print(f"测试集 R²: {r2_test:.4f}")  

# 绘制散点图和拟合直线  
plt.scatter(x_train, y_train, color='blue', label='训练数据')  # 绘制训练数据的散点图  
plt.plot(x_train, y_train_pred, color='red', label='拟合线')  # 绘制拟合的直线  

plt.legend()  
plt.show()