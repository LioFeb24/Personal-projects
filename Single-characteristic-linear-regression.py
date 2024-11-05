from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

boston = read_csv('BostonHousing.csv')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

# 处理异常值 中位数替代
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
#画散点图
def scatter_(data)->None:
    colums_info = {
        'crim':"人均城镇犯罪率",
        'zn':"25,000平方英尺以上土地的住宅用地比例",
        'indus':"每个城镇非零售业务英亩的比例",
        'chas':"查尔斯河虚拟变量（如果束缚河，则为1；否则为0）",
        'nox':"氧化氮浓度（百万分之一）",
        'rm':"每个住宅的平均房间数",
        'age':"1940年之前建造的自有住房的比例",
        'dis':"到五个波士顿就业中心的加权距离",
        'rad':"径向公路的可达性指数",
        'tax':"每10,000美元的全值财产税率",
        'ptratio':"各镇师生比例",
        'b':"1000（Bk-0.63）^ 2，其中Bk是按城镇划分的黑人比例",
        'lstat':"人口状况降低百分比",
        'medv':"自有住房的中位价格（以$ 1000为单位）"
        }
    for key,value in colums_info.items():
        if not key == 'medv':  
            plt.scatter(x=data[key] ,y= data['medv'])
            plt.xlabel(value)
            plt.ylabel('房价')
            plt.show()
#scatter_(boston)

class line_model():
    def __init__(self):
        self.weight = None
        self.bias = None
    
    def fit(self,x,y):#训练模型 求weight以及bias
        l = len(x)
        # 计算各项求和
        sum_xiyi = 0  
        sum_xi = 0     
        sum_yi = 0     
        sum_xi_squared = 0  
    
        for i in range(l):
            sum_xiyi += x.iloc[i] * y.iloc[i]
            sum_xi += x.iloc[i]
            sum_yi += y.iloc[i]
            sum_xi_squared += x.iloc[i] ** 2

        # 计算斜率 weight
        son = sum_xiyi - (sum_xi * sum_yi) 
        mother = sum_xi_squared - (sum_xi ** 2)
        self.weight = son / mother
        # 计算截距 bias
        self.bias = (sum_yi / l) - self.weight * (sum_xi / l)
        
    def predict(self,x)->float: # 定义预测函数
        y_hat = self.weight*x + self.bias
        return y_hat # 返回结果


#划分测试集和训练集
x_train,x_test,y_train,y_test = train_test_split(boston['rm'],boston['medv'],test_size=0.2)
#print('x_train(房间数量):\n',x_train,"\nx_test(房间数量):\n",x_test,"\ny_train(房子价格):\n",y_train,"\ny_test(房子价格):\n",y_test)
model = line_model()
model.fit(x = x_train,y = y_train)

# 预测训练集和测试集的值
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

# 计算训练集和测试集的得分
mse_train = mean_squared_error(y_train, y_train_pred)  # 训练集的均方误差
mse_test = mean_squared_error(y_test, y_test_pred)      # 测试集的均方误差
r2_train = r2_score(y_train, y_train_pred)              # 训练集的 R² 得分
r2_test = r2_score(y_test, y_test_pred)                # 测试集的 R² 得分
# 预测训练集的值
y_train_pred = model.predict(x_train)
# 绘制散点图和拟合直线
plt.scatter(x_train, y_train, color='blue', label='训练数据')  # 绘制训练数据的散点图
plt.plot(x_train, y_train_pred, color='red', label='拟合线')  # 绘制拟合的直线
plt.xlabel('房间数量')
plt.ylabel('房价')
plt.legend()
plt.title('房间数量与房价的关系及模型拟合')
plt.show()
# 输出得分
print(f"训练集 MSE: {mse_train:.4f}")
print(f"测试集 MSE: {mse_test:.4f}")
print(f"训练集 R²: {r2_train:.4f}")
print(f"测试集 R²: {r2_test:.4f}")

