from sklearn.datasets import load_iris
import pandas as pd
import collections

ordata = load_iris()
frist_c = ordata.keys()
data = pd.DataFrame(ordata.data,columns=ordata.feature_names)
data['target'] = ordata.target
print(data)

# 120训练集，30测试集
train = data.sample(n=120)
test = data.sample(n=len(data) - 120)


# 划分训练集和测试集
def classbyme(train):
    l = len(train)
    d = []
    target = []
    for i in range(l):
        line = [float(train.iloc[i]['sepal length (cm)']), float(train.iloc[i]['sepal width (cm)']),
                float(train.iloc[i]['petal length (cm)']), float(train.iloc[i]['petal width (cm)'])]
        d.append(line)
        target.append(int(train.iloc[i]['target']))
    return d, target


x_train_me, y_train_me = classbyme(train)
x_test_me, y_test_me = classbyme(test)


# 计算距离并且分类
def reclass(x_train, y_train, x_test, k):
    '''
    x_train:训练集的X轴(属性),
    y_train:训练集的Y轴(分类结果),
    x_test:所分类测试单元的X轴(属性)
    k:取邻居数量
    '''
    d = []
    l = len(x_train)
    for i in range(l):
        a1, a2, a3, a4 = x_test[0], x_test[1], x_test[2], x_test[3]
        b1, b2, b3, b4 = x_train_me[i][0], x_train_me[i][1], x_train_me[i][2], x_train_me[i][3]
        jl = pow((pow((a1 - b1), 2.0) + pow((a2 - b2), 2.0) + pow((a3 - b3), 2.0) + pow((a4 - b4), 2.0)), 1 / 2)
        d.append([jl, y_train[i]])
    sorted_data = sorted(d, key=lambda x: x[0])[:k]

    # 提取第二个元素
    classindex = [item[1] for item in sorted_data]

    # 计算众数(分类类别)
    counter = collections.Counter(classindex)
    mode = counter.most_common(1)
    return mode[0][0]


# 计算得分:
l = len(y_test_me)
sus = 0
for i in range(l):
    if reclass(x_train=x_test_me, y_train=y_train_me, x_test=x_test_me[i], k=3) == y_test_me[i]:
        sus += 1

print(f'得分为{sus / l}')