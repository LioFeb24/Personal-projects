from sklearn.datasets import load_iris
import pandas as pd
import numpy as np  
import scipy.stats as stats  
import collections
from scipy.stats import norm


ordata = load_iris()
frist_c = ordata.keys()
data = pd.DataFrame(ordata.data,columns=ordata.feature_names)
data['target'] = ordata.target

# 120训练集，30测试集
train = data.sample(n=120)
test = data.sample(n=len(data) - 120)

# 划分训练集和测试集
def itemsclass(train):
    l = len(train)
    d = []
    target = []
    for i in range(l):
        line = [float(train.iloc[i]['sepal length (cm)']), float(train.iloc[i]['sepal width (cm)']),
                float(train.iloc[i]['petal length (cm)']), float(train.iloc[i]['petal width (cm)'])]
        d.append(line)
        target.append(int(train.iloc[i]['target']))
    return d, target
x_train, y_train = itemsclass(train)
x_test, y_test = itemsclass(test)

class GaussianNaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.means = []
        self.stds = []
        self.priors = []

        for c in self.classes:
            X_c = X[y == c]
            self.means.append(X_c.mean(axis=0))
            self.stds.append(X_c.std(axis=0))
            self.priors.append(len(X_c) / len(X))
    def predict(self, X):
        posteriors = []

        for i, c in enumerate(self.classes):
            prior = np.log(self.priors[i])
            likelihood = np.sum(np.log(norm.pdf(X, loc=self.means[i], scale=self.stds[i])), axis=1)
            posterior = prior + likelihood
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors, axis=0)]

X = np.array(x_train)
y = np.array(y_train)

model = GaussianNaiveBayes()
model.fit(X, y)

X_test = np.array(x_test)
predictions = model.predict(X_test)
score = 0
l = len(y_test)
for i in range(l):
    if predictions[i] == y_test[i]:
        score+=1
print(score/l)
