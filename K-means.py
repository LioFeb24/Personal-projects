import random
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

# 数据加载
iris = load_iris()
X_data = iris.data
y_data = iris.target
x_train, x_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=1)


class modle:
    def fit(self, X, iteration=10):
        '''
        X:训练集
        iteration:迭代次数
        '''
        heart = []  # 存储质心
        for i in range(3):
            c = random.choice(X).tolist()
            heart.append(c)
        heart = np.array(heart)

        def class_cluster(X, heart):
            d_1_items, d_2_items, d_3_items = [], [], []
            for _ in X:
                # 计算该点到三个质心的距离
                this_d1 = (pow((_[0] - heart[0][0]), 2) + pow((_[1] - heart[0][1]), 2) + pow((_[2] - heart[0][2]),
                                                                                             2) + pow(
                    (_[3] - heart[0][3]), 2))
                this_d2 = (pow((_[0] - heart[1][0]), 2) + pow((_[1] - heart[1][1]), 2) + pow((_[2] - heart[1][2]),
                                                                                             2) + pow(
                    (_[3] - heart[1][3]), 2))
                this_d3 = (pow((_[0] - heart[2][0]), 2) + pow((_[1] - heart[2][1]), 2) + pow((_[2] - heart[2][2]),
                                                                                             2) + pow(
                    (_[3] - heart[2][3]), 2))
                # 判断所属的簇
                if this_d1 < this_d2 and this_d1 < this_d3:
                    d_1_items.append(_.tolist())
                if this_d2 < this_d1 and this_d2 < this_d3:
                    d_2_items.append(_.tolist())
                if this_d3 < this_d2 and this_d3 < this_d1:
                    d_3_items.append(_.tolist())
            return d_1_items, d_2_items, d_3_items

        def update_heart(d_1_items, d_2_items, d_3_items):
            # 更新质心
            d1_v = np.mean(d_1_items, axis=0)
            d2_v = np.mean(d_2_items, axis=0)
            d3_v = np.mean(d_3_items, axis=0)
            return np.array([d1_v, d2_v, d3_v])

        def sum_d(d_1_items, d_2_items, d_3_items, heart):
            d_1, d_2, d_3 = 0, 0, 0
            for _ in d_1_items:
                d_1 += (pow((_[0] - heart[0][0]), 2) + pow((_[1] - heart[0][1]), 2) + pow((_[2] - heart[0][2]),
                                                                                          2) + pow((_[3] - heart[0][3]),
                                                                                                   2))
            for _ in d_2_items:
                d_2 += (pow((_[0] - heart[1][0]), 2) + pow((_[1] - heart[1][1]), 2) + pow((_[2] - heart[1][2]),
                                                                                          2) + pow((_[3] - heart[1][3]),
                                                                                                   2))
            for _ in d_3_items:
                d_3 += (pow((_[0] - heart[2][0]), 2) + pow((_[1] - heart[2][1]), 2) + pow((_[2] - heart[2][2]),
                                                                                          2) + pow((_[3] - heart[2][3]),
                                                                                                   2))
            return d_1, d_2, d_3

        loss = []
        for i in range(1, iteration + 1):
            print(f'==================================round={i}=start==================================')
            d_1_items, d_2_items, d_3_items = class_cluster(X, heart)
            heart = update_heart(d_1_items, d_2_items, d_3_items)
            d1, d2, d3 = sum_d(d_1_items, d_2_items, d_3_items, heart)
            tot = d1 + d2 + d3
            print(f'迭代{i}次后三簇内距离和分别是:{d1},{d2},{d3}\n其和为:{tot}')
            loss.append(tot)
            self.sca(np.array(d_1_items), np.array(d_2_items), np.array(d_3_items), heart)
            tit = '迭代' + str(i) + '次后簇内距离和如下'
            plt.plot(loss)
            plt.title(tit)
            plt.xlabel('迭代次数')
            plt.ylabel('簇内距离和')
            plt.show()
            print(f'==================================round={i}=end==================================')

    def sca(self, cluster1, cluster2, cluster3, heart):
        # 使用PCA将数据降维到二维
        pca = PCA(n_components=2)
        all_data = np.vstack((cluster1, cluster2, cluster3))
        reduced_data = pca.fit_transform(all_data)

        # 创建散点图
        plt.figure(figsize=(8, 6))
        plt.scatter(reduced_data[:len(cluster1), 0], reduced_data[:len(cluster1), 1], c='red', label='簇 1')
        plt.scatter(reduced_data[len(cluster1):len(cluster1) + len(cluster2), 0],
                    reduced_data[len(cluster1):len(cluster1) + len(cluster2), 1], c='blue', label='簇 2')
        plt.scatter(reduced_data[len(cluster1) + len(cluster2):, 0], reduced_data[len(cluster1) + len(cluster2):, 1],
                    c='green', label='簇 3')

        # 添加质心
        centroids_reduced = pca.transform(heart)  # 降维质心
        plt.scatter(centroids_reduced[:, 0], centroids_reduced[:, 1], c='yellow', s=200, marker='X', label='质心')

        plt.title('聚类结果可视化')
        plt.legend()
        plt.grid()
        plt.show()


modle = modle()
modle.fit(x_train, 12)