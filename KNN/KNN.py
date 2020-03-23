import numpy as np
import pandas as pd
import time
from collections import Counter

# Mnist
# KNN (100样例, k=20)
# spend time: 419s
# acc: 0.969


def loadData(filaName):
    data = pd.read_csv(filaName, header=None)

    data = data.values

    y_label = data[:, 0]

    x_data = np.mat(data[:, 1:])

    return x_data / 255, y_label


class KNN(object):

    def __init__(self, k):
        self.k = k

    def train(self, X, Y):
        self.X = X
        self.Y = Y


    def distance(self, xi, xj, mode='L2'):
        if mode == 'L2':
            return np.sqrt(np.sum(np.square(xi-xj)))

    def find(self, X):
        ds = []
        for x in self.X:
            ds.append(self.distance(x, X))
        
        k_nearst = np.argsort(np.array(ds))[:self.k]

        # 拿到k近邻的标签
        res = np.array(self.Y)[k_nearst]

        # a = [1,2,3,4,4,4,4,4,4,4,1,1,1,23,2,2,2,3,1]
        # b=Counter(a)
        # b.most_common(1)
        # out:[(4,7)] 4是元素，7是出现次数
        belonging = Counter(res).most_common(1)
        print(belonging)
        belonging = belonging[0][0]
        return belonging

def test_knn(X, Y, test_X, test_Y, k):
    knn = KNN(k)

    knn.train(X, Y)
    cnt = 0
    for i in range(100):
        pred_y = knn.find(test_X[i])
        if pred_y == test_Y[i]:
            cnt += 1
    
    # print('acc is {}'.format(cnt / X.shape[0])) 太慢了
    print('acc is {}'.format(cnt / 100))

if __name__ == '__main__':
    
    startTime = time.time()
    train_data_path = './datasets/Mnist/mnist_train.csv'
    test_data_path = './datasets/Mnist/mnist_test.csv'

    x_train, y_train = loadData(train_data_path)
    x_test, y_test = loadData(test_data_path)

    test_knn(x_train, y_train, x_test, y_test, k=20)


    endTime = time.time()

    print('spend time is :{}'.format(endTime - startTime))

