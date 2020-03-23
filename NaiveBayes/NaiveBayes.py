import numpy as np
import pandas as pd
import time
from collections import Counter

# Mnist
# spent time: 372s
# acc: 0.60

def loadData(fileName):
    data = pd.read_csv(fileName, header=None)
    data = data.values
    Y = np.array(data[:, 0])
    X = np.array(data[:, 1:])

    # 特征需要转化
    X[X < 128] = 0
    X[X >= 128] = 1

    return X, Y

# 计算先验 和 条件概率
def pre_cacu(X, Y, lam=1):
    y_class = 10
    S_j = 2
    num_features = 784

    py = np.zeros((y_class, 1))
    for y in Y:
        py[y] += 1
    py = py / len(Y)
    # print(py)
    print('-------')
    pxy = np.zeros((y_class, num_features, 2))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            pxy[Y[i]][j][X[i][j]] += 1
    print('-------')
    for i in range(y_class):
        for j in range(num_features):
            pxy[i][j][0] = pxy[i][j][0] + lam / py[i] + S_j * lam
            pxy[i][j][1] = pxy[i][j][1] + lam / py[i] + S_j * lam
    
    # print(pxy[:1])
    # return py, pxy
    return np.log(py), np.log(pxy)

# 预测
def testNByes(X, Y, py, pxy):
    y_class = 10
    num_features = 784
    cnt = 0
    for index in range(X.shape[0]):
        p = np.zeros((y_class, 1))
        for i in range(y_class):
            for j in range(num_features):
                p[i] += pxy[i][j][X[index][j]]
            p[i] += py[i]
        print(index)
        pred_y = np.argmax(p)

        if pred_y == Y[index]:
            cnt += 1
    print('acc is {}'.format(cnt / X.shape[0]))

def test():
    startTime = time.time()
    train_data_path = './datasets/Mnist/mnist_train.csv'
    test_data_path = './datasets/Mnist/mnist_test.csv'
    # 读取训练文件
    X_train, y_train = loadData(train_data_path)

    # 读取测试文件
    X_test, y_test = loadData(test_data_path)

    py, pxy = pre_cacu(X_train, y_train)

    testNByes(X_test, y_test, py, pxy)

    endTime = time.time()

    print('spent time is:{}'.format(endTime - startTime))

if __name__ == '__main__':
    test()
    