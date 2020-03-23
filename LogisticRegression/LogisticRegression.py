import pandas as pd
import numpy as np
import time

# Mnist
# spent time: 118s
# acc: 0.82

def loadData(fileName):
    data = pd.read_csv(fileName, header=None)

    # 增加一列
    data[785] = 1

    data = data.values

    Y = data[:, 0]
    X = data[:, 1:]

    Y[Y > 0] = 1

    return X, Y

def sigmoid(x):
    return 1 / (1 + np.exp(-1 * x))

def logisticRegression(X, Y, epochs):

    w = np.random.uniform(0, 1, (len(X[0]), 1))

    X = np.mat(X)
    Y = np.mat(Y)

    lr = 0.001

    for i in range(epochs):
        # w [785, 1], X [60000, 785]
        hx = sigmoid(X.dot(w))

        # 二分类损失
        # loss = -1 * (Y * np.log(hx) + (1-Y) * np.log(1-hx))
        # print('{} epoch loss:{}'.format(i, loss))

        # grad = X * (hx - Y)
        # print(hx.shape, Y.shape)
        w -= lr * X.T.dot(hx - Y.T)

    return w

def predict(X, w):
    hx = sigmoid(X.dot(w))
    hx[hx >= 0.5] = 1
    hx[hx < 0.5] = 0
    return hx

def test(X, Y, w):
    res = predict(X, w)
    acc = np.sum(Y == res) / len(Y)
    print('acc is:{}'.format(acc))

if __name__ == '__main__':
    
    startTime = time.time()
    train_data_path = './datasets/Mnist/mnist_train.csv'
    test_data_path = './datasets/Mnist/mnist_test.csv'
    # 读取训练文件
    X_train, y_train = loadData(train_data_path)
    # 读取测试文件
    X_test, y_test = loadData(test_data_path)
    w = logisticRegression(X_train, y_train, 200)
    test(X_test, y_test, w)
    endTime = time.time()
    print('spent time is:{}'.format(endTime - startTime))
