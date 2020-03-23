import numpy as np
import pandas as pd
import time

# Mnist
# spend time: 312s
# acc: 0.7904

def loadData(fileName):
    data = pd.read_csv(fileName, header=None)

    data = data.values

    y_label = data[:, 0]
    # 这里注意顺序
    y_label[y_label < 5] = -1
    y_label[y_label >= 5] = 1
    
    x_label = np.mat(data[:, 1:])

    return np.mat(x_label / 255), np.mat(y_label).T


def train_perceptron(X, Y, lr=0.001, iters=100):
    # 都初始化为 0
    w = np.zeros((X.shape[1], 1)) # [784, 1]
    b = 0 

    for iter in range(iters):
        cnt = 0
        for i, x in enumerate(X):
            y = Y[i]
            # x [1, 784] w[784, 1]
            if y * (x.dot(w) + b) <= 0:
                cnt += 1
                w += lr * np.multiply(y, x.T)
                b += lr * y
        print('{} iter acc is {}'.format(iter+1, 1 - cnt * 1.0 / X.shape[0]))
    return w, b

def test_perceptron(X, Y, w, b):
    pred_y = X.dot(w) + b
    pred_y[pred_y > 0] = int(1)
    pred_y[pred_y <= 0] = int(-1)
    correct = np.sum(pred_y == Y)
    acc = correct / X.shape[0]
    print(acc)

if __name__ == '__main__':

    startTime = time.time()

    train_data_path = './datasets/Mnist/mnist_train.csv'
    x_train, y_train = loadData(train_data_path)
    # print(x_train.shape) # (60000, 784)
    # print(y_train.shape) # (60000, 1)

    # w (784)
    # b (1)
    w, b = train_perceptron(x_train, y_train)

    test_data_path = './datasets/Mnist/mnist_test.csv'
    x_test, y_test = loadData(test_data_path)

    test_perceptron(x_test, y_test, w, b)

    endTime = time.time()

    print("Total Spend time is {}".format(endTime - startTime))