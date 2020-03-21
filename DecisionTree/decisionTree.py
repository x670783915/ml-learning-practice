import pandas as pd
import numpy as np
import time
from collections import Counter

# 计算太慢了, 程序测试能够运行
# ID3 acc: , cost time:
# C4.5 acc: , cost time:

def loadData(fileName):
    data = pd.read_csv(fileName, header=None)
    data = data.values

    Y = data[:, 0]
    X = data[:, 1:]

    X[X < 128] = 0
    X[X >= 128] = 1

    # return np.array(X), np.array(Y)
    return X, Y

# 计算信息熵
def cacu_H_D(Y):
    types = set([i for i in Y])
    HD = 0

    for i in types:
        D_i = np.sum(Y == i)
        HD += (-1) * (D_i / len(Y)) * np.log((D_i / len(Y)))
    return HD

# 计算条件概率
def cacu_H_D_A(column, Y):
    types = set([i for i in column])

    H_D_A = 0
    for i in types:
        # 计算 |Di| / |D|
        Di_D = np.sum(column == i) / len(column)
        # 条件概率
        H_Di = cacu_H_D(Y[column == i])
        # 累加和
        H_D_A += Di_D * H_Di

    return H_D_A

# 计算特征熵
def cacu_A(column):
    types = set([i for i in column])
    H_A = 0
    for i in types:
        D_i = np.sum(column == i)
        H_A += (-1) * (D_i / len(column)) * np.log((D_i / len(column)))
    return H_A

# 找列
def findMaxFeature(X, Y, mode='id3'):
    num_features = X.shape[1]

    H_D = cacu_H_D(Y)
    H_D_A = 0
    max_gain = -np.inf
    max_feature = -1

    for i in range(num_features):
        H_D_A = cacu_H_D_A(X[:, i], Y)

        if mode == 'id3':
            if H_D - H_D_A > max_gain:
                max_gain = H_D - H_D_A
                max_feature = i
        elif mode == 'c45':
            gain_r = (H_D - H_D_A) / cacu_A(X[:, i])
            if gain_r > max_gain:
                max_gain = gain_r
                max_feature = i
    
    return max_feature, max_gain

# 找列中类别最多的
def findMostCommot(column):
    res = Counter(column)
    return res.most_common(1)[0][0]

def cutData(X, Y, Ag, ai):
    ret_train_data = []
    ret_train_label = []

    for i in range(X.shape[0]):
        if X[i][Ag] == ai:
            ret_train_data.append(np.concatenate([X[i][0:Ag], X[i][Ag+1:]]))
            ret_train_label.append(Y[i])

    return np.array(ret_train_data), np.array(ret_train_label)

def createTree(X, Y):
    epsilon = 0.1

    clusters = set([i for i in Y])

    if len(clusters) == 1:
        return Y[0]

    if len(X[0]) == 0:
        return findMostCommot(Y)

    feature, gain = findMaxFeature(X, Y)

    if gain < epsilon:
        return findMostCommot(Y)
    
    types = set([i for i in X[:, feature]])

    tree_dic = {feature:{}}

    for i in types:
        rest_X, rest_Y = cutData(X, Y, feature, i)
        tree_dic[feature][i] = createTree(rest_X, rest_Y)
    
    return tree_dic

def predict(X, tree):
    while True:
        # 注意这里的逗号
        (key, value), = tree.items()
        if isinstance(value, dict):
            feature = X[key]

            # 注意x_test需要为list，才可以用del
            # del X[key]
            X = np.delete(X, key, axis=0)

            tree = value[feature]
            # print(type(tree))
            if not isinstance(tree, dict):
                return tree
        else:
            return value

def test(X, Y, tree):
    cnt = 0
    for i in range(X.shape[0]):
        # print(i)
        pred_y = predict(X[i], tree)
        if pred_y == Y[i]:
            cnt += 1
    print('acc is {}'.format(cnt / X.shape[0]))


if __name__ == '__main__':

    startTime = time.time()
    train_data_path = './datasets/Mnist/mnist_train.csv'
    test_data_path = './datasets/Mnist/mnist_test.csv'
    # 读取训练文件
    X_train, y_train = loadData(train_data_path)
    X_train = X_train[:100]
    y_train = y_train[:100]
    # 读取测试文件
    X_test, y_test = loadData(test_data_path)
    X_test = X_test[:100]
    y_test = y_test[:100]
    tree = createTree(X_train, y_train)

    test(X_test, y_test, tree)

    endTime = time.time()

    print('spent time is:{}'.format(endTime - startTime))