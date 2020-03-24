import pandas as pd
import numpy as np
import time

# Mnist 训练集1000 测试集200
# spent time: 367s
# acc: 0.98

def loadData(fileName):
    data = pd.read_csv(fileName, header=None)

    data = data.values

    Y = data[:, 0]
    X = data[:, 1:]

    #数据二值化，返回数据
    #因为xi的取值范围为0-255，那么划分点太多了，我们进行二值化
    # 二值化之后，我们使用-0.5,0.5,1.5三个点即可
    X[X < 128] = 0
    X[X >= 128] = 1

    # 以0作为分类。0设置为-1，其他设置为1
    Y[Y == 0] = -1
    Y[Y >= 1] = 1

    return X, Y

def cal_Gx_e(X, Y, div, relu, D, feature):
    '''
    用于计算在该特征下，使用条件为rule，样本权重分布为D，划分点为div，返回划分结果和误差率
    :param:X 样本
    :param:Y 标签
    :param div: 划分点
    :param rule: 划分规则，大于div为1还是0
    :param D: 样本权重分布
    :param feature: 样本的的几个特征（总共有784（28*28）个）
    :return: Gx，e
    '''
    x = X[:, feature]
    Gx = []
    e = 0

    # LessIsOne：即小于划分点为1，大于为0
    # BiggerIsOne：大于划分点为1，小于为-1
    if relu == 'LessIsOne':
        L, B = 1, -1
    else:
        L, B = -1, 1
    
    for i in range(len(x)):
        if x[i] > div:
            Gxi = B
        else:
            Gxi = L
        Gx.append(Gxi)
        if Gxi != Y[i]:
            e += D[i]
    
    return np.array(Gx), e


def create_single_boosting_tree(X, Y, D):
    single_boosting_tree = {}
    m, n = X.shape

    single_boosting_tree['e'] = 1
    for i in range(n):
        for rule in ['LessIsOne', 'BiggerIsOne']:
            for div in [-0.5, 0.5, 1.5]:
                tmpGx, tmpe = cal_Gx_e(X, Y, div, rule, D, i)
                if tmpe < single_boosting_tree['e']:
                    single_boosting_tree['e'] = tmpe
                    single_boosting_tree['Gx'] = tmpGx
                    single_boosting_tree['div'] = div
                    single_boosting_tree['rule'] = rule
                    single_boosting_tree['feature'] = i
    single_boosting_tree['alpha'] = 0.5 * np.log((1-single_boosting_tree['e'])/single_boosting_tree['e'])
    return single_boosting_tree


def create_boosting_tree(X, Y, tree_num=50):
    m, n = X.shape
    D = np.array([1/m] * m)
    Fx = [0] * m
    boosting_tree = []

    for i in range(tree_num):
        single_boosting_tree = create_single_boosting_tree(X, Y, D)

        Zm = np.sum(D*np.exp(-1*single_boosting_tree['alpha']*Y*single_boosting_tree['Gx']))
        
        D = D / Zm * np.exp(-1*single_boosting_tree['alpha']*Y*single_boosting_tree['Gx'])

        boosting_tree.append(single_boosting_tree)

        Fx += single_boosting_tree['alpha'] * single_boosting_tree['Gx']

        Gx = np.sign(Fx)

        total_error_num = np.sum([1 for i in range(m) if Gx[i] != Y[i]])
        total_error_rate = total_error_num / m
        if total_error_num == 0:
            return boosting_tree
        
        print(f'in {i}th epoch, error={single_boosting_tree["e"]}. total error is {total_error_rate}')
    return boosting_tree

def predict(x, tree):
    fx = 0

    for i in range(len(tree)):
        div = tree[i]['div']
        relu = tree[i]['rule']
        alpha = tree[i]['alpha']
        feature = tree[i]['feature']

        if relu == 'LessIsOne':
            if x[feature] < div:
                fx += alpha
            else:
                fx -= alpha
        else:
            if x[feature] < div:
                fx -= alpha
            else:
                fx += alpha
    
    Gx = np.sign(fx)
    return Gx

def test(X, Y, tree):
    acc = 0
    print('Testing...')
    for i in range(len(X)):
        Gx = predict(X[i], tree)
        if Gx == Y[i]:
            acc += 1
        if i % 10 == 0:
            print('*')
    print('acc is {}'.format(acc * 1.0 / len(X)))

def testiris():
    # # 鸢尾花数据集 100%
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=3)
    y_train[y_train > 0] = 1
    y_train[y_train == 0] = -1
    y_test[y_test > 0] = 1
    y_test[y_test == 0] = -1
    
    boosting_tree=create_boosting_tree(X_train,y_train,10)
    test(X_test,y_test,boosting_tree)


if __name__ == '__main__':
    startTime = time.time()
    train_data_path = './datasets/Mnist/mnist_train.csv'
    test_data_path = './datasets/Mnist/mnist_test.csv'
    # # 读取训练文件
    # X_train, y_train = loadData(train_data_path)
    # # 读取测试文件
    # X_test, y_test = loadData(test_data_path)
    # tree = create_boosting_tree(X_train[:1000], y_train[:1000], 30)
    # test(X_test[:200], y_test[:200], tree)
    testiris()

    endTime = time.time()
    print('spent time is:{}'.format(endTime - startTime))
