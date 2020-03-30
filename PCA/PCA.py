import numpy as np

def normalize(X):
    m, n = X.shape

    # 每个特征维度的均值
    X_mean = np.mean(X, axis=1)

    S = [0 for _ in range(n)]
    for i in range(m):
        # 特征为均值方差
        S[i] = (1 / (n-1) * np.sum(np.square(X[i] - X_mean[i])))
        for j in range(n):
            X[i][j] = (np.float(X[i][j]) - X_mean[i]) / np.sqrt(S[i])
    return X

def nor(X):
    X_mean = np.mean(X, axis=1)
    for i in range(len(X)):
        X[i] = X[i] - X_mean[i]
    return X

def createX(X):
    _, n = X.shape
    return 1 / (np.sqrt(n-1)) * X.T

def svd(X, target=0.85):
    U, sigma, VT = np.linalg.svd(X)

    lambdas = [sigma[i] for i in range(len(sigma))]
    k = 0
    total = sum(lambdas)

    for i in range(1, len(lambdas)):
        if sum(lambdas[:i]) / total > target:
            k = i
            break
    
    return VT.T[:, :k]

def svd_k(X, k):
    U, sigma, VT = np.linalg.svd(X)
    return VT.T[:, :k]

if __name__ == '__main__':

    from sklearn.decomposition import PCA

    X = np.array([[-1., 1.,3.], [-2., -1.,4.], [-3., -2.,5.], [1., 1.,-2.], [2., 1.,-4.], [3., 2.,-5.]])
    pca = PCA(n_components=2)
    pca.fit(X)
    pca.transform(X)
    print(pca.transform(X))

    # 注意X样本[m, n] m是维度，n代表样本数，
    # 所以这里用书中方法需要进行转置
    X_norm = nor(X.T)
    X_new = createX(X_norm)
    VT = svd_k(X_new, 2)
    VT2 = svd(X_new, 0.85)

    print(VT.T @ X.T)
    print(VT.T @ X.T)

    # 可以发现，标准化不同，得到的最后主成分也有不同
    X_norm = normalize(X.T)
    X_new = createX(X_norm)
    VT = svd_k(X_new, 2)
    VT2 = svd(X_new, 0.85)

    print(VT.T @ X.T)
    print(VT2.T @ X.T)

    # a = np.array([[1, 2],[1, 2]])
    # b = np.array([[1, 2],[1, 2]])
    # print(a * b) # 对应元素点乘
    # print(a @ b) # 矩阵乘法
    # print(a.dot(b)) # 矩阵乘法

    # a = np.array([1， 2])
    # b = np.array([1， 2])
    # print(a * b) # 对应元素点乘
    # print(a @ b) # 矩阵乘法
    # print(a.dot(b)) # 矩阵乘法