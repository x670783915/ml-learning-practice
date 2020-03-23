import pandas as pd
import numpy as np
import time

# Mnist 训练集 2000, 测试集 400
# spent time: 327s
# acc: 0.99

def loadData(fileName):

    data = pd.read_csv(fileName)

    data = data.values

    Y = data[:, 0]
    X = data[:, 1:] / 255

    Y[Y > 0] = 1
    Y[Y == 0] = -1

    return X, Y

class SVM:
    
    def __init__(self, X, Y, sigma=10, C=200, toler=0.0001):
        self.m, self.n = X.shape
        self.X, self.Y = X, Y
        self.sigma = sigma
        self.C = C
        self.toler = toler
        self.E = -Y
        self.alpha = [0] * len(Y)
        self.K = self.calKernel()
        self.b = 0
        self.S = []
    
    def calKernel(self):
        print('generate the Gaussion kernel matrix...')
        K = np.zeros((self.m, self.m))
        for i in range(self.m):
            xi = self.X[i]
            for j in range(self.m):
                xj = self.X[j]
                Kij = np.exp(-1*np.sum(np.square(xi-xj)) / (2*self.sigma**2))
                K[i][j], K[j][i] = Kij, Kij
        print('the Gaussion kernel matrix generated!')
        return K

    def calGx(self, i):
        index = [j for j in range(self.m) if self.alpha[j] != 0]
        Gxi = 0
        for j in index:
            Gxi += self.alpha[j] * self.Y[j] * self.K[j][i]
        Gxi += self.b
        return Gxi

    def calEi(self, i):
        return self.calGx(i) - self.Y[i]

    def alpha1_break_KKT(self, i):
        yi = self.Y[i]
        Gxi = self.calGx(i)
        alpha1 = self.alpha[i]

        if alpha1 > -self.toler and alpha1 <= self.C + self.toler and np.abs(Gxi*yi-1) <= self.toler:
            return False
        elif np.abs(alpha1) <= self.toler and yi*Gxi >= 1:
            return False
        elif np.abs(alpha1-self.C) <= self.toler and yi*Gxi <= 1:
            return False
        return True
        
    def getAlpha2(self, i, Ei):
        if Ei > 0:
            index = np.argmin(self.E)
            return index, self.calEi(index)
        else:
            index = np.argmax(self.E)
            return index, self.calEi(index)
    
    def train(self, epoch=100):
        step = 0
        alphaChanged = 1

        while step < epoch and alphaChanged > 0:
            print(f'in the {step}th train epoch')
            step += 1
            alphaChanged = 0

            # SMO 算法
            for i in range(self.m):
                
                if self.alpha1_break_KKT(i):
                    E1 = self.calEi(i)

                    j,E2 = self.getAlpha2(i, E1)
                    alpha1 = self.alpha[i]
                    alpha2 = self.alpha[j]

                    if self.Y[i] == self.Y[j]:
                        k = alpha1 + alpha2
                        L = max(0, k - self.C)
                        H = min(self.C, k)
                    else:
                        k = alpha1 - alpha2
                        L = max(0, -k)
                        H = min(self.C, self.C - k)
                    
                    if L == H:
                        continue
                        
                    eta = self.K[i][i] + self.K[j][j] - 2*self.K[i][j]
                    alpha2_new = alpha2 + self.Y[j] * (E1-E2) / eta

                    if alpha2_new > H:
                        alpha2_new = H
                    elif alpha2_new < L:
                        alpha2_new = L
                    
                    alpha1_new = alpha1 + self.Y[i]*self.Y[j]*(alpha2 - alpha2_new)

                    b1_new = -1 * E1 - self.Y[i] * self.K[i][i]*(alpha1_new - alpha1) - self.Y[j] * self.K[i][j]*(alpha2_new - alpha2) + self.b
                    b2_new = -1 * E2 - self.Y[i] * self.K[i][j]*(alpha1_new - alpha1) - self.Y[j] * self.K[j][j]*(alpha2_new - alpha1) + self.b

                    if alpha1_new > 0 and alpha1_new < self.C:
                        self.b = b1_new
                    elif alpha2_new > 0 and alpha2_new < self.C:
                        self.b = b2_new
                    else:
                        self.b = (b1_new + b2_new) / 2
                    
                    self.alpha[i] = alpha1_new
                    self.alpha[j] = alpha2_new

                    self.E[i] = self.calEi(i)
                    self.E[j] = self.calEi(j)
            
                    if np.abs(alpha2_new - alpha2) >= 1e-5:
                        alphaChanged += 1
                    
                    print(f'step num:{step},changed alpha num: {alphaChanged}')
        for i in range(self.m):
            if self.alpha[i] > 0:
                self.S.append(i)

    def guassionF(self, xi, xj):
        return np.exp(-1*np.sum(np.square(xi-xj)) / (2 * self.sigma ** 2))

    def predict(self, x):
        Gx = 0
        for i in self.S:
            Gx += self.alpha[i] * self.Y[i] * self.guassionF(x, self.X[i])
        Gx += self.b
        return np.sign(Gx)
    
    def test(self, X_test, Y_test):
        acc = 0
        for i in range(len(X_test)):
            pred = self.predict(X_test[i])
            if pred == Y_test[i]:
                acc += 1
                print(acc)
        print('acc is: {}'.format(acc * 1.0 / len(Y_test)))


if __name__ == '__main__':

    startTime = time.time()
    train_data_path = './datasets/Mnist/mnist_train.csv'
    test_data_path = './datasets/Mnist/mnist_test.csv'
    # 读取训练文件
    X_train, y_train = loadData(train_data_path)
    # 读取测试文件
    X_test, y_test = loadData(test_data_path)
    
    svm = SVM(X_train[:2000], y_train[:2000])
    svm.train(100)
    svm.test(X_test[:400], y_test[:400])

    endTime = time.time()
    print('spent time is:{}'.format(endTime - startTime))