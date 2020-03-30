'''
高斯混合模型
EM算法的一个重要应用

几何角度
高斯混合模型中，每一个数据就是一个加权平均，是由多个高斯分布叠加而成

生成模型角度
数据生成过程如下
1.首先在多个模型中，依据概率 alpha_k 选择一个模型 Gaussion_k
2.按照这个概率模型 Gaussion(y | theta_k) 随机产生一个观测值 y_i
3.反复上面的过程，就可以产生所有数据


构造高斯混合模型比如
real_alpha_list = [0.1, 0.4, 0.5]
real_mu_list = [3, -1, 0]
real_sigma_list = [1, 4, 3]

预测值
alpha [0.0800, 0.4632, 0.4569]
mu:   [2.8467, -1.0024, 0.3894]
sigma:[0.6850, 3.8692, 3.0489]
'''

import numpy as np

def produce_data(alpha_list, mu_list, sigma_list, length):
    '''
    产生高斯混合模型数据
    :param alpha_list: 所有alpha的值
    :param mu_list: 所有mu的值
    :param sigma_list: 所有sigma的值
    :param length: 数据总长度
    :return: 返回高斯混合模型数据
    '''
    data = []
    for i in range(len(alpha_list)):
        # 设置种子方便复现
        np.random.seed(3)
        # normal(均值，方差，大小)
        data_i = np.random.normal(mu_list[i], sigma_list[i], int(length * alpha_list[i]))
        # 数组拼接，都是一维则是一维扩展
        data.extend(data_i)
    return np.array(data)

def gaussion(y, mu, sigma):
    '''
    单个高斯模型的概率密度函数值
    :param y:观测数据
    :param mu:单个高斯模型的均值
    :param sigma:单个高斯模型的标准差
    :return: 单个高斯模型概率密度函数值
    '''
    return 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-np.square(y - mu) / (2*sigma**2))

def e_step(data, alpha_list, mu_list, sigma_list, length):
    '''
    计算gamma_jk
    对应于算法9.2 E步
    :param data：所有样本
    :param alpha_list: 所有alpha的值
    :param m2: 所有mu的值
    :param sigma_list: 所有sigma的值
    :param length:数据长度
    :return: gamma
    '''
    K = len(alpha_list)
    gamma = np.array([[0.] * K for _ in range(length)])

    for j in range(length):
        k = 0
        for k in range(K):
            gamma[j][k] = alpha_list[k] * gaussion(data[j], mu_list[k], sigma_list[k])
        
        if k == K-1:
            gamma[j, :] = gamma[j, :] / sum(gamma[j])
    
    return gamma

def m_step(data, gamma, alpha_list, mu_list, sigma_list):
    '''
    # 对应于算法9.2M步
    :param data:
    :param gamma: 响应度
    :param mu_list:
    :param sigma_list:
    :param alpha_list:
    :return: 更新之后的参数
    '''
    for k in range(len(alpha_list)):
        sigma_list[k] = np.sqrt(np.sum(gamma[:, k] @ np.square(data-mu_list[k])) / sum(gamma[:, k]))

        mu_list[k] = np.sum(gamma[:, k] @ data) / np.sum(gamma[:, k])

        alpha_list[k] = np.sum(gamma[:, k]) / len(data)
    return alpha_list, mu_list, sigma_list

def GMMEM(data, epochs=500):
    # 算法第一步，取初始值
    # 可以随机选取，但注意，EM算法最后并不一定得到全局最优解
    # 初始值的选取可能会对算法结果有较大的影响
    alpha_list=[0.2,0.4,0.4]
    mu_list=[0,-2,2]
    sigma_list=[1,2,3]

    for i in range(epochs):
        # 反复迭代2，3步骤
        gamma=e_step(data,alpha_list,mu_list,sigma_list,len(data))
        alpha_list,mu_list,sigma_list=m_step(data,gamma,alpha_list,mu_list,sigma_list)
        if i%100==0:
            print(f'epoch={i}')
            print(f'alpha={alpha_list}')
            print(f'mu={mu_list}')
            print(f'sigma={sigma_list}')

    # 返回参数
    return alpha_list,mu_list,sigma_list

if __name__=='__main__':

    # 设置缓和高斯模型参数，以生成数据

    real_alpha_list = [0.1, 0.4, 0.5]
    real_mu_list = [3, -1, 0]
    real_sigma_list = [1, 4, 3]

    print(f'real model parameter is: alpha {real_alpha_list};mu:{real_mu_list};sigma:{real_sigma_list}')
    data = produce_data(real_alpha_list,real_mu_list,real_sigma_list,2000)

    alpha, mu, sigma = GMMEM(data)

    print(f'predict model parameter is: alpha {alpha};mu:{mu};sigma:{sigma}')