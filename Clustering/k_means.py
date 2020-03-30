import numpy as np
import collections

def cacuDist(x1, x2):
    return np.sum(np.square(x1-x2))

def cacu_cluster_d(D, cluster1, cluster2):
    minD = np.inf
    for i in cluster1:
        for j in cluster2:
            minD = min(minD, D[i][j])
    return minD

def find_miniDist_index(D, cluster):
    minD = np
    indexi, indexj = -1, -1
    for i in cluster.keys():
        for j in cluster.keys():
            if i!=j:
                d = cacu_cluster_d(D, cluster[i], cluster[j])
                if d < minD:
                    minD = d
                    indexi = i
                    indexj = j
    return indexi, indexj

def hierarchical_clustering(data, k):
    cluster = {}
    for i in range(len(data)):
        cluster[i] = i
    D = [[0 for _ in range(len(data))] for _ in range(len(data))]
    for i in range(len(data)):
        for j in range(len(data)):
            d = cacuDist(data[i], data[j])
            D[i][j] = d
            D[j][i] = d
    clusters = len(cluster)

    while clusters > k:
        i, j = find_miniDist_index(D, cluster)
        cluster[i].extend(cluster[j])
        del cluster[j]
        clusters = len(cluster)
    
    initial_start = []
    for i in cluster.keys():
        center = np.array(0. for _ in range(data.shape[1]))
        for j in range(len(cluster[i])):
            center += data[cluster[i][j]]
        center /= len(cluster[i])
    
        miniDist = np.inf
        index = -1
        for j in range(len(cluster[i])):
            tmp = cacuDist(center, data[cluster[i][j]])
            if tmp < miniDist:
                miniDist = tmp
                index = cluster[i][j]
        initial_start.append(index)
    
    return initial_start

def loadData(fileName):
    data = []
    with open(fileName) as file:
        for line in file.readlines():
            line = line.split(',')[:-1]
            linedata = []
            for i in range(len(line)):
                linedata.append(eval(line[i]))
            linedata = np.array(linedata)
            data.append(linedata)
    return np.array(data)

def k_mean(start, data, k):
    m, n = data.shape
    cluster = {}
    cluster_center = {}
    for i in range(m):
        cluster[i] = -1
    for i in range(k):
        cluster_center[i] = data[start[i]]
    
    changed_data = 1
    while changed_data:
        changed_data = 0
        for i in range(m):
            minDist = np.inf
            cluster_belonging = -1
            for c in range(len(cluster_center)):
                distance = cacuDist(data[i], cluster_center[c])
                if distance < minDist:
                    minDist = distance
                    cluster_belonging = c
            if cluster_belonging != cluster[i]:
                changed_data += 1
                cluster[i] = cluster_belonging
        count = [0 for _ in range(k)]
        center=[np.array([0. for _ in range(n)]) for _ in range(k)]
        for index, c in cluster.items():
            # index 为0-m，表示样本的序号
            # c 0，1，2。表示的样本的类别
            # 计算得到每一类样本点的和，还需要需要除以个数得到中心点
            center[c]+=data[index]
            count[c]+=1
        # 修改中心
        for i in range(k):
            cluster_center[i]=center[i]/count[i]

    return cluster