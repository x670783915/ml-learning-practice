import numpy as np
import pandas as pd
import time
import heapq
from collections import Counter

# KDTree (100样例, k=10)
# spent time: 169s
# acc: 0.97
"""
 KDTree 如果对于特征是二进制特征的话，
 很明显KDTree会退化为线性结构，就没意义了
 比如ORB特征Fast检测，BRIEF转换成二进制特征向量
"""

def loadData(filaName):
    data = pd.read_csv(filaName, header=None)
    data = data.values
    y_label = data[:, 0]
    x_data = np.array(data[:, 1:])
    return x_data / 255, y_label

class Node:
    def __init__(self, data, label, sp=0, left=None, right=None):
        self.data = data
        self.label = label
        self.sp = sp
        self.left = left
        self.right = right

class KDTree:
    def __init__(self, datasets, labels):
        self.k = datasets.shape[1]
        self.root = self.create(datasets, labels, 0)

    def create(self, dataset, labels, sp):
        if len(dataset) == 0:
            return None
        
        index = dataset[:, sp].argsort()
        mid = len(dataset) // 2
        dat = dataset[index[mid]]
        label = labels[index[mid]]
        left = self.create(dataset[index[:mid]], labels[index[:mid]], (sp+1) % self.k)
        right = self.create(dataset[index[mid+1:]], labels[index[mid+1:]], (sp+1) % self.k)
        return Node(dat, label, sp, left, right)
    
    def nearest(self, x, near_k=1, p=2):
        self.knn = [(-np.inf, None)] * near_k
        
        def visit(node):
            if node:
               dis = x[node.sp] - node.data[node.sp]
               visit(node.left if dis < 0 else node.right)
               curr_dis = np.linalg.norm(x-node.data, p)
               heapq.heappushpop(self.knn, (-curr_dis, node))

               if -(self.knn[0][0]) > abs(dis):
                   visit(node.right if dis < 0 else node.left)

        visit(self.root)
        self.knn_data = np.array([i[1].data for i in heapq.nlargest(near_k, self.knn)])
        self.knn_label = np.array([i[1].label for i in self.knn])
        # print(self.knn_label)
        belonging = Counter(self.knn_label).most_common(1)
        return belonging[0][0]

def test_kdtree_mnist():
    startTime = time.time()
    train_data_path = './datasets/Mnist/mnist_train.csv'
    test_data_path = './datasets/Mnist/mnist_test.csv'

    x_train, y_train = loadData(train_data_path)
    x_test, y_test = loadData(test_data_path)

    kd_tree = KDTree(x_train, y_train)

    cnt = 0
    for i in range(100):
        pred_y = kd_tree.nearest(x_test[i], near_k=10, p=2)
        print(pred_y, y_test[i], i)
        if pred_y == y_test[i]:
            cnt += 1
    print("acc is {}".format(cnt / 100))

    endTime = time.time()
    print('spend time is :{}'.format(endTime - startTime))


if __name__ == '__main__':
    
    # from pylab import *
    # data = array([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]])
    # labels = array([1, 2, 3, 4, 5, 6])
    # kdtree = KDTree(data, labels)
    # target = array([7.5, 3])
    # res = kdtree.nearest(target, 2)
    # print(res)
    # plot(*data.T, 'o')
    # plot(*target.T, '.r')
    # plot(*kdtree.knn_data.T, 'r+')
    # show()

    test_kdtree_mnist()



# 理清思路很重要
# def calDistance(self, xi, xj):
#     return np.sqrt(np.sum(np.square(xi-xj)))

# class Node(object):
#     def __init__(self, data=None, label=None, left=None, right=None, parent=None):
#         self.data = data
#         self.label = label
#         self.left = left
#         self.right = right
#         self.parent = parent

# class KdNode(Node):
#     def __init__(self, data=None, label=None, left=None, right=None, parent=None, axis=None, dimensions=None):
#         """为KD树创建一个新的节点
#         """
#         super(KdNode, self).__init__(data, label, left, right, parent)
#         self.axis = axis
#         self.dimensions = dimensions

# class KdTree(object):
#     def __init__(self, data, label, k):
#         self.data = data
#         self.label = label
#         self.k = k
#         self.tree = None
    
#     def gen_tree(self):
#         startTime = time.time()
#         self.tree = self.create(data, label, dimensions=len(data[0]), axis=0)
#         endTime = time.time()

#     # left = create(point_list[:median], dimensions)
#     # right = create(point_list[median+1:], dimensions)
#     def create(self, point_list=None, label_list=None, parent=None, dimensions=None, axis=0):
#         """从一个列表输入中创建一个kd树
#         列表中的所有点必须有相同的维度。
#         如果输入的point_list为空，一颗空树将被创建，这时必须提供dimensions的值
#         如果point_list和dimensions都提供了，那么必须保证前者维度为dimensions
#         axis表示根节点切分数据的位置"""

#         if not point_list:
#             return None

#         assert len(point_list[0]) == dimensions
#         assert len(point_list) == len(label_list)
        
#         point_index = point_list[:, axis].argsort()
#         median = len(point_index) // 2
#         point_list = point_list[point_index[median]]
#         label_list = label_list[point_index[median]]

#         loc_data = point_list[median]
#         loc_label = label_list[median]
#         node = KdNode(loc_data, loc_label, left, right, parent, axis=axis, dimensions=dimensions)
#         child_axis = (axis + 1) % dimensions
#         left = self.create(point_list[point_index[:median]], label_list[point_index[:median]], node, dimensions, child_axis)
#         right = self.create(point_list[point_index[median+1:]], label_list[point_index[median+1:]], node, dimensions, child_axis)
#         return node

#     def find(self, x, k=1):
        
#         node = self.tree
#         if not node:
#             return None
        
#         nearst_list = []

#         nearst_node = self._find_nearst(node, x)

#     def _find_nearst(self, node, x):
#         if not node.left and not node.right:
#             return node
        
#         if node[node.axis] >= x[node.axis] and node.left:
#             return self._find_nearst(node.left, x)
#         if node[node.axis] < x[node.axis] and node.right:
#             return self._find_nearst(node.right, x)

# class BounderPriorityQueue:
#     def __init__(self, k):
#         # 使用堆实现的优先队列
#         self.heap = []
#         self.k = k
#         self.min_dist = None
    
#     def items(self):
#         return self.heap

#     def parent(self, index):
#         return int(index // 2)
    
#     def left_child(self, index):
#         return index * 2
    
#     def right_child(self, index):
#         return index * 2 + 1
    
#     def max_heapify(self, index):
#         left_index = self.left_child(index)
#         right_index = self.right_child(index)

#         largst = index
#         if left_index < len(self.heap) and self._dist(left_index) > self._dist(index):
#             largst = left_index
#         if right_index < len(self.heap) and self._dist(right_index) > self._dist(largst):
#             largst = right_index
#         if largst != index:
#             self.heap[index], self.heap[largst] = self.heap[largst], self.heap[index]
#             self.max_heapify(largst)
        
#     def propagate_up(self, index):
#         """在index位置添加新元素后，通过不断和父节点比较并交换
#             维持最大堆的特性，即保持堆中父节点的值永远大于子节点"""
#         while index != 0 and self._dist(self.parent(index)) <= self._dist(index):
#             self.heap[index], self.heap[self.parent(index)] = self.heap[self.parent(index)], self.heap[index]
#             index = self.parent(index)
    
#     def add(self, obj):
#         size = self.size()
#         if size == self.k:
#             max_elem = self.max()
#             if obj[1] < max_elem:
#                 self.extract_max()
#                 self.heap_append(obj)
#         else:
#             self.heap_append(obj)
    
#     def size(self):
#         return len(self.heap)
    
#     def max(self):
#         return self.heap[0][4]
    
#     def extract_max(self):
#         top = self.heap[0]
#         data = self.heap.pop()
#         if len(self.heap) > 0:
#             self.heap[0] = data
#             self.max_heapify(0)
#         return top

#     def _seach_node(self, point, k, result, get_dist):
#         if not self:
#             return
        
#         nodeDist = get_dist(self)

#         results.add((self, nodeDist))
#         split_plane = self.data[self.axis]
#         place_dist = point[self.axis] - split_plane
#         place_dist2 = place_dist ** 2

#         #如果当前节点小于队列中至少一个节点，则将该节点添加入队列
#         #该功能由BoundedPriorityQueue类实现
#         results.add((self,nodeDist))

#         #获得当前节点的切分平面
#         split_plane = self.data[self.axis]
#         plane_dist = point[self.axis] - split_plane
#         plane_dist2 = plane_dist ** 2

#         #从根节点递归向下访问，若point的axis维小于且分点坐标
#         #则移动到左子节点，否则移动到右子节点
#         if point[self.axis] < split_plane:
#             if self.left is not None:
#                 self.left._search_node(point,k,results,get_dist)
#         else:
#             if self.right is not None:
#                 self.right._search_node(point,k,results,get_dist)

#         #检查父节点的另一子节点是否存在比当前子节点更近的点
#         #判断另一区域是否与当前最近邻的圆相交
#         if plane_dist2 < results.max() or results.size() < k:
#             if point[self.axis] < self.data[self.axis]:
#                 if self.right is not None:
#                     self.right._search_node(point,k,results,get_dist)
#             else:
#                 if self.left is not None:
#                     self.left._search_node(point,k,results,get_dist)
    
#     def search_knn(self,point,k,dist=None):
#         """返回k个离point最近的点及它们的距离"""

#         if dist is None:
#             get_dist = lambda n:n.dist(point)
#         else:
#             gen_dist = lambda n:dist(n.data, point)

#         results = BoundedPriorityQueue(k)
#         self._search_node(point,k,results,get_dist)

#         #将最后的结果按照距离排序
#         BY_VALUE = lambda kv: kv[1]
#         return sorted(results.items(), key=BY_VALUE)
