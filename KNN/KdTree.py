import numpy as np
import pandas as pd
import time
from collections import Counter


def loadData(filaName):
    data = pd.read_csv(filaName, header=None)

    data = data.values

    y_label = data[:, 0]

    x_data = np.mat(data[:, 1:])

    return x_data / 255, y_label

class Node(object):
    def __init__(self, data=None, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right

class KdNode(Node):
    def __init__(self, data=None, left=None, right=None, axis=None, sel_axis=None, dimensions=None):
        """为KD树创建一个新的节点
        如果该节点在树中被使用，axis和sel_axis必须被提供。
        sel_axis(axis)在创建当前节点的子节点中将被使用，
        输入为父节点的axis，输出为子节点的axis"""
        
        super(KdNode, self).__init__(data, left, right)
        self.axis = axis
        self.sel_axis = sel_axis
        self.dimensions = dimensions

# left = create(point_list[:median], dementions, sel_axis(axis))
# right = create(point_list[median+1:], dementions, sel_axis(axis))
def create(point_list=None, dimensions=None, axis=0, sel_axis=None):
    """从一个列表输入中创建一个kd树
    列表中的所有点必须有相同的维度。
    如果输入的point_list为空，一颗空树将被创建，这时必须提供dimensions的值
    如果point_list和dimensions都提供了，那么必须保证前者维度为dimensions
    axis表示根节点切分数据的位置，sel_axis(axis)在创建子节点时将被使用，
    它将返回子节点的axis"""

    if not point_list and not dimensions:
        raise ValueError('either point_list or dimensions should be provided')
    elif point_list:
        dimensions = check_dimentionality(point_list, dimensions)
    
    sel_axis = sel_axis or (lambda pre_axis: (pre_axis+1) % demensions)
    
    if not point_list:
        return KdNode(sel_axis=sel_axis, axis=axis, dimensions=dimensions)
    
    point_list = list(point_list)
    point_list.sort(key=lambda point: point[axis])
    median = len(point_list) // 2

    loc = point_list[median]
    left = create(point_list[:median], dimensions, sel_axis(axis))
    right = create(point_list[median+1:], dimensions, sel_axis(axis))
    return KdNode(loc, left, right, axis=axis, sel_axis=sel_axis, dimensions=dimensions)

def check_dimentionality(point_list, dimentsions=None):
    dimentsions = dimentsions or len(point_list[0])
    for p in point_list:
        if len(p) != dimentsions:
            raise ValueError('All Points in the point_list must have the same dimensionality')
    return dimentsions

class BounderPriorityQueue:
    def __init__(self, k):
        # 使用堆实现的优先队列
        self.heap = []
        self.k = k
    
    def items(self):
        return self.heap

    def parent(self, index):
        return int(index // 2)
    
    def left_child(self, index):
        return index * 2
    
    def right_child(self, index):
        return index * 2 + 1
    
    def _dist(self, index):
        return self.heap[index][3]
    
    def max_heapify(self, index):
        left_index = self.left_child(index)
        right_index = self.right_child(index)

        largst = index
        if left_index < len(self.heap) and self._dist(left_index) > self._dist(index):
            largst = left_index
        if right_index < len(self.heap) and self._dist(right_index) > self._dist(largst):
            largst = right_index
        if largst != index:
            self.heap[index], self.heap[largst] = self.heap[largst], self.heap[index]
            self.max_heapify(largst)
        
    def propagate_up(self, index):
        """在index位置添加新元素后，通过不断和父节点比较并交换
            维持最大堆的特性，即保持堆中父节点的值永远大于子节点"""
        while index != and self._dist(self.parent(index)) <= self._dist(index):
            self.heap[index], self.heap[self.parent(index)] = self.heap[self.parent(index)], self.heap[index]
            index = self.parent(index)
    
    def add(self, obj):
        size = self.size()
        if size == self.k:
            max_elem = self.max()
            if obj[1] < max_elem:
                self.extract_max()
                self.heap_append(obj)
        else:
            self.heap_append(obj)
    
    def size(self):
        return len(self.heap)
    
    def max(self):
        return self.heap[0][4]
    
    def extract_max(self):
        top = self.heap[0]
        data = self.heap.pop()
        if len(self.heap) > 0:
            self.heap[0] = data
            self.max_heapify(0)
        return top

    def _seach_node(self, point, k, result, get_dist):
        if not self:
            return
        
        nodeDist = get_dist(self)

        results.add((self, nodeDist))
        split_plane = self.data[self.axis]
        place_dist = point[self.axis] - split_plane
        place_dist2 = place_dist ** 2

        #如果当前节点小于队列中至少一个节点，则将该节点添加入队列
        #该功能由BoundedPriorityQueue类实现
        results.add((self,nodeDist))

        #获得当前节点的切分平面
        split_plane = self.data[self.axis]
        plane_dist = point[self.axis] - split_plane
        plane_dist2 = plane_dist ** 2

        #从根节点递归向下访问，若point的axis维小于且分点坐标
        #则移动到左子节点，否则移动到右子节点
        if point[self.axis] < split_plane:
            if self.left is not None:
                self.left._search_node(point,k,results,get_dist)
        else:
            if self.right is not None:
                self.right._search_node(point,k,results,get_dist)

        #检查父节点的另一子节点是否存在比当前子节点更近的点
        #判断另一区域是否与当前最近邻的圆相交
        if plane_dist2 < results.max() or results.size() < k:
            if point[self.axis] < self.data[self.axis]:
                if self.right is not None:
                    self.right._search_node(point,k,results,get_dist)
            else:
                if self.left is not None:
                    self.left._search_node(point,k,results,get_dist)
    
    def search_knn(self,point,k,dist=None):
        """返回k个离point最近的点及它们的距离"""

        if dist is None:
            get_dist = lambda n:n.dist(point)
        else:
            gen_dist = lambda n:dist(n.data, point)

        results = BoundedPriorityQueue(k)
        self._search_node(point,k,results,get_dist)

        #将最后的结果按照距离排序
        BY_VALUE = lambda kv: kv[1]
        return sorted(results.items(), key=BY_VALUE)


def test_kdtree():
    pass

if __name__ == '__main__':
    
    startTime = time.time()
    train_data_path = './datasets/Mnist/mnist_train.csv'
    test_data_path = './datasets/Mnist/mnist_test.csv'

    x_train, y_train = loadData(train_data_path)
    x_test, y_test = loadData(test_data_path)

    test_kdtree(x_train, y_train, x_test, y_test, k=20)


    endTime = time.time()

    print('spend time is :{}'.format(endTime - startTime))
